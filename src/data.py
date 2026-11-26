"""Loading, cleaning, splitting and preprocessing for HAM10000. Owner: Data Engineer.

Two leakage rules from spec 3.2, both enforced here:
1. Images of the same lesion (same lesion_id) never end up in different splits.
2. Anything "learned" from data (pixel scaler, class weights) uses the training split only.
"""

import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from src.augment import build_augmentation
from src.config import ROOT


# ---------------------------------------------------------------- loading and cleaning

def load_metadata(cfg):
    """Read HAM10000_metadata.csv and add two columns: `path` (image file) and `label` (0..6)."""
    root = ROOT / cfg["data"]["root"]
    df = pd.read_csv(root / cfg["data"]["metadata_file"])
    # The images are split over two folders; build a lookup image_id -> file path.
    paths = {p.stem: str(p) for p in root.glob("HAM10000_images_part_*/*.jpg")}
    df["path"] = df["image_id"].map(paths)
    classes = cfg["data"]["classes"]
    df["label"] = df["dx"].map({name: i for i, name in enumerate(classes)})
    return df


def clean(df):
    """Fix the known data problems and print what was changed.

    - Rows whose image file is missing are dropped (cannot train on them).
    - Exact duplicate rows (same image_id twice) are dropped.
    - age: a few values are missing. We only use images for the models, so we keep the
      rows and fill age with the median just for the EDA plots.
    - sex / localization contain the word "unknown"; we keep it as its own category.
    """
    before = len(df)
    df = df.dropna(subset=["path"])
    df = df.drop_duplicates(subset="image_id")
    df = df.assign(age=df["age"].fillna(df["age"].median()))
    print(f"clean(): kept {len(df)} of {before} rows")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------- splitting

def split(df, cfg):
    """Stratified 70/15/15 split done at LESION level, then mapped back to images.

    Why: one lesion can have several photos. If photo 1 is in train and photo 2 in test,
    the model is tested on something it has almost seen, and the score is too optimistic.
    Steps:
      1. Make a table with one row per lesion and its diagnosis.
      2. Split the lesions with train_test_split(stratify=diagnosis), twice: first test, then val.
      3. Each image goes to the split of its lesion.
    """
    seed = cfg["run"]["seed"]
    val_frac, test_frac = cfg["data"]["split"]["val"], cfg["data"]["split"]["test"]
    group = cfg["data"]["group_col"]

    lesions = df.groupby(group)["dx"].first().reset_index()   # one row per lesion
    trval, test = train_test_split(lesions, test_size=test_frac,
                                   stratify=lesions["dx"], random_state=seed)
    # val is 15% of everything = 15/85 of what is left after removing the test set
    train, val = train_test_split(trval, test_size=val_frac / (1 - test_frac),
                                  stratify=trval["dx"], random_state=seed)

    out = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        out[name] = df[df[group].isin(part[group])].reset_index(drop=True)
    check_no_leakage(out, group)
    return out


def check_no_leakage(splits, group):
    """Fail loudly if any lesion or image appears in two splits."""
    for col in [group, "image_id"]:
        tr, va, te = (set(splits[s][col]) for s in ["train", "val", "test"])
        assert not (tr & va) and not (tr & te) and not (va & te), f"leakage on {col}"


def split_summary(splits, classes):
    """Table of class percentages per split, to show the stratification worked."""
    table = pd.DataFrame({name: part["dx"].value_counts(normalize=True).reindex(classes)
                          for name, part in splits.items()}).fillna(0).round(3)
    table.loc["n_images"] = [len(p) for p in splits.values()]
    return table


# ---------------------------------------------------------------- imbalance

def class_weights(train_df, n_classes):
    """'Balanced' weights from the TRAINING labels: weight_c = n_samples / (n_classes * n_c).

    A rare class (e.g. df) gets a big weight, the common class (nv) a small one, so each
    class contributes about equally to the loss.
    """
    y = train_df["label"].to_numpy()
    w = compute_class_weight("balanced", classes=np.arange(n_classes), y=y)
    return {i: float(v) for i, v in enumerate(w)}


# ---------------------------------------------------------------- image pipeline (Keras models)

def load_image(path, size):
    """Read one JPEG, resize to size x size, scale pixels from [0, 255] to [0, 1]."""
    img = tf.io.decode_jpeg(tf.io.read_file(path), channels=3)
    img = tf.image.resize(img, (size, size))
    return img / 255.0


def make_dataset(df, cfg, training):
    """tf.data pipeline: (image, label) batches. Shuffle + augment only when training=True."""
    size, batch = cfg["data"]["image_size"], cfg["train"]["batch_size"]
    ds = tf.data.Dataset.from_tensor_slices((df["path"].to_numpy(), df["label"].to_numpy()))
    if training:
        ds = ds.shuffle(len(df), seed=cfg["run"]["seed"])
    ds = ds.map(lambda p, y: (load_image(p, size), y), num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch)
    if training:
        augment = build_augmentation(cfg)
        ds = ds.map(lambda x, y: (augment(x, training=True), y),
                    num_parallel_calls=tf.data.AUTOTUNE)
    return ds.prefetch(tf.data.AUTOTUNE)


# ---------------------------------------------------------------- flat pixels (baselines)

def load_flat_pixels(df, size):
    """Load every image as a small size x size x 3 array and flatten it to one row of numbers.

    Used only by the logistic-regression baseline. Returns X (n, size*size*3) in [0, 1] and y.
    """
    X = np.stack([np.asarray(Image.open(p).convert("RGB").resize((size, size)),
                             dtype=np.float32).ravel() / 255.0 for p in df["path"]])
    return X, df["label"].to_numpy()


def prepare(cfg):
    """Everything the training code needs: cleaned metadata split into train/val/test."""
    return split(clean(load_metadata(cfg)), cfg)
