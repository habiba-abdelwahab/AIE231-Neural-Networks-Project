"""Baselines and neural architectures.

Owners: Baseline & Analysis Engineer (baselines, Model A), Model Architect (Models B and C).
"""
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from tensorflow import keras


# ---------------------------------------------------------------- baselines (spec 3.3)

def build_baseline(name, cfg):
    """Return an (unfitted) scikit-learn baseline.

    majority: always predicts the most common class (nv). Its balanced accuracy is 1/7 = 0.143,
              which is the "do nothing" score every model must beat.
    logreg:   logistic regression on 32x32x3 flattened pixels. The StandardScaler is inside
              the pipeline, so `.fit(X_train)` fits it on the training set only (no leakage).
              class_weight="balanced" handles the imbalance.
    """
    seed = cfg["run"]["seed"]
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "logreg":
        weight = "balanced" if cfg["data"]["class_weights"] else None
        return make_pipeline(StandardScaler(),
                             LogisticRegression(max_iter=1000, C=0.01, class_weight=weight,
                                                random_state=seed))  # GUESS: C=0.01, tune on val
    raise ValueError(f"unknown baseline: {name}")


# ---------------------------------------------------------------- neural models (spec 3.4)
# TODO(Model A, Baseline & Analysis Engineer): custom CNN from scratch.
# TODO(Model B, Model Architect): VGG16 or ResNet50 (ImageNet weights) frozen + new classifier.
# TODO(Model C, Model Architect): same as B with the last block unfrozen and a small learning rate.
# For each: output softmax with 7 units + sparse_categorical_crossentropy, Dropout,
# hand parameter counts (Conv2D = (F*F*C_in + 1) * filters) checked against model.summary().

def build_model(cfg):
    """Return the Keras model named in cfg['model']['name']."""
    raise NotImplementedError(f"TODO: build {cfg['model']['name']}")


def build_optimizer(cfg):
    """SGD, SGD with momentum or Adam (spec 3.4 asks to compare all three)."""
    t = cfg["train"]
    if t["optimizer"] == "sgd":
        return keras.optimizers.SGD(learning_rate=t["learning_rate"])
    if t["optimizer"] == "sgd_momentum":
        return keras.optimizers.SGD(learning_rate=t["learning_rate"], momentum=t["momentum"])
    if t["optimizer"] == "adam":
        return keras.optimizers.Adam(learning_rate=t["learning_rate"])
    raise ValueError(f"unknown optimizer: {t['optimizer']}")
