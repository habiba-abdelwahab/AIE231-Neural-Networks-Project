"""Training entry point: runs one experiment from a config file. Owner: Tech Lead.

Usage (from the project root):
    python -m src.train --config configs/baseline_logreg.yaml --author <github-login>
    python -m src.train --config configs/baseline_majority.yaml --author <github-login>
"""
import argparse
import json
import time
import uuid
from datetime import datetime, timezone

from sklearn.metrics import recall_score

from src import data as D
from src import models as M
from src.config import load_config, set_seed
from src.evaluate import classification_metrics
from src.experiment_log import log_run


def run_baseline(cfg, splits):
    """Fit a baseline on train, score it on train and validation. Returns (model, train_m, val_m)."""
    classes = cfg["data"]["classes"]
    size = cfg["data"]["baseline_image_size"]
    X_tr, y_tr = D.load_flat_pixels(splits["train"], size)
    X_va, y_va = D.load_flat_pixels(splits["val"], size)
    model = M.build_baseline(cfg["model"]["name"], cfg).fit(X_tr, y_tr)
    train_m = classification_metrics(y_tr, model.predict(X_tr), classes)
    val_pred = model.predict(X_va)
    val_m = classification_metrics(y_va, val_pred, classes)
    # Per-class recall on validation: which classes the baseline finds and which it misses.
    recalls = recall_score(y_va, val_pred, labels=range(len(classes)), average=None, zero_division=0)
    print("val recall per class:", "  ".join(f"{c}={r:.2f}" for c, r in zip(classes, recalls)))
    return model, train_m, val_m


def run(cfg, config_path):
    """Run one configuration, log it."""
    set_seed(cfg["run"]["seed"])
    splits = D.prepare(cfg)
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
    start = time.time()
    if cfg["model"]["name"] in ("majority", "logreg"):
        _, train_m, val_m = run_baseline(cfg, splits)
        params, epochs = "", ""
    else:
        raise NotImplementedError("Keras models come in milestone 3")
    log_run({
        "run_id": run_id, "date": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "author": cfg["run"]["author"], "config": str(config_path), "model": cfg["model"]["name"],
        "seed": cfg["run"]["seed"], "hyperparameters": json.dumps({**cfg["model"], **cfg["train"]}),
        "params": params, "train_time_s": round(time.time() - start, 1), "epochs_run": epochs,
        "train_balanced_accuracy": round(train_m["balanced_accuracy"], 4),
        "val_balanced_accuracy": round(val_m["balanced_accuracy"], 4),
        "val_macro_f1": round(val_m["macro_f1"], 4), "val_accuracy": round(val_m["accuracy"], 4),
        "val_recall_mel": round(val_m["recall_mel"], 4), "notes": cfg["run"].get("notes", ""),
    })
    print(f"[{run_id}] {cfg['model']['name']}  val: " +
          "  ".join(f"{k}={v:.3f}" for k, v in val_m.items()))
    return run_id, val_m


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--author", help="your GitHub login, written in the experiment log")
    p.add_argument("--seed", type=int, help="override run.seed (for the 3-seed final run)")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.author:
        cfg["run"]["author"] = args.author
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    run(cfg, args.config)


if __name__ == "__main__":
    main()
