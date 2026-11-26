"""Training entry point and experiment log. Owners: Tech Lead (entry point), Experimentation Engineer (log).

Usage:
    python -m src.train --config configs/baseline_majority.yaml
    python -m src.train --config configs/baseline_logreg.yaml
    python -m src.train --config configs/base.yaml --seed 7     # Keras models, once built
"""
import argparse
import csv
import json
import time
import uuid
from datetime import datetime, timezone

from src import data as D
from src import models as M
from src.config import ROOT, load_config, set_seed
from src.evaluate import classification_metrics

LOG_PATH = ROOT / "experiments" / "experiments.csv"
LOG_FIELDS = ["run_id", "date", "author", "config", "model", "seed", "hyperparameters", "params",
              "train_time_s", "epochs_run", "train_balanced_accuracy", "val_balanced_accuracy",
              "val_macro_f1", "val_accuracy", "val_recall_mel", "notes"]


def log_run(row):
    """Append one row to experiments/experiments.csv (spec 3.5). Test metrics are NEVER logged here."""
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    is_new = not LOG_PATH.exists()
    with LOG_PATH.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def run_baseline(cfg, splits):
    """Fit a baseline on train, score it on train and validation. Returns (model, train_m, val_m)."""
    classes = cfg["data"]["classes"]
    size = cfg["data"]["baseline_image_size"]
    X_tr, y_tr = D.load_flat_pixels(splits["train"], size)
    X_va, y_va = D.load_flat_pixels(splits["val"], size)
    model = M.build_baseline(cfg["model"]["name"], cfg).fit(X_tr, y_tr)
    train_m = classification_metrics(y_tr, model.predict(X_tr), classes)
    val_m = classification_metrics(y_va, model.predict(X_va), classes)
    return model, train_m, val_m


def run(cfg, config_path):
    """Run one configuration and log it."""
    set_seed(cfg["run"]["seed"])
    splits = D.prepare(cfg)
    start = time.time()
    if cfg["model"]["name"] in ("majority", "logreg"):
        _, train_m, val_m = run_baseline(cfg, splits)
        params, epochs = "", ""
    else:
        # TODO(Tech Lead, milestone 3): build model, compile, fit on D.make_dataset(...),
        # EarlyStopping(monitor="val_loss"), class_weight=D.class_weights(...), save curves.
        raise NotImplementedError("Keras training comes in milestone 3")
    run_id = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:4]}"
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
    return val_m


def main():
    """Command-line entry point."""
    p = argparse.ArgumentParser()
    p.add_argument("--config", default="configs/base.yaml")
    p.add_argument("--seed", type=int, help="override run.seed (for the 3-seed final run)")
    args = p.parse_args()
    cfg = load_config(args.config)
    if args.seed is not None:
        cfg["run"]["seed"] = args.seed
    run(cfg, args.config)


if __name__ == "__main__":
    main()
