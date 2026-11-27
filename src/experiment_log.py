"""Experiment log and learning curves (spec 3.5). Owner: Experimentation Engineer.

Every run is one row in a CSV. Each member writes to their OWN file
(experiments/runs_<github-login>.csv), so two people never edit the same file and there
are no merge conflicts. `load_all_runs()` joins all the files into one experiment table.
Test-set metrics are NEVER logged here: model selection uses validation only.
"""
import csv
import json

import pandas as pd

from src.config import ROOT

EXP_DIR = ROOT / "experiments"
CURVES_DIR = EXP_DIR / "curves"
LOG_FIELDS = ["run_id", "date", "author", "config", "model", "seed", "hyperparameters", "params",
              "train_time_s", "epochs_run", "train_balanced_accuracy", "val_balanced_accuracy",
              "val_macro_f1", "val_accuracy", "val_recall_mel", "notes"]


def log_run(row):
    """Append one run to experiments/runs_<author>.csv (creates the file the first time)."""
    path = EXP_DIR / f"runs_{row['author']}.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    is_new = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerow({k: row.get(k, "") for k in LOG_FIELDS})


def load_all_runs():
    """One table with every member's runs, newest last."""
    files = sorted(EXP_DIR.glob("runs_*.csv"))
    if not files:
        return pd.DataFrame(columns=LOG_FIELDS)
    return pd.concat([pd.read_csv(f) for f in files]).sort_values("date").reset_index(drop=True)


def save_curves(run_id, history, **extra):
    """Save the per-epoch history of a Keras run as experiments/curves/<run_id>.json."""
    CURVES_DIR.mkdir(parents=True, exist_ok=True)
    (CURVES_DIR / f"{run_id}.json").write_text(json.dumps({**extra, **history}))


def load_curves(run_id):
    """Read back a history saved by save_curves."""
    return json.loads((CURVES_DIR / f"{run_id}.json").read_text())


def plot_curves(run_ids, labels=None):
    """Training vs validation loss and accuracy for one or more runs, side by side.

    How to read it: if training loss keeps falling while validation loss rises, the model
    is OVERFITTING (memorising the training images). If both stay high, it is UNDERFITTING.
    A dotted line marks where Model C's fine-tuning phase starts.
    """
    import matplotlib.pyplot as plt
    labels = labels or run_ids
    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    for run_id, label in zip(run_ids, labels):
        h = load_curves(run_id)
        for ax, metric in zip(axes, ["loss", "accuracy"]):
            line, = ax.plot(h[metric], label=f"{label} train")
            ax.plot(h[f"val_{metric}"], "--", color=line.get_color(), label=f"{label} val")
            if "phase2_starts_at_epoch" in h:
                ax.axvline(h["phase2_starts_at_epoch"], color=line.get_color(), linestyle=":")
    for ax, metric in zip(axes, ["loss", "accuracy"]):
        ax.set_xlabel("epoch"); ax.set_ylabel(metric); ax.legend(fontsize=8)
    plt.tight_layout()
    return fig
