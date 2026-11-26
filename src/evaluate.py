"""Metrics, confusion matrix, ROC and error analysis. Owner: Evaluation & Deployment Engineer.

Primary metric: balanced accuracy (= the average of the 7 per-class recalls).
Why: plain accuracy is misleading here, because predicting "nv" for everything already
gives about 67% accuracy. Balanced accuracy gives that useless model only 1/7 = 14.3%.
We also watch melanoma recall on its own, because a missed melanoma is the costliest error.
The test set is used ONCE, at the end, for the final model and the baselines (spec 3.6).
"""
import numpy as np
from sklearn.metrics import (accuracy_score, balanced_accuracy_score, classification_report,
                             confusion_matrix, f1_score, recall_score, roc_auc_score)


def classification_metrics(y_true, y_pred, classes, watch_class="mel"):
    """Metrics we log for every run on train and validation."""
    per_class_recall = recall_score(y_true, y_pred, labels=range(len(classes)),
                                    average=None, zero_division=0)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        f"recall_{watch_class}": float(per_class_recall[classes.index(watch_class)]),
    }


def full_report(y_true, proba, classes):
    """Final test report: per-class precision/recall/F1, confusion matrix, one-vs-rest ROC-AUC."""
    y_pred = proba.argmax(axis=1)
    return {
        **classification_metrics(y_true, y_pred, classes),
        "roc_auc_ovr": float(roc_auc_score(y_true, proba, multi_class="ovr",
                                           labels=range(len(classes)))),
        "per_class": classification_report(y_true, y_pred, labels=range(len(classes)),
                                            target_names=classes, output_dict=True,
                                            zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=range(len(classes))).tolist(),
    }


def worst_errors(df, proba, classes, n=20):
    """The n wrong predictions the model was MOST confident about (for spec 3.6 error analysis).

    TODO: look at each image and add a `failure_category` column by hand
    (similar classes, poor image quality, rare class, label noise, unusual pattern).
    """
    y_true = df["label"].to_numpy()
    y_pred = proba.argmax(axis=1)
    wrong = np.where(y_pred != y_true)[0]
    worst = wrong[np.argsort(-proba[wrong].max(axis=1))][:n]
    out = df.iloc[worst][["image_id", "lesion_id", "dx", "path"]].copy()
    out["pred"] = [classes[i] for i in y_pred[worst]]
    out["confidence"] = proba[worst].max(axis=1)
    return out


# TODO: plot_confusion_matrix, plot_roc, plot_learning_curves, and a CLI that
# reproduces the final test numbers with one command.
