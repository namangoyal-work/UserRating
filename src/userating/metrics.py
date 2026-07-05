"""Evaluation helpers. The course metric is (f1_micro + f1_macro) / 2; the rest is
diagnostics for where the model actually loses points (the rare 1-3 star classes)."""

import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, mean_absolute_error


def eval_metrics(true, preds):
    f1_micro = f1_score(true, preds, average='micro')
    f1_macro = f1_score(true, preds, average='macro')
    print(f"    F1 micro: {f1_micro}")
    print(f"    F1 macro: {f1_macro}")
    print(f"    Final score: {(f1_micro + f1_macro) / 2}")
    return f1_micro, f1_macro


def eval_report(true, preds):
    """eval_metrics plus per-class F1, MAE (ratings are ordinal) and the confusion matrix."""
    f1_micro, f1_macro = eval_metrics(true, preds)

    labels = [1, 2, 3, 4, 5]
    per_class = f1_score(true, preds, average=None, labels=labels)
    mae = mean_absolute_error(true, preds)

    print(f"    MAE (stars): {mae}")
    print("    Per-class F1: " + "  ".join(f"{s}*={f:.3f}" for s, f in zip(labels, per_class)))

    cm = confusion_matrix(true, preds, labels=labels)
    print("    Confusion matrix (rows=true 1..5, cols=pred 1..5):")
    for row in cm:
        print("      " + " ".join(f"{c:6d}" for c in row))

    return {
        'f1_micro': f1_micro,
        'f1_macro': f1_macro,
        'final': (f1_micro + f1_macro) / 2,
        'mae': mae,
        'per_class_f1': np.asarray(per_class),
        'confusion': cm,
    }
