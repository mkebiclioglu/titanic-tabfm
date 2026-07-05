"""Shared metric computation and result I/O for the benchmark.

Both models report the same set of classification metrics on the same test
set, so this lives in one place.
"""

from __future__ import annotations

import os

import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(_REPO_ROOT, "results")

# Order used everywhere so CSVs and the report line up.
METRIC_NAMES = ["accuracy", "roc_auc", "f1", "precision", "recall"]


def compute_metrics(y_true, y_pred, y_proba) -> dict[str, float]:
    """Compute the classification metrics required by the benchmark.

    ``y_proba`` is the predicted probability of the positive class (survived).
    ROC AUC is the main comparison metric.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "roc_auc": float(roc_auc_score(y_true, y_proba)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }


def save_metrics(model_name: str, metrics: dict[str, float], filename: str) -> str:
    """Write a one-row metrics CSV to the results directory. Returns the path."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = os.path.join(RESULTS_DIR, filename)
    row = {"model": model_name, **{name: metrics[name] for name in METRIC_NAMES}}
    pd.DataFrame([row]).to_csv(path, index=False)
    return path


def print_metrics(model_name: str, metrics: dict[str, float]) -> None:
    print(f"\n{model_name} test metrics:")
    for name in METRIC_NAMES:
        print(f"  {name:10s} {metrics[name]:.4f}")
