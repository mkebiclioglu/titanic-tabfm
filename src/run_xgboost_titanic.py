"""XGBoost baseline on the Titanic survival task.

This is the strong traditional-ML baseline. We use a reasonable default
configuration and deliberately do NOT tune hyperparameters in this first
version. XGBoost consumes the categorical columns natively via
``enable_categorical=True``, so no one-hot encoding is needed.

Run with:

    python -m src.run_xgboost_titanic
"""

from __future__ import annotations

from xgboost import XGBClassifier

from src.data import RANDOM_STATE, get_titanic_split
from src.metrics import compute_metrics, print_metrics, save_metrics

MODEL_NAME = "xgboost_default"
RESULTS_FILE = "xgboost_titanic_metrics.csv"


def build_model() -> XGBClassifier:
    """A reasonable default XGBoost classifier (no heavy tuning)."""
    return XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        eval_metric="logloss",
        enable_categorical=True,
        tree_method="hist",
        random_state=RANDOM_STATE,
    )


def main() -> dict[str, float]:
    split = get_titanic_split()

    model = build_model()
    model.fit(split.X_train, split.y_train)

    y_pred = model.predict(split.X_test)
    y_proba = model.predict_proba(split.X_test)[:, 1]

    metrics = compute_metrics(split.y_test, y_pred, y_proba)
    print_metrics(MODEL_NAME, metrics)

    path = save_metrics(MODEL_NAME, metrics, RESULTS_FILE)
    print(f"\nSaved metrics to {path}")
    return metrics


if __name__ == "__main__":
    main()
