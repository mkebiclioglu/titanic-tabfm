"""TabFM zero-shot benchmark on the Titanic survival task.

TabFM (Google Research's tabular foundation model) makes predictions via
in-context learning: ``fit`` simply hands it the training rows as context and
prepares encoders/scalers -- no gradient training happens on our data. This is
the "zero-shot / no fine-tuning" claim we are testing.

We use TabFM's default hyperparameters and the PyTorch CPU backend, with one
deliberate exception: the ensemble size ``n_estimators``. TabFM's library
default is 32, which means 32 forward passes over the full training context per
prediction. On CPU that takes tens of minutes for this dataset. We lower it to
``N_ESTIMATORS`` below so the benchmark runs in a couple of minutes on a laptop.

This is a compute/speed knob, not accuracy tuning, but it is still a deviation
from TabFM out-of-the-box: a larger ensemble would likely match or slightly
improve TabFM's numbers. This trade-off is called out in the report's caveats.
To reproduce TabFM's true default, set ``N_ESTIMATORS = 32`` (slow on CPU; use
a GPU/TPU for the full ensemble).

The same shared train/test split, target, and preprocessing from ``src.data``
are used as for the XGBoost baseline so the comparison is apples-to-apples.

The first run downloads the pretrained weights from the Hugging Face Hub and
compiles the model, which can take a few minutes.

Run with:

    python -m src.run_tabfm_titanic
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.data import get_titanic_split
from src.metrics import compute_metrics, print_metrics, save_metrics

MODEL_NAME = "tabfm_default"
RESULTS_FILE = "tabfm_titanic_metrics.csv"

# Ensemble size. TabFM's library default is 32; we use a smaller ensemble so the
# CPU run is tractable. See module docstring and the report caveats.
N_ESTIMATORS = 8


def _to_tabfm_frame(X: pd.DataFrame, categorical_features: list[str]) -> pd.DataFrame:
    """Present columns the way TabFM expects: numerics as float, categoricals
    as plain strings (object dtype). TabFM infers column types from dtype."""
    X = X.copy()
    for col in X.columns:
        if col in categorical_features:
            X[col] = X[col].astype(str)
        else:
            X[col] = X[col].astype(float)
    return X


def _load_classifier():
    """Load the TabFM v1.0.0 classifier (PyTorch backend, default config)."""
    import tabfm

    if not hasattr(tabfm, "tabfm_v1_0_0_pytorch"):
        raise RuntimeError(
            "TabFM PyTorch backend is not available. Install it with "
            "`pip install -e 'third_party/tabfm[pytorch]'` (see README)."
        )

    model = tabfm.tabfm_v1_0_0_pytorch.load(model_type="classification")
    return tabfm.TabFMClassifier(model=model, n_estimators=N_ESTIMATORS)


def main() -> dict[str, float]:
    split = get_titanic_split()

    X_train = _to_tabfm_frame(split.X_train, split.categorical_features)
    X_test = _to_tabfm_frame(split.X_test, split.categorical_features)
    y_train = split.y_train.to_numpy()

    clf = _load_classifier()
    clf.fit(X_train, y_train)

    proba = np.asarray(clf.predict_proba(X_test))
    # Probability of the positive class (survived == 1).
    pos_idx = list(clf.classes_).index(1)
    y_proba = proba[:, pos_idx]
    y_pred = clf.predict(X_test)

    metrics = compute_metrics(split.y_test, y_pred, y_proba)
    print_metrics(MODEL_NAME, metrics)

    path = save_metrics(MODEL_NAME, metrics, RESULTS_FILE)
    print(f"\nSaved metrics to {path}")
    return metrics


if __name__ == "__main__":
    main()
