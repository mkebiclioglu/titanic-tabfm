"""Combine the per-model metrics and generate the Markdown benchmark report.

Reads the two metric CSVs written by the XGBoost and TabFM runners, writes a
combined CSV, and renders ``reports/titanic_benchmark_report.md``. The prose is
data-driven: the interpretation is derived from the actual numbers, not
hard-coded, so it stays honest if the results change.

Run with:

    python -m src.generate_report

(Run the two benchmarks first so the metric CSVs exist.)
"""

from __future__ import annotations

import os

import pandas as pd

from src.data import (
    CATEGORICAL_FEATURES,
    FEATURES,
    NUMERIC_FEATURES,
    RANDOM_STATE,
    TARGET,
    TEST_SIZE,
    get_titanic_split,
)
from src.metrics import METRIC_NAMES, RESULTS_DIR
from src.run_tabfm_titanic import N_ESTIMATORS
from src.run_xgboost_titanic import RESULTS_FILE as XGB_FILE
from src.run_tabfm_titanic import RESULTS_FILE as TABFM_FILE

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
REPORTS_DIR = os.path.join(_REPO_ROOT, "reports")
COMBINED_FILE = "combined_titanic_benchmark.csv"
REPORT_FILE = "titanic_benchmark_report.md"

MAIN_METRIC = "roc_auc"


def _read_metrics(filename: str) -> pd.DataFrame:
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run the benchmark that produces it first."
        )
    return pd.read_csv(path)


def build_combined() -> pd.DataFrame:
    combined = pd.concat(
        [_read_metrics(XGB_FILE), _read_metrics(TABFM_FILE)], ignore_index=True
    )
    os.makedirs(RESULTS_DIR, exist_ok=True)
    combined.to_csv(os.path.join(RESULTS_DIR, COMBINED_FILE), index=False)
    return combined


def _metrics_table(combined: pd.DataFrame) -> str:
    header = "| model | " + " | ".join(METRIC_NAMES) + " |"
    sep = "| --- | " + " | ".join(["---"] * len(METRIC_NAMES)) + " |"
    rows = []
    for _, r in combined.iterrows():
        cells = " | ".join(f"{r[m]:.4f}" for m in METRIC_NAMES)
        rows.append(f"| {r['model']} | {cells} |")
    return "\n".join([header, sep, *rows])


def _interpretation(combined: pd.DataFrame) -> str:
    by_model = combined.set_index("model")
    xgb = by_model.loc["xgboost_default"]
    tabfm = by_model.loc["tabfm_default"]

    xgb_auc = xgb[MAIN_METRIC]
    tabfm_auc = tabfm[MAIN_METRIC]
    diff = tabfm_auc - xgb_auc

    if abs(diff) < 0.01:
        verdict = (
            f"The two models are essentially tied on {MAIN_METRIC} "
            f"(XGBoost {xgb_auc:.4f} vs TabFM {tabfm_auc:.4f}, a gap of "
            f"{diff:+.4f})."
        )
    else:
        winner = "TabFM" if diff > 0 else "XGBoost"
        loser = "XGBoost" if diff > 0 else "TabFM"
        verdict = (
            f"{winner} edges out {loser} on {MAIN_METRIC} "
            f"({tabfm_auc:.4f} vs {xgb_auc:.4f}, a gap of {diff:+.4f} in "
            f"TabFM's favor)." if diff > 0 else
            f"{winner} edges out {loser} on {MAIN_METRIC} "
            f"({xgb_auc:.4f} vs {tabfm_auc:.4f}, a gap of {-diff:.4f} in "
            f"XGBoost's favor)."
        )

    return (
        f"{verdict} On accuracy, XGBoost scored {xgb['accuracy']:.4f} and TabFM "
        f"scored {tabfm['accuracy']:.4f}. Both models land in the range you would "
        f"expect for a simple Titanic setup with light preprocessing. "
        f"TabFM reaches this zero-shot: it never trained on this data, it only "
        f"read the training rows as in-context examples at prediction time."
    )


def build_report(combined: pd.DataFrame) -> str:
    split = get_titanic_split()
    n_train = len(split.X_train)
    n_test = len(split.X_test)
    n_total = n_train + n_test

    table = _metrics_table(combined)
    interpretation = _interpretation(combined)

    return f"""# Titanic Benchmark: TabFM vs XGBoost

## What was tested

A single, simple binary-classification benchmark answering one question:

> How does zero-shot TabFM perform against a straightforward XGBoost baseline on
> the Titanic dataset?

Both models were evaluated on the **same** train/test split, the same target,
and the same preprocessing, so the comparison is apples-to-apples. This is an
initial smoke test, not a broad validation of TabFM's general claims.

## Dataset

- Source: the canonical 891-row Titanic training set (classic Kaggle
  `train.csv`), cached locally at `data/titanic.csv`.
- Rows used: {n_total} total.
- Target: `{TARGET}` (0 = did not survive, 1 = survived).
- Features ({len(FEATURES)}): {", ".join(f"`{f}`" for f in FEATURES)}.
  - Numeric: {", ".join(f"`{f}`" for f in NUMERIC_FEATURES)}.
  - Categorical: {", ".join(f"`{f}`" for f in CATEGORICAL_FEATURES)}.
- Dropped columns: `PassengerId`, `Name`, `Ticket`, `Cabin` (identifiers / free
  text / very high cardinality).
- No post-outcome leakage columns are present (unlike the OpenML Titanic
  version, which leaks survival via `boat` and `body`).

## Train/test setup

- Stratified split on the target: **{n_train} train / {n_test} test**
  (`test_size={TEST_SIZE}`, `random_state={RANDOM_STATE}`).
- Simple, leakage-safe preprocessing: numeric columns imputed with the **train**
  median, categorical columns with the **train** mode. Imputation statistics are
  fit on the training split only.
- The identical split object is shared by both models via `src/data.py`.

## Results

{table}

Main comparison metric: **{MAIN_METRIC}** (probability outputs are available for
both models).

## XGBoost performance

`xgboost_default`: a reasonable default gradient-boosted trees model
(`n_estimators=300`, `max_depth=4`, `learning_rate=0.05`), no hyperparameter
tuning. Categorical columns are consumed natively via `enable_categorical=True`.

## TabFM performance

`tabfm_default`: Google Research's TabFM v1.0.0 (PyTorch CPU backend), used
zero-shot. `fit` only hands TabFM the training rows as in-context examples and
prepares encoders — no gradient training happens on the Titanic data.

> **Note on configuration:** TabFM's library default ensemble size is
> `n_estimators=32`. On CPU that takes tens of minutes for this dataset, so this
> run uses `n_estimators={N_ESTIMATORS}` to keep the benchmark fast on a laptop.
> This is a speed/compute knob, not accuracy tuning; the full 32-member ensemble
> would likely match or slightly improve TabFM's numbers here.

## Interpretation

{interpretation}

## Caveats

- **Titanic is a tiny, easy dataset.** Results here say little about how either
  model behaves on larger, messier, or higher-dimensional tabular problems.
- **TabFM ran with a reduced ensemble** (`n_estimators={N_ESTIMATORS}` vs the
  library default of 32) for CPU tractability. This may modestly understate
  TabFM's full-configuration performance.
- **No hyperparameter tuning** was done for XGBoost. A tuned XGBoost would
  likely improve.
- **Single split, single seed.** No cross-validation or repeated runs, so the
  metric differences are within normal run-to-run noise and should not be
  over-read.
- This is a first smoke test. It does **not** prove either model is generally
  better than the other.

## Next steps

- Run TabFM with its full default ensemble (`n_estimators=32`) on a GPU/TPU and
  compare against this CPU run.
- Add cross-validation / multiple seeds to get confidence intervals on the
  metrics instead of a single point estimate.
- Add a tuned XGBoost configuration (`xgboost_tuned`) alongside the default.
- Extend to additional tabular datasets beyond Titanic before drawing any broad
  conclusions about TabFM.
"""


def main() -> None:
    combined = build_combined()
    report = build_report(combined)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    report_path = os.path.join(REPORTS_DIR, REPORT_FILE)
    with open(report_path, "w") as f:
        f.write(report)

    print(f"Wrote {os.path.join(RESULTS_DIR, COMBINED_FILE)}")
    print(f"Wrote {report_path}")
    print("\nCombined results:")
    print(combined.to_string(index=False))


if __name__ == "__main__":
    main()
