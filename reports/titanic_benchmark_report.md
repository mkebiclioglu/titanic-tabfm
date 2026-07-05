# Titanic Benchmark: TabFM vs XGBoost

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
- Rows used: 891 total.
- Target: `Survived` (0 = did not survive, 1 = survived).
- Features (7): `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`, `Embarked`.
  - Numeric: `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`.
  - Categorical: `Sex`, `Embarked`.
- Dropped columns: `PassengerId`, `Name`, `Ticket`, `Cabin` (identifiers / free
  text / very high cardinality).
- No post-outcome leakage columns are present (unlike the OpenML Titanic
  version, which leaks survival via `boat` and `body`).

## Train/test setup

- Stratified split on the target: **712 train / 179 test**
  (`test_size=0.2`, `random_state=42`).
- Simple, leakage-safe preprocessing: numeric columns imputed with the **train**
  median, categorical columns with the **train** mode. Imputation statistics are
  fit on the training split only.
- The identical split object is shared by both models via `src/data.py`.

## Results

| model | accuracy | roc_auc | f1 | precision | recall |
| --- | --- | --- | --- | --- | --- |
| xgboost_default | 0.7877 | 0.8074 | 0.7077 | 0.7541 | 0.6667 |
| tabfm_default | 0.8156 | 0.8618 | 0.7402 | 0.8103 | 0.6812 |

Main comparison metric: **roc_auc** (probability outputs are available for
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
> run uses `n_estimators=8` to keep the benchmark fast on a laptop.
> This is a speed/compute knob, not accuracy tuning; the full 32-member ensemble
> would likely match or slightly improve TabFM's numbers here.

## Interpretation

TabFM edges out XGBoost on roc_auc (0.8618 vs 0.8074, a gap of +0.0543 in TabFM's favor). On accuracy, XGBoost scored 0.7877 and TabFM scored 0.8156. Both models land in the range you would expect for a simple Titanic setup with light preprocessing. TabFM reaches this zero-shot: it never trained on this data, it only read the training rows as in-context examples at prediction time.

## Caveats

- **Titanic is a tiny, easy dataset.** Results here say little about how either
  model behaves on larger, messier, or higher-dimensional tabular problems.
- **TabFM ran with a reduced ensemble** (`n_estimators=8` vs the
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
