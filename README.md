# titanic-tabfm

A small, reproducible benchmark comparing **Google Research's TabFM** (zero-shot
tabular foundation model) against a straightforward **XGBoost** baseline on the
Titanic survival classification task.

It answers one question:

> How does zero-shot TabFM perform against a straightforward XGBoost baseline on
> the Titanic dataset?

Both models use the **same** train/test split, target, and preprocessing.

## Results

On the held-out test set (179 passengers), **zero-shot TabFM beat the default
XGBoost baseline on every metric** — without ever training on the data:

| model | accuracy | ROC AUC | F1 | precision | recall |
|---|---|---|---|---|---|
| XGBoost (default) | 0.788 | 0.807 | 0.708 | 0.754 | 0.667 |
| **TabFM (zero-shot)** | **0.816** | **0.862** | **0.740** | **0.810** | **0.681** |

- **+0.054 ROC AUC** for TabFM (the main comparison metric), the equivalent of a
  meaningfully better-ranked model.
- TabFM was **zero-shot**: no gradient training on Titanic. It read the training
  rows as in-context examples at prediction time.
- It won with a **reduced ensemble** (`n_estimators=8` vs the library default of
  32, for CPU speed), so this likely *understates* TabFM's full configuration.

**Read it honestly:** Titanic is a tiny, easy dataset and this is a single-split
smoke test. It shows TabFM is a credible out-of-the-box tabular predictor here —
it does **not** prove TabFM beats gradient boosting in general. See
[`reports/titanic_benchmark_report.md`](reports/titanic_benchmark_report.md) for
the full write-up and caveats.

## Layout

```
src/
  data.py                  # Titanic loading + one shared reproducible split
  metrics.py               # shared metric computation + result I/O
  run_xgboost_titanic.py   # XGBoost baseline
  run_tabfm_titanic.py     # TabFM zero-shot benchmark
  generate_report.py       # combine metrics + render the Markdown report
  run_titanic_benchmark.py # run everything end-to-end
data/titanic.csv           # cached dataset (committed for reproducibility)
results/                   # per-model + combined metric CSVs
reports/                   # generated Markdown report
```

## Setup

Everything runs inside a local virtual environment.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Confirm the environment is active (the path should point inside `.venv`):

```bash
which python
python --version
```

### macOS note: XGBoost needs OpenMP

On macOS, XGBoost needs the OpenMP runtime:

```bash
brew install libomp
```

### Installing TabFM

TabFM is installed separately from the official Google Research repository (it
is not on PyPI as a pinned dependency). This repo installs the **PyTorch CPU**
backend:

```bash
git clone https://github.com/google-research/tabfm.git third_party/tabfm
python -m pip install -e 'third_party/tabfm[pytorch]'
python -m pip install safetensors   # needed to load the HF weights
```

The pretrained weights (~hundreds of MB) download automatically from the
Hugging Face Hub on the first TabFM run.

## Running the benchmark

Run everything at once:

```bash
python -m src.run_titanic_benchmark
```

Or run each step individually:

```bash
python -m src.run_xgboost_titanic   # -> results/xgboost_titanic_metrics.csv
python -m src.run_tabfm_titanic     # -> results/tabfm_titanic_metrics.csv
python -m src.generate_report       # -> results/combined_titanic_benchmark.csv
                                    #    reports/titanic_benchmark_report.md
```

## Metrics

For each model we report accuracy, ROC AUC, F1, precision, and recall on the
held-out test set. **ROC AUC** is the main comparison metric.

## A note on TabFM ensemble size (speed vs. default)

TabFM's library default ensemble size is `n_estimators=32`. On CPU that takes
tens of minutes for this dataset, so the benchmark uses a smaller ensemble
(`N_ESTIMATORS` in `src/run_tabfm_titanic.py`, default `8`) to stay fast on a
laptop. This is a compute/speed knob, not accuracy tuning. To reproduce TabFM's
true default, set `N_ESTIMATORS = 32` and run on a GPU/TPU. The trade-off is
documented in the report's caveats.

## Reproducibility

- Single stratified split, `random_state=42`, shared by both models via
  `src/data.py`.
- The dataset is cached at `data/titanic.csv` and committed, so the benchmark
  runs offline after the initial TabFM weight download.
