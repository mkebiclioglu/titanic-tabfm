"""Titanic data loading and a single reproducible train/test split.

Both the XGBoost and TabFM benchmarks import from here so that they use the
exact same rows, the same target, and the same simple preprocessing. This is
what makes the comparison fair.

Design choices (kept deliberately simple):

* Source: the canonical 891-row Titanic training set (the classic Kaggle
  ``train.csv``). It is downloaded once and cached to ``data/titanic.csv``.
* Target: ``Survived`` (0 = died, 1 = survived).
* Features: ``Pclass, Sex, Age, SibSp, Parch, Fare, Embarked``.
  We drop ``PassengerId``, ``Name``, ``Ticket`` and ``Cabin`` because they are
  identifiers / free text / very high-cardinality and are not needed for a
  simple first benchmark.
* Leakage: this dataset contains no post-outcome columns (unlike the OpenML
  version, which leaks survival through ``boat`` and ``body``). Imputation
  statistics are computed on the TRAIN split only and then applied to test.
"""

from __future__ import annotations

import os
import ssl
import urllib.request
from dataclasses import dataclass
from io import StringIO

import pandas as pd
from sklearn.model_selection import train_test_split

# --- Reproducibility knobs (shared by both models) -------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "Survived"

NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare"]
CATEGORICAL_FEATURES = ["Sex", "Embarked"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

_DATA_URL = (
    "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
)
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_CACHE_PATH = os.path.join(_REPO_ROOT, "data", "titanic.csv")


@dataclass
class TitanicSplit:
    """A single train/test split plus the column-type metadata models need."""

    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    numeric_features: list[str]
    categorical_features: list[str]


def load_titanic_raw() -> pd.DataFrame:
    """Return the raw Titanic dataframe, downloading and caching it if needed."""
    if os.path.exists(_CACHE_PATH):
        return pd.read_csv(_CACHE_PATH)

    os.makedirs(os.path.dirname(_CACHE_PATH), exist_ok=True)
    # Use certifi's CA bundle so the download works on machines where the
    # system Python cannot verify TLS certificates (common on macOS).
    try:
        import certifi

        context = ssl.create_default_context(cafile=certifi.where())
    except ImportError:
        context = ssl.create_default_context()

    with urllib.request.urlopen(_DATA_URL, context=context) as response:
        csv_text = response.read().decode("utf-8")

    df = pd.read_csv(StringIO(csv_text))
    df.to_csv(_CACHE_PATH, index=False)
    return df


def _preprocess(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Select features/target and coerce dtypes. No imputation here."""
    X = df[FEATURES].copy()
    y = df[TARGET].astype(int)

    for col in NUMERIC_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype("string")

    return X, y


def _impute(
    X_train: pd.DataFrame, X_test: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Simple, leakage-safe imputation using train statistics only.

    Numeric columns are filled with the train median; categorical columns with
    the train mode. Categoricals are returned as pandas ``category`` dtype so
    that XGBoost (``enable_categorical=True``) and TabFM both see clean, typed
    columns.
    """
    X_train = X_train.copy()
    X_test = X_test.copy()

    for col in NUMERIC_FEATURES:
        median = X_train[col].median()
        X_train[col] = X_train[col].fillna(median)
        X_test[col] = X_test[col].fillna(median)

    for col in CATEGORICAL_FEATURES:
        mode = X_train[col].mode(dropna=True)
        fill = mode.iloc[0] if not mode.empty else "Unknown"
        X_train[col] = X_train[col].fillna(fill).astype("category")
        # Match train categories so both frames share the same category set.
        X_test[col] = (
            X_test[col].fillna(fill).astype("category")
            .cat.set_categories(X_train[col].cat.categories)
        )

    return X_train, X_test


def get_titanic_split(
    test_size: float = TEST_SIZE, random_state: int = RANDOM_STATE
) -> TitanicSplit:
    """Build the single shared, stratified train/test split for both models."""
    df = load_titanic_raw()
    X, y = _preprocess(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    X_train, X_test = _impute(X_train, X_test)

    return TitanicSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        numeric_features=list(NUMERIC_FEATURES),
        categorical_features=list(CATEGORICAL_FEATURES),
    )


if __name__ == "__main__":
    split = get_titanic_split()
    print("Train shape:", split.X_train.shape)
    print("Test shape:", split.X_test.shape)
    print("Target balance (train):")
    print(split.y_train.value_counts(normalize=True).round(3).to_string())
    print("\nFeature dtypes:")
    print(split.X_train.dtypes.to_string())
    print("\nMissing values after imputation:",
          int(split.X_train.isna().sum().sum() + split.X_test.isna().sum().sum()))
