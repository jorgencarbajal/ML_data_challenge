from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import f1_score, precision_score, recall_score
from pandas.api.types import is_numeric_dtype
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier


DATA_PATH = Path("data/v1.csv")
TARGET = "injury_present"
RANDOM_SEED = 13
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20
TREE_MAX_DEPTH = 8
TREE_MIN_SAMPLES_LEAF = 100


EXCLUDED_COLUMNS = {
    "target": [
        "injury_present",
    ],
    "identifier": [
        "Report Key",
    ],
    "date or reporting period": [
        "Date",
        "year",
    ],
    "injury or fatality outcome": [
        "Crossing Users Injured",
        "Crossing Users Killed",
        "Employees Injured",
        "Employees Killed",
        "Passengers Injured",
        "Passengers Killed",
        "Total Injured Form 55A",
        "Total Injured Form 57",
        "Total Killed Form 55A",
        "Total Killed Form 57",
        "fatality_present",
    ],
    "post-incident consequence": [
        "Vehicle Damage Cost",
    ],
}


EXCLUSION_NOTES = {
    "Date": "Exact date is too specific and can encode reporting artifacts.",
    "year": "Kept for audit only, not as a baseline model input.",
    "fatality_present": "This is another outcome variable.",
    "Vehicle Damage Cost": "Damage cost happens after the incident.",
}


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, low_memory=False)


def get_excluded_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    excluded = {
        reason: [column for column in columns if column in df.columns]
        for reason, columns in EXCLUDED_COLUMNS.items()
    }

    constant_columns = [
        column for column in df.columns
        if column != TARGET
        and column not in {item for columns in excluded.values() for item in columns}
        and df[column].nunique(dropna=True) <= 1
    ]

    if constant_columns:
        excluded["constant column"] = constant_columns

    return excluded


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    excluded = get_excluded_columns(df)
    excluded_columns = {column for columns in excluded.values() for column in columns}
    predictors = [column for column in df.columns if column not in excluded_columns]

    if TARGET in predictors:
        raise ValueError("Target column entered predictor list.")

    risky_words = ["injur", "kill", "fatal", "damage", "cost", "report", "key", "date"]
    risky_predictors = [
        column for column in predictors
        if any(word in column.lower() for word in risky_words)
    ]

    if risky_predictors:
        raise ValueError(f"Possible leakage predictors found: {risky_predictors}")

    return predictors


def get_feature_groups(
    df: pd.DataFrame,
    predictors: list[str],
) -> dict[str, list[str]]:
    groups = {"numeric": [], "categorical": [], "binary": []}

    for column in predictors:
        values = set(df[column].dropna().unique().tolist())

        if values and values.issubset({0, 1}):
            groups["binary"].append(column)
        elif is_numeric_dtype(df[column]):
            groups["numeric"].append(column)
        else:
            groups["categorical"].append(column)

    return groups


def make_splits(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    X_model, X_test, y_model, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    X_train, X_val, y_train, y_val = train_test_split(
        X_model,
        y_model,
        test_size=VALIDATION_SIZE,
        random_state=RANDOM_SEED,
        stratify=y_model,
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


def build_preprocessor(feature_groups: dict[str, list[str]]) -> ColumnTransformer:
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=True)),
    ])

    binary_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
    ])

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, feature_groups["numeric"]),
            ("categorical", categorical_pipeline, feature_groups["categorical"]),
            ("binary", binary_pipeline, feature_groups["binary"]),
        ],
        remainder="drop",
    )


def build_tree_model(feature_groups: dict[str, list[str]]) -> Pipeline:
    return Pipeline([
        ("preprocessor", build_preprocessor(feature_groups)),
        (
            "tree",
            DecisionTreeClassifier(
                max_depth=TREE_MAX_DEPTH,
                min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
                random_state=RANDOM_SEED,
            ),
        ),
    ])


def find_best_f1_threshold(
    y_true: pd.Series,
    probabilities: np.ndarray,
) -> pd.Series:
    thresholds = np.unique(np.concatenate(([0.0, 0.5, 1.0], probabilities)))
    rows = []

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        rows.append({
            "threshold": threshold,
            "f1": f1_score(y_true, predictions, zero_division=0),
            "precision": precision_score(y_true, predictions, zero_division=0),
            "recall": recall_score(y_true, predictions, zero_division=0),
            "predicted_positive_count": int(predictions.sum()),
        })

    results = pd.DataFrame(rows)
    results = results.sort_values(
        ["f1", "precision", "threshold"],
        ascending=[False, False, False],
    )

    return results.iloc[0]


def class_balance_row(split_name: str, y: pd.Series) -> dict[str, int | float | str]:
    injury_count = int((y == 1).sum())
    no_injury_count = int((y == 0).sum())

    return {
        "split": split_name,
        "rows": len(y),
        "no_injury_count": no_injury_count,
        "injury_count": injury_count,
        "injury_percent": round(injury_count / len(y) * 100, 2),
    }
