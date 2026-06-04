"""
This is a shared helper function file that helps define common functions that will be used throughout the folder. It defines constants, random seeds, hyperparameters, etc. It also contains reusable functions that help load dataset, exclude columns, choosing the predictors, grouping features, splitting data, building the decision tree model, etc.  
"""

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
# tree hyperparameters
TREE_MAX_DEPTH = 8
TREE_MIN_SAMPLES_LEAF = 100


# columns that we will exclude
EXCLUDED_COLUMNS = [
    "injury_present",
    "Report Key",
    "Date",
    "year",
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
    "Vehicle Damage Cost",
]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, low_memory=False)


def get_excluded_columns(df: pd.DataFrame) -> list[str]:
    """
    Returns the excluded columns and the constant columns (columns with one value). 
    """

    excluded = []
    for column in EXCLUDED_COLUMNS:
        if column in df.columns and column not in excluded:
            excluded.append(column)

    excluded_set = set(excluded)

    constant_columns = [
        column for column in df.columns
        if column != TARGET
        and column not in excluded_set
        and df[column].nunique(dropna=True) <= 1
    ]

    return excluded + constant_columns


def get_predictor_columns(df: pd.DataFrame) -> list[str]:
    """
    Use the excluded columns to filter for columsn that will be used as features. Return the allowed feature columns. The goal it to make sure we dont have any leakage.
    """

    excluded_columns = set(get_excluded_columns(df))
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


def get_feature_groups(df: pd.DataFrame, predictors: list[str]) -> dict[str, list[str]]:
    """
    This separates the feature columns into groups and returns the groupings as a dictionary.
    """

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


def make_splits(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """
    We split the data into 3 sub categories. X_train, y_train for training. X_val, y_val is a set to help find the threshold that best works with the model. X_test, y_test is the testing set, untouched during the training.
    """
    
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
    """
    This builds a preprocessing step for each feature group. 
    """
    
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
