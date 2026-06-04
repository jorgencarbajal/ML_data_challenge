from pathlib import Path
import json

import numpy as np
import pandas as pd
from pandas.api.types import is_numeric_dtype
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "v1.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "classification" / "tab_net"

FEATURE_METADATA_PATH = OUTPUT_DIR / "feature_metadata.json"
PREPARED_TRAIN_PATH = OUTPUT_DIR / "prepared_train.csv"
PREPARED_VALIDATION_PATH = OUTPUT_DIR / "prepared_validation.csv"
PREPARED_TEST_PATH = OUTPUT_DIR / "prepared_test.csv"
TRAIN_TARGET_PATH = OUTPUT_DIR / "train_target.csv"
VALIDATION_TARGET_PATH = OUTPUT_DIR / "validation_target.csv"
TEST_TARGET_PATH = OUTPUT_DIR / "test_target.csv"
TABNET_RESULTS_PATH = OUTPUT_DIR / "tabnet_results.json"
TABNET_VALIDATION_RESULTS_PATH = OUTPUT_DIR / "tabnet_validation_results.json"
TABNET_MODEL_PATH = OUTPUT_DIR / "tabnet_model"
TABNET_MODEL_ZIP_PATH = OUTPUT_DIR / "tabnet_model.zip"
TABNET_FEATURE_IMPORTANCE_PATH = OUTPUT_DIR / "tabnet_feature_importance.csv"

TARGET = "injury_present"
RANDOM_SEED = 13
TEST_SIZE = 0.20
VALIDATION_SIZE = 0.20

TABNET_PARAMS = {
    "n_d": 8,
    "n_a": 8,
    "n_steps": 3,
    "gamma": 1.3,
    "lambda_sparse": 0.001,
    "cat_emb_dim": 1,
    "seed": RANDOM_SEED,
    "verbose": 1,
    "device_name": "cpu",
}

TABNET_FIT_PARAMS = {
    "max_epochs": 30,
    "patience": 5,
    "batch_size": 8192,
    "virtual_batch_size": 512,
    "num_workers": 0,
    "drop_last": False,
    "weights": 0,
    "eval_metric": ["auc"],
    "pin_memory": False,
}

CATEGORICAL_MISSING_VALUE = "__MISSING__"
CATEGORICAL_UNKNOWN_VALUE = "__UNKNOWN__"

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

RISKY_PREDICTOR_WORDS = [
    "injur",
    "kill",
    "fatal",
    "damage",
    "cost",
    "report",
    "key",
    "date",
]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def load_data() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATA_PATH}")

    return pd.read_csv(DATA_PATH, low_memory=False)


def get_excluded_columns(df: pd.DataFrame) -> list[str]:
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
    excluded_columns = set(get_excluded_columns(df))
    predictors = [column for column in df.columns if column not in excluded_columns]

    if TARGET in predictors:
        raise ValueError("Target column entered predictor list.")

    risky_predictors = [
        column for column in predictors
        if any(word in column.lower() for word in RISKY_PREDICTOR_WORDS)
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


def save_json(data: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")

    return json.loads(path.read_text(encoding="utf-8"))


def load_feature_metadata() -> dict:
    if not FEATURE_METADATA_PATH.exists():
        raise FileNotFoundError(
            "Feature metadata not found. Run 02_prepare_features.py first."
        )

    return json.loads(FEATURE_METADATA_PATH.read_text(encoding="utf-8"))


def load_prepared_tabnet_data() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    dict,
]:
    required_paths = [
        PREPARED_TRAIN_PATH,
        PREPARED_VALIDATION_PATH,
        PREPARED_TEST_PATH,
        TRAIN_TARGET_PATH,
        VALIDATION_TARGET_PATH,
        TEST_TARGET_PATH,
        FEATURE_METADATA_PATH,
    ]

    missing_paths = [path for path in required_paths if not path.exists()]

    if missing_paths:
        raise FileNotFoundError(
            "Prepared TabNet artifacts are missing. Run 02_prepare_features.py first. "
            f"Missing: {missing_paths}"
        )

    X_train = pd.read_csv(PREPARED_TRAIN_PATH)
    X_val = pd.read_csv(PREPARED_VALIDATION_PATH)
    X_test = pd.read_csv(PREPARED_TEST_PATH)

    y_train = pd.read_csv(TRAIN_TARGET_PATH)[TARGET]
    y_val = pd.read_csv(VALIDATION_TARGET_PATH)[TARGET]
    y_test = pd.read_csv(TEST_TARGET_PATH)[TARGET]
    metadata = load_feature_metadata()

    return X_train, X_val, X_test, y_train, y_val, y_test, metadata


def make_metric_row(
    y_true: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> dict[str, float | int]:
    from sklearn.metrics import average_precision_score

    predictions = (probabilities >= threshold).astype(int)

    return {
        "pr_auc_average_precision": round(
            average_precision_score(y_true, probabilities),
            4,
        ),
        "f1": round(f1_score(y_true, predictions, zero_division=0), 4),
        "precision": round(
            precision_score(y_true, predictions, zero_division=0),
            4,
        ),
        "recall": round(recall_score(y_true, predictions, zero_division=0), 4),
        "predicted_injury_cases": int(predictions.sum()),
        "actual_injury_cases": int((y_true == 1).sum()),
        "rows": len(y_true),
    }
