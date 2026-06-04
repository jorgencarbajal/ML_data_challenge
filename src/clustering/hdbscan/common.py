from pathlib import Path
import json

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "v1.csv"

K_MEDOIDS_OUTPUT_DIR = REPO_ROOT / "data" / "clustering" / "k_medoids"
K_MEDOIDS_SAMPLED_MODEL_FEATURES_PATH = (
    K_MEDOIDS_OUTPUT_DIR / "sampled_model_features.csv"
)
K_MEDOIDS_SAMPLED_PROFILE_COLUMNS_PATH = (
    K_MEDOIDS_OUTPUT_DIR / "sampled_profile_columns.csv"
)
K_MEDOIDS_DISTANCE_MATRIX_PATH = K_MEDOIDS_OUTPUT_DIR / "distance_matrix.npy"

OUTPUT_DIR = REPO_ROOT / "data" / "clustering" / "hdbscan"
SAMPLED_MODEL_FEATURES_PATH = OUTPUT_DIR / "sampled_model_features.csv"
SAMPLED_PROFILE_COLUMNS_PATH = OUTPUT_DIR / "sampled_profile_columns.csv"
DISTANCE_MATRIX_PATH = OUTPUT_DIR / "distance_matrix.npy"
HDBSCAN_EVALUATION_PATH = OUTPUT_DIR / "hdbscan_evaluation.csv"
SELECTED_MODEL_PATH = OUTPUT_DIR / "selected_hdbscan_model.json"
CLUSTER_ASSIGNMENTS_PATH = OUTPUT_DIR / "cluster_assignments.csv"
CLUSTER_NUMERIC_PROFILE_PATH = OUTPUT_DIR / "cluster_numeric_medians.csv"
CLUSTER_BINARY_PROFILE_PATH = OUTPUT_DIR / "cluster_binary_rates.csv"
CLUSTER_CATEGORICAL_PROFILE_PATH = OUTPUT_DIR / "cluster_top_categorical_values.csv"
CLUSTER_OUTCOME_PROFILE_PATH = OUTPUT_DIR / "cluster_posthoc_outcome_rates.csv"
CLUSTER_SUMMARY_PATH = OUTPUT_DIR / "cluster_profile_summary.json"

RANDOM_SEED = 42

HDBSCAN_GRID = [
    {
        "min_cluster_size": min_cluster_size,
        "min_samples": min_samples,
        "cluster_selection_method": cluster_selection_method,
    }
    for cluster_selection_method in ["eom", "leaf"]
    for min_cluster_size in [5, 10, 15, 20, 25, 30, 50]
    for min_samples in [None, 1, 3, 5, 10, 20]
]

MIN_CLUSTER_SIZE_FOR_SELECTION = 20
MAX_NOISE_PERCENT_FOR_SELECTION = 90.0

NUMERIC_FEATURES = [
    "Train Speed",
    "Estimated Vehicle Speed",
    "Number Vehicle Occupants",
    "Number of Cars",
    "Temperature",
]

ALL_CATEGORICAL_FEATURES = [
    "season",
    "time_of_day",
    "Highway User",
    "Highway User Position",
    "Equipment Involved",
    "Equipment Struck",
    "Equipment Type",
    "Track Type",
    "Warning Connected To Signal",
    "Crossing Illuminated",
    "Visibility",
    "Weather Condition",
    "View Obstruction",
    "Highway User Action",
    "Driver Passed Vehicle",
    "Driver In Vehicle",
]

FITTED_CATEGORICAL_FEATURES = [
    feature for feature in ALL_CATEGORICAL_FEATURES
    if feature != "time_of_day"
]

BINARY_FEATURES = [
    "has_gate",
    "has_cantilever_fls",
    "has_standard_fls",
    "has_wig_wags",
    "has_highway_traffic_signals",
    "has_audible",
    "has_crossbucks",
    "has_stop_signs",
    "has_watchman",
    "has_flagged_by_crew",
    "has_other_warning",
    "has_no_warning_device",
]

FITTED_FEATURES = (
    NUMERIC_FEATURES
    + FITTED_CATEGORICAL_FEATURES
    + BINARY_FEATURES
)

PROFILE_COLUMNS = [
    "original_row_index",
    "Report Key",
    "year",
    "fatality_present",
    "injury_present",
    "time_of_day",
]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def get_missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, low_memory=False)


def load_kmedoids_sampled_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if not K_MEDOIDS_SAMPLED_MODEL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "K-medoids sampled model features not found. "
            "Run src/clustering/k_medoids/03_build_distance_matrix.py first."
        )

    if not K_MEDOIDS_SAMPLED_PROFILE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "K-medoids sampled profile columns not found. "
            "Run src/clustering/k_medoids/03_build_distance_matrix.py first."
        )

    if not K_MEDOIDS_DISTANCE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            "K-medoids distance matrix not found. "
            "Run src/clustering/k_medoids/03_build_distance_matrix.py first."
        )

    model_sample = pd.read_csv(
        K_MEDOIDS_SAMPLED_MODEL_FEATURES_PATH,
        low_memory=False,
    )
    profile_sample = pd.read_csv(
        K_MEDOIDS_SAMPLED_PROFILE_COLUMNS_PATH,
        low_memory=False,
    )
    distance_matrix = np.load(K_MEDOIDS_DISTANCE_MATRIX_PATH)

    return model_sample, profile_sample, distance_matrix


def validate_sampled_inputs(
    model_sample: pd.DataFrame,
    profile_sample: pd.DataFrame,
    distance_matrix: np.ndarray,
) -> None:
    missing_model_columns = get_missing_columns(model_sample, FITTED_FEATURES)
    missing_profile_columns = get_missing_columns(profile_sample, PROFILE_COLUMNS)

    if missing_model_columns:
        raise ValueError(f"Sampled model features missing: {missing_model_columns}")

    if missing_profile_columns:
        raise ValueError(f"Sampled profile columns missing: {missing_profile_columns}")

    if len(model_sample) != len(profile_sample):
        raise ValueError("Sampled model/profile files have different row counts.")

    if distance_matrix.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")

    if distance_matrix.shape[0] != len(model_sample):
        raise ValueError("Distance matrix rows do not match sampled feature rows.")

    if model_sample[FITTED_FEATURES].isna().any().any():
        columns_with_missing = (
            model_sample[FITTED_FEATURES]
            .columns[model_sample[FITTED_FEATURES].isna().any()]
            .tolist()
        )
        raise ValueError(f"Sampled fitted features still contain missing values: {columns_with_missing}")

    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix is not symmetric.")

    if distance_matrix.min() < 0 or distance_matrix.max() > 1:
        raise ValueError("Distance matrix has values outside the 0-to-1 range.")


def save_sampled_inputs(
    model_sample: pd.DataFrame,
    profile_sample: pd.DataFrame,
    distance_matrix: np.ndarray,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model_sample.to_csv(SAMPLED_MODEL_FEATURES_PATH, index=False)
    profile_sample.to_csv(SAMPLED_PROFILE_COLUMNS_PATH, index=False)
    np.save(DISTANCE_MATRIX_PATH, distance_matrix)


def load_hdbscan_inputs() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if not SAMPLED_MODEL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "HDBSCAN sampled model features not found. Run 01_inspect_data.py first."
        )

    if not SAMPLED_PROFILE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "HDBSCAN sampled profile columns not found. Run 01_inspect_data.py first."
        )

    if not DISTANCE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            "HDBSCAN distance matrix not found. Run 01_inspect_data.py first."
        )

    model_sample = pd.read_csv(SAMPLED_MODEL_FEATURES_PATH, low_memory=False)
    profile_sample = pd.read_csv(SAMPLED_PROFILE_COLUMNS_PATH, low_memory=False)
    distance_matrix = np.load(DISTANCE_MATRIX_PATH)

    validate_sampled_inputs(
        model_sample=model_sample,
        profile_sample=profile_sample,
        distance_matrix=distance_matrix,
    )

    return model_sample, profile_sample, distance_matrix


def load_selected_model() -> dict:
    if not SELECTED_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Selected HDBSCAN model not found. Run 02_evaluate_hdbscan.py first."
        )

    with SELECTED_MODEL_PATH.open("r", encoding="utf-8") as file:
        return json.load(file)


def run_hdbscan(
    distance_matrix: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    cluster_selection_method: str,
) -> dict:
    from sklearn.cluster import HDBSCAN

    model = HDBSCAN(
        metric="precomputed",
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=cluster_selection_method,
        n_jobs=-1,
        copy=True,
    )

    labels = model.fit_predict(distance_matrix)

    return {
        "labels": labels,
        "probabilities": model.probabilities_,
        "parameters": {
            "min_cluster_size": min_cluster_size,
            "min_samples": min_samples,
            "cluster_selection_method": cluster_selection_method,
        },
    }
