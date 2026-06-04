from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[3]
DATA_PATH = REPO_ROOT / "data" / "v1.csv"
OUTPUT_DIR = REPO_ROOT / "data" / "clustering" / "k_medoids"
PREPARED_MODEL_FEATURES_PATH = OUTPUT_DIR / "prepared_model_features.csv"
PREPARED_PROFILE_COLUMNS_PATH = OUTPUT_DIR / "prepared_profile_columns.csv"
SAMPLED_MODEL_FEATURES_PATH = OUTPUT_DIR / "sampled_model_features.csv"
SAMPLED_PROFILE_COLUMNS_PATH = OUTPUT_DIR / "sampled_profile_columns.csv"
DISTANCE_MATRIX_PATH = OUTPUT_DIR / "distance_matrix.npy"
K_EVALUATION_PATH = OUTPUT_DIR / "k_evaluation.csv"
SELECTED_K_PATH = OUTPUT_DIR / "selected_k.json"
CLUSTER_ASSIGNMENTS_PATH = OUTPUT_DIR / "cluster_assignments.csv"
CLUSTER_NUMERIC_PROFILE_PATH = OUTPUT_DIR / "cluster_numeric_medians.csv"
CLUSTER_BINARY_PROFILE_PATH = OUTPUT_DIR / "cluster_binary_rates.csv"
CLUSTER_CATEGORICAL_PROFILE_PATH = OUTPUT_DIR / "cluster_top_categorical_values.csv"
CLUSTER_OUTCOME_PROFILE_PATH = OUTPUT_DIR / "cluster_posthoc_outcome_rates.csv"
CLUSTER_SUMMARY_PATH = OUTPUT_DIR / "cluster_profile_summary.json"

RANDOM_SEED = 42
PAM_SAMPLE_SIZE = 2_000
K_VALUES = range(2, 9)

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
    "Report Key",
    "year",
    "fatality_present",
    "injury_present",
    "time_of_day",
]

REQUIRED_COLUMNS = FITTED_FEATURES + PROFILE_COLUMNS


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    return pd.read_csv(path, low_memory=False)


def get_missing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    return [column for column in columns if column not in df.columns]


def validate_required_columns(df: pd.DataFrame) -> None:
    missing_columns = get_missing_columns(df, REQUIRED_COLUMNS)

    if missing_columns:
        raise ValueError(f"Missing required columns: {missing_columns}")


def prepare_features(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    validate_required_columns(df)

    model_df = df[FITTED_FEATURES].copy()
    profile_df = df[PROFILE_COLUMNS].copy()
    audit_rows = []

    for feature in NUMERIC_FEATURES:
        model_df[feature] = pd.to_numeric(model_df[feature], errors="coerce")
        missing_before = int(model_df[feature].isna().sum())
        median_value = model_df[feature].median()

        if pd.isna(median_value):
            raise ValueError(f"Numeric feature has no usable values: {feature}")

        model_df[feature] = model_df[feature].fillna(median_value)

        audit_rows.append({
            "column": feature,
            "feature_group": "numeric",
            "missing_before": missing_before,
            "missing_percent": missing_before / len(model_df) * 100,
            "preparation": f"converted to numeric; missing filled with median {median_value}",
        })

    for feature in FITTED_CATEGORICAL_FEATURES:
        missing_before = int(model_df[feature].isna().sum())
        model_df[feature] = model_df[feature].fillna("Missing").astype(str)

        audit_rows.append({
            "column": feature,
            "feature_group": "categorical",
            "missing_before": missing_before,
            "missing_percent": missing_before / len(model_df) * 100,
            "preparation": 'missing filled with category "Missing"',
        })

    for feature in BINARY_FEATURES:
        model_df[feature] = pd.to_numeric(model_df[feature], errors="coerce")
        missing_before = int(model_df[feature].isna().sum())
        observed_values = set(model_df[feature].dropna().unique().tolist())
        invalid_values = observed_values - {0, 1}

        if invalid_values:
            raise ValueError(
                f"Binary feature {feature} has values outside 0/1: "
                f"{sorted(invalid_values)}"
            )

        if missing_before > 0:
            raise ValueError(
                f"Binary feature {feature} has {missing_before:,} missing values."
            )

        model_df[feature] = model_df[feature].astype(int)

        audit_rows.append({
            "column": feature,
            "feature_group": "binary",
            "missing_before": missing_before,
            "missing_percent": 0.0,
            "preparation": "validated as 0/1 with no missing values",
        })

    if model_df.isna().any().any():
        columns_with_missing = model_df.columns[model_df.isna().any()].tolist()
        raise ValueError(
            f"Missing values remain in fitted features: {columns_with_missing}"
        )

    profile_df["time_of_day"] = profile_df["time_of_day"].fillna("Missing").astype(str)

    audit_df = pd.DataFrame(audit_rows)

    return model_df, profile_df, audit_df


def load_prepared_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not PREPARED_MODEL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Prepared fitted features not found. Run 02_prepare_features.py first."
        )

    if not PREPARED_PROFILE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Prepared profile columns not found. Run 02_prepare_features.py first."
        )

    model_df = pd.read_csv(PREPARED_MODEL_FEATURES_PATH, low_memory=False)
    profile_df = pd.read_csv(PREPARED_PROFILE_COLUMNS_PATH, low_memory=False)

    return model_df, profile_df


def create_pam_sample(
    model_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    sample_size: int = PAM_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if len(model_df) != len(profile_df):
        raise ValueError("Prepared model/profile files have different row counts.")

    if sample_size > len(model_df):
        sample_size = len(model_df)

    sampled_indices = model_df.sample(
        n=sample_size,
        random_state=random_seed,
    ).index

    model_sample = model_df.loc[sampled_indices].copy()
    profile_sample = profile_df.loc[sampled_indices].copy()

    profile_sample.insert(0, "original_row_index", sampled_indices)

    model_sample = model_sample.reset_index(drop=True)
    profile_sample = profile_sample.reset_index(drop=True)

    return model_sample, profile_sample


def compute_distance_matrix(
    model_sample: pd.DataFrame,
    numeric_reference_df: pd.DataFrame,
) -> tuple[np.ndarray, list[str]]:
    if model_sample.empty:
        raise ValueError("Cannot build a distance matrix from an empty sample.")

    missing_columns = get_missing_columns(model_sample, FITTED_FEATURES)

    if missing_columns:
        raise ValueError(f"Sample is missing fitted features: {missing_columns}")

    if model_sample[FITTED_FEATURES].isna().any().any():
        columns_with_missing = (
            model_sample[FITTED_FEATURES]
            .columns[model_sample[FITTED_FEATURES].isna().any()]
            .tolist()
        )
        raise ValueError(f"Sample still has missing fitted values: {columns_with_missing}")

    n_rows = len(model_sample)
    distance_sum = np.zeros((n_rows, n_rows), dtype=np.float32)
    weight_sum = np.zeros((n_rows, n_rows), dtype=np.float32)
    skipped_numeric_features = []

    for feature in NUMERIC_FEATURES:
        values = model_sample[feature].to_numpy(dtype=np.float32)
        reference_min = numeric_reference_df[feature].min()
        reference_max = numeric_reference_df[feature].max()
        feature_range = reference_max - reference_min

        if feature_range == 0:
            skipped_numeric_features.append(feature)
            continue

        pairwise_distance = np.abs(values[:, None] - values[None, :]) / feature_range

        distance_sum += pairwise_distance
        weight_sum += 1

    for feature in FITTED_CATEGORICAL_FEATURES:
        values = model_sample[feature].astype(str).to_numpy()
        pairwise_distance = (values[:, None] != values[None, :]).astype(np.float32)

        distance_sum += pairwise_distance
        weight_sum += 1

    for feature in BINARY_FEATURES:
        values = model_sample[feature].to_numpy(dtype=np.int8)
        pair_is_relevant = (values[:, None] == 1) | (values[None, :] == 1)
        pairwise_difference = values[:, None] != values[None, :]

        distance_sum += (pairwise_difference & pair_is_relevant).astype(np.float32)
        weight_sum += pair_is_relevant.astype(np.float32)

    distance_matrix = np.divide(
        distance_sum,
        weight_sum,
        out=np.zeros_like(distance_sum),
        where=weight_sum != 0,
    )

    np.fill_diagonal(distance_matrix, 0.0)

    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix is not symmetric.")

    if distance_matrix.min() < 0 or distance_matrix.max() > 1:
        raise ValueError("Distance matrix has values outside the 0-to-1 range.")

    return distance_matrix, skipped_numeric_features


def load_distance_matrix() -> np.ndarray:
    if not DISTANCE_MATRIX_PATH.exists():
        raise FileNotFoundError(
            "Distance matrix not found. Run 03_build_distance_matrix.py first."
        )

    distance_matrix = np.load(DISTANCE_MATRIX_PATH)

    if distance_matrix.ndim != 2:
        raise ValueError("Distance matrix must be two-dimensional.")

    if distance_matrix.shape[0] != distance_matrix.shape[1]:
        raise ValueError("Distance matrix must be square.")

    return distance_matrix


def load_sampled_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not SAMPLED_MODEL_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "Sampled fitted features not found. Run 03_build_distance_matrix.py first."
        )

    if not SAMPLED_PROFILE_COLUMNS_PATH.exists():
        raise FileNotFoundError(
            "Sampled profile columns not found. Run 03_build_distance_matrix.py first."
        )

    model_sample = pd.read_csv(SAMPLED_MODEL_FEATURES_PATH, low_memory=False)
    profile_sample = pd.read_csv(SAMPLED_PROFILE_COLUMNS_PATH, low_memory=False)

    return model_sample, profile_sample


def run_pam(distance_matrix: np.ndarray, k: int) -> dict:
    import kmedoids

    result = kmedoids.pam(
        diss=distance_matrix,
        medoids=k,
        init="build",
        random_state=RANDOM_SEED,
    )

    return {
        "labels": result.labels,
        "medoids": result.medoids,
        "total_cost": float(result.loss),
        "iterations": int(getattr(result, "n_iter", -1)),
        "swaps": int(getattr(result, "n_swap", -1)),
    }
