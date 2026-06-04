import numpy as np
import pandas as pd
from sklearn.cluster import HDBSCAN

from common import (
    BINARY_FEATURES,
    DISTANCE_MATRIX_PATH,
    FITTED_CATEGORICAL_FEATURES,
    FITTED_FEATURES,
    K_MEDOIDS_DISTANCE_MATRIX_PATH,
    K_MEDOIDS_SAMPLED_MODEL_FEATURES_PATH,
    K_MEDOIDS_SAMPLED_PROFILE_COLUMNS_PATH,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    PROFILE_COLUMNS,
    SAMPLED_MODEL_FEATURES_PATH,
    SAMPLED_PROFILE_COLUMNS_PATH,
    format_percent,
    load_kmedoids_sampled_inputs,
    print_section,
    save_sampled_inputs,
    validate_sampled_inputs,
)


def print_source_paths() -> None:
    print_section("SOURCE ARTIFACTS")
    print("HDBSCAN will reuse the exact K-medoids sampled inputs.")
    print(f"K-medoids sampled fitted features: {K_MEDOIDS_SAMPLED_MODEL_FEATURES_PATH}")
    print(f"K-medoids sampled profile columns: {K_MEDOIDS_SAMPLED_PROFILE_COLUMNS_PATH}")
    print(f"K-medoids distance matrix: {K_MEDOIDS_DISTANCE_MATRIX_PATH}")


def print_method_setup() -> None:
    print_section("METHOD SETUP")
    print("Implementation: sklearn.cluster.HDBSCAN")
    print(f"HDBSCAN class: {HDBSCAN}")
    print("Distance input: precomputed Gower-style distance matrix")
    print("Comparison rule: same sampled incidents and same distance matrix as K-medoids")
    print("Noise label: -1")


def print_feature_setup() -> None:
    print_section("FEATURE SETUP")
    print(f"Numeric fitted features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical fitted features: {len(FITTED_CATEGORICAL_FEATURES)}")
    print(f"Binary warning-device fitted features: {len(BINARY_FEATURES)}")
    print(f"Total fitted features: {len(FITTED_FEATURES)}")
    print(f"Profile columns: {PROFILE_COLUMNS}")
    print("Excluded from fitting but kept for profiling: time_of_day")


def print_input_summary(
    model_sample: pd.DataFrame,
    profile_sample: pd.DataFrame,
    distance_matrix: np.ndarray,
) -> None:
    print_section("SAMPLED INPUT CHECK")
    print(f"Sampled fitted feature rows: {len(model_sample):,}")
    print(f"Sampled profile rows: {len(profile_sample):,}")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Distance matrix memory: {distance_matrix.nbytes / 1024**2:.2f} MB")

    upper_triangle = distance_matrix[
        np.triu_indices(len(distance_matrix), k=1)
    ]

    print(f"Minimum pairwise distance: {upper_triangle.min():.4f}")
    print(f"Median pairwise distance: {np.median(upper_triangle):.4f}")
    print(f"Maximum pairwise distance: {upper_triangle.max():.4f}")
    print(f"Symmetric: {np.allclose(distance_matrix, distance_matrix.T)}")
    print(
        "Within 0-to-1 range: "
        f"{distance_matrix.min() >= 0 and distance_matrix.max() <= 1}"
    )


def print_column_audit(model_sample: pd.DataFrame, profile_sample: pd.DataFrame) -> None:
    print_section("COLUMN AUDIT")

    fitted_audit = pd.DataFrame({
        "column": FITTED_FEATURES,
        "source": "fitted_feature",
        "dtype": [str(model_sample[column].dtype) for column in FITTED_FEATURES],
        "missing": [int(model_sample[column].isna().sum()) for column in FITTED_FEATURES],
        "missing_percent": [
            format_percent(model_sample[column].isna().mean() * 100)
            for column in FITTED_FEATURES
        ],
        "unique_nonmissing": [
            int(model_sample[column].nunique(dropna=True))
            for column in FITTED_FEATURES
        ],
    })

    profile_audit = pd.DataFrame({
        "column": PROFILE_COLUMNS,
        "source": "profile_only",
        "dtype": [str(profile_sample[column].dtype) for column in PROFILE_COLUMNS],
        "missing": [int(profile_sample[column].isna().sum()) for column in PROFILE_COLUMNS],
        "missing_percent": [
            format_percent(profile_sample[column].isna().mean() * 100)
            for column in PROFILE_COLUMNS
        ],
        "unique_nonmissing": [
            int(profile_sample[column].nunique(dropna=True))
            for column in PROFILE_COLUMNS
        ],
    })

    print(pd.concat([fitted_audit, profile_audit]).to_string(index=False))


def save_outputs(
    model_sample: pd.DataFrame,
    profile_sample: pd.DataFrame,
    distance_matrix: np.ndarray,
) -> None:
    save_sampled_inputs(
        model_sample=model_sample,
        profile_sample=profile_sample,
        distance_matrix=distance_matrix,
    )

    print_section("FILES WRITTEN")
    print(f"HDBSCAN output directory: {OUTPUT_DIR}")
    print(f"Sampled fitted features: {SAMPLED_MODEL_FEATURES_PATH}")
    print(f"Sampled profile columns: {SAMPLED_PROFILE_COLUMNS_PATH}")
    print(f"Distance matrix: {DISTANCE_MATRIX_PATH}")


def main() -> None:
    print_source_paths()
    print_method_setup()
    print_feature_setup()

    model_sample, profile_sample, distance_matrix = load_kmedoids_sampled_inputs()
    validate_sampled_inputs(
        model_sample=model_sample,
        profile_sample=profile_sample,
        distance_matrix=distance_matrix,
    )

    print_input_summary(
        model_sample=model_sample,
        profile_sample=profile_sample,
        distance_matrix=distance_matrix,
    )
    print_column_audit(model_sample, profile_sample)
    save_outputs(model_sample, profile_sample, distance_matrix)


if __name__ == "__main__":
    main()
