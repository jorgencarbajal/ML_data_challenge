import numpy as np

from common import (
    DISTANCE_MATRIX_PATH,
    PAM_SAMPLE_SIZE,
    RANDOM_SEED,
    SAMPLED_MODEL_FEATURES_PATH,
    SAMPLED_PROFILE_COLUMNS_PATH,
    compute_distance_matrix,
    create_pam_sample,
    load_prepared_features,
    print_section,
)


def print_sample_summary(full_rows: int, sample_rows: int) -> None:
    print_section("PAM PILOT SAMPLE")
    print(f"Prepared rows available: {full_rows:,}")
    print(f"Sample size: {sample_rows:,}")
    print(f"Random seed: {RANDOM_SEED}")


def print_distance_summary(distance_matrix, skipped_numeric_features) -> None:
    upper_triangle = distance_matrix[
        np.triu_indices(len(distance_matrix), k=1)
    ]

    print_section("DISTANCE MATRIX CHECKS")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Memory size: {distance_matrix.nbytes / 1024**2:.2f} MB")
    print(f"Minimum pairwise distance: {upper_triangle.min():.4f}")
    print(f"Median pairwise distance: {np.median(upper_triangle):.4f}")
    print(f"Maximum pairwise distance: {upper_triangle.max():.4f}")
    print(f"Symmetric: {np.allclose(distance_matrix, distance_matrix.T)}")
    print(
        "Within 0-to-1 range: "
        f"{distance_matrix.min() >= 0 and distance_matrix.max() <= 1}"
    )

    if skipped_numeric_features:
        print(f"Skipped numeric features with zero range: {skipped_numeric_features}")
    else:
        print("Skipped numeric features with zero range: none")


def save_outputs(model_sample, profile_sample, distance_matrix) -> None:
    model_sample.to_csv(SAMPLED_MODEL_FEATURES_PATH, index=False)
    profile_sample.to_csv(SAMPLED_PROFILE_COLUMNS_PATH, index=False)
    np.save(DISTANCE_MATRIX_PATH, distance_matrix)

    print_section("FILES WRITTEN")
    print(f"Sampled fitted features: {SAMPLED_MODEL_FEATURES_PATH}")
    print(f"Sampled profile columns: {SAMPLED_PROFILE_COLUMNS_PATH}")
    print(f"Distance matrix: {DISTANCE_MATRIX_PATH}")


def main() -> None:
    model_df, profile_df = load_prepared_features()
    model_sample, profile_sample = create_pam_sample(
        model_df=model_df,
        profile_df=profile_df,
    )

    print_sample_summary(len(model_df), len(model_sample))

    distance_matrix, skipped_numeric_features = compute_distance_matrix(
        model_sample=model_sample,
        numeric_reference_df=model_df,
    )

    print_distance_summary(distance_matrix, skipped_numeric_features)
    save_outputs(model_sample, profile_sample, distance_matrix)


if __name__ == "__main__":
    main()
