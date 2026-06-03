from pathlib import Path
from sklearn.metrics import silhouette_score

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
import kmedoids


DATA_PATH = Path("data/v1.csv")   # Change this if your cleaned file has another name.

RANDOM_SEED = 42

# Start small while debugging the pipeline.
# We can increase this after everything works correctly.
PAM_SAMPLE_SIZE = 2_000 

# Candidate cluster counts for the first diagnostic run.
K_VALUES = range(2, 9)

# Numeric features should remain numeric for K-medoids.
NUMERIC_FEATURES = [
    "Train Speed",
    "Estimated Vehicle Speed",
    "Number Vehicle Occupants",
    "Number of Cars",
    "Temperature",
]

# Binary indicator features created during refinement.
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

# Categorical incident-characteristic features.
CATEGORICAL_FEATURES = [
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

# These are the only columns used to form clusters.
CLUSTER_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BINARY_FEATURES
)

# These columns are retained for auditing or post-cluster description only.
# They must not be included when fitting K-medoids.
PROFILE_ONLY_COLUMNS = [
    "Report Key",
    "year",
    "fatality_present",
    "injury_present",
]

KMEDOIDS_CATEGORICAL_FEATURES = [
    feature for feature in CATEGORICAL_FEATURES
    if feature != "time_of_day"
]



def load_and_validate_data(data_path: Path) -> pd.DataFrame:
    """
    Load the refined dataset and confirm that all columns required
    for the K-medoids baseline are available.
    """

    df = pd.read_csv(data_path, low_memory=False)

    required_columns = CLUSTER_FEATURES + PROFILE_ONLY_COLUMNS

    missing_columns = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"The following required columns are missing from the dataset: "
            f"{missing_columns}"
        )

    duplicate_reports = df["Report Key"].duplicated().sum()

    if duplicate_reports > 0:
        raise ValueError(
            f"Found {duplicate_reports} duplicate Report Key values. "
            f"Resolve duplicates before clustering."
        )

    selected_df = df[required_columns].copy()

    print("Dataset successfully loaded.")
    print(f"Rows available: {len(selected_df):,}")
    print(f"Clustering features: {len(CLUSTER_FEATURES)}")
    print(f"Profile-only columns: {PROFILE_ONLY_COLUMNS}")

    return selected_df


def prepare_kmedoids_data(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Prepare the clustering feature matrix for a Version 1 K-medoids run.

    Returns:
        model_df:
            Contains only the cleaned clustering features used to calculate distance.

        profile_df:
            Contains Report Key and outcome/context columns for post-cluster analysis only.

        audit_df:
            Summary of missing values and preparation decisions for each clustering feature.
    """

    model_df = df[CLUSTER_FEATURES].copy()
    profile_df = df[PROFILE_ONLY_COLUMNS].copy()

    audit_rows = []

    # ---------------------------------------------------------
    # Numeric features
    # ---------------------------------------------------------
    for feature in NUMERIC_FEATURES:
        # Converts malformed numeric values to missing values.
        model_df[feature] = pd.to_numeric(
            model_df[feature],
            errors="coerce"
        )

        missing_before = model_df[feature].isna().sum()

        # Median imputation is a simple Version 1 choice that is less
        # sensitive to extreme values than filling with the mean.
        median_value = model_df[feature].median()

        if pd.isna(median_value):
            raise ValueError(
                f"Numeric feature '{feature}' contains no usable numeric values."
            )

        model_df[feature] = model_df[feature].fillna(median_value)

        audit_rows.append({
            "Feature": feature,
            "Feature Type": "Numeric",
            "Missing Before Preparation": missing_before,
            "Missing Percent": missing_before / len(model_df) * 100,
            "Preparation Decision": f"Missing values filled with median: {median_value}"
        })

    # ---------------------------------------------------------
    # Categorical features
    # ---------------------------------------------------------
    for feature in CATEGORICAL_FEATURES:
        missing_before = model_df[feature].isna().sum()

        # Gower-style distance can compare category labels directly.
        # Missing is kept as an explicit category for this first diagnostic run.
        model_df[feature] = (
            model_df[feature]
            .fillna("Missing")
            .astype(str)
        )

        audit_rows.append({
            "Feature": feature,
            "Feature Type": "Categorical",
            "Missing Before Preparation": missing_before,
            "Missing Percent": missing_before / len(model_df) * 100,
            "Preparation Decision": "Missing values represented as category: Missing"
        })

    # ---------------------------------------------------------
    # Binary warning-device features
    # ---------------------------------------------------------
    for feature in BINARY_FEATURES:
        model_df[feature] = pd.to_numeric(
            model_df[feature],
            errors="coerce"
        )

        missing_before = model_df[feature].isna().sum()

        valid_values = set(model_df[feature].dropna().unique())
        invalid_values = valid_values - {0, 1}

        if invalid_values:
            raise ValueError(
                f"Binary feature '{feature}' contains invalid values: "
                f"{sorted(invalid_values)}"
            )

        if missing_before > 0:
            raise ValueError(
                f"Binary feature '{feature}' contains {missing_before} missing values. "
                f"Determine why the warning-device indicator is missing before clustering."
            )

        model_df[feature] = model_df[feature].astype(int)

        audit_rows.append({
            "Feature": feature,
            "Feature Type": "Binary",
            "Missing Before Preparation": missing_before,
            "Missing Percent": missing_before / len(model_df) * 100,
            "Preparation Decision": "Validated as binary indicator with values 0 or 1"
        })

    # ---------------------------------------------------------
    # Final validation
    # ---------------------------------------------------------
    if model_df.isna().any().any():
        remaining_missing = model_df.columns[model_df.isna().any()].tolist()

        raise ValueError(
            f"Missing values remain after preparation in: {remaining_missing}"
        )

    audit_df = pd.DataFrame(audit_rows)

    print("\nK-medoids preparation complete.")
    print(f"Modeling rows: {len(model_df):,}")
    print(f"Modeling features: {model_df.shape[1]}")
    print("\nFeatures with missing values before preparation:")
    print(
        audit_df[audit_df["Missing Before Preparation"] > 0][
            ["Feature", "Feature Type", "Missing Before Preparation", "Missing Percent"]
        ]
    )

    return model_df, profile_df, audit_df


def create_pam_sample(
    model_df: pd.DataFrame,
    profile_df: pd.DataFrame,
    sample_size: int = PAM_SAMPLE_SIZE,
    random_seed: int = RANDOM_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create a reproducible random sample for the Version 1 PAM experiment.

    model_sample:
        Features used to form clusters.

    profile_sample:
        Report Key and outcome/context columns kept only for later interpretation.
    """

    if len(model_df) != len(profile_df):
        raise ValueError(
            "model_df and profile_df do not contain the same number of rows."
        )

    if sample_size > len(model_df):
        print(
            f"Requested sample size of {sample_size:,} exceeds available rows. "
            f"Using all {len(model_df):,} rows instead."
        )
        sample_size = len(model_df)

    # Sample row indices once so the modeling and profile tables remain aligned.
    sampled_indices = model_df.sample(
        n=sample_size,
        random_state=random_seed
    ).index

    model_sample = model_df.loc[sampled_indices].copy()
    profile_sample = profile_df.loc[sampled_indices].copy()

    # Preserve the original dataset row number for traceability.
    profile_sample.insert(
        0,
        "Original Row Index",
        sampled_indices
    )

    # Reset row numbers so the distance matrix and cluster labels use 0...n-1.
    model_sample = model_sample.reset_index(drop=True)
    profile_sample = profile_sample.reset_index(drop=True)

    print("\nPAM pilot sample created.")
    print(f"Original prepared rows: {len(model_df):,}")
    print(f"Sampled rows: {len(model_sample):,}")
    print(f"Random seed: {random_seed}")

    print("\nSample outcome rates for later post-cluster interpretation only:")
    print(
        profile_sample[["fatality_present", "injury_present"]]
        .mean()
        .rename("Rate")
    )

    return model_sample, profile_sample


def compute_distance_matrix(
    model_sample: pd.DataFrame,
    numeric_reference_df: pd.DataFrame | None = None,
    numeric_features: list[str] = NUMERIC_FEATURES,
    categorical_features: list[str] = CATEGORICAL_FEATURES,
    binary_features: list[str] = BINARY_FEATURES,
) -> np.ndarray:
    """
    Compute a Gower-style dissimilarity matrix for mixed-type clustering data.

    Numeric features:
        Difference is divided by the feature range so each numeric feature
        contributes on a 0-to-1 scale.

    Categorical features:
        Same category contributes 0 distance.
        Different category contributes 1 distance.

    Binary warning-device features:
        Treated asymmetrically. Shared absence of a warning device does not
        make two incidents more similar. Matching presence does count.

    numeric_reference_df:
        Optional full prepared dataset used to calculate numeric ranges.
        This keeps numeric scaling consistent across different pilot samples.
    """

    if model_sample.empty:
        raise ValueError("Cannot compute distances from an empty sample.")

    active_features = numeric_features + categorical_features + binary_features

    missing_columns = [
        feature for feature in active_features
        if feature not in model_sample.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Distance calculation is missing clustering features: {missing_columns}"
        )

    if model_sample.isna().any().any():
        remaining_missing = model_sample.columns[
            model_sample.isna().any()
        ].tolist()

        raise ValueError(
            f"Missing values remain before distance calculation: {remaining_missing}"
        )

    if numeric_reference_df is None:
        numeric_reference_df = model_sample

    n_rows = len(model_sample)

    # Accumulates feature-by-feature distances.
    distance_sum = np.zeros((n_rows, n_rows), dtype=np.float32)

    # Counts how many features are relevant for each pair of records.
    # This changes pairwise for asymmetric binary features.
    weight_sum = np.zeros((n_rows, n_rows), dtype=np.float32)

    skipped_numeric_features = []

    # ---------------------------------------------------------
    # Numeric features
    # ---------------------------------------------------------
    for feature in numeric_features:
        values = model_sample[feature].to_numpy(dtype=np.float32)

        reference_min = numeric_reference_df[feature].min()
        reference_max = numeric_reference_df[feature].max()
        feature_range = reference_max - reference_min

        if feature_range == 0:
            skipped_numeric_features.append(feature)
            continue

        pairwise_difference = np.abs(
            values[:, None] - values[None, :]
        ) / feature_range

        distance_sum += pairwise_difference
        weight_sum += 1

    # ---------------------------------------------------------
    # Categorical features
    # ---------------------------------------------------------
    for feature in categorical_features:
        values = model_sample[feature].astype(str).to_numpy()

        pairwise_difference = (
            values[:, None] != values[None, :]
        ).astype(np.float32)

        distance_sum += pairwise_difference
        weight_sum += 1

    # ---------------------------------------------------------
    # Binary warning-device features
    # ---------------------------------------------------------
    for feature in binary_features:
        values = model_sample[feature].to_numpy(dtype=np.int8)

        # The feature matters for a pair only if at least one incident
        # actually has that warning device.
        pair_is_relevant = (
            (values[:, None] == 1) |
            (values[None, :] == 1)
        )

        pairwise_difference = (
            values[:, None] != values[None, :]
        )

        distance_sum += (
            pairwise_difference & pair_is_relevant
        ).astype(np.float32)

        weight_sum += pair_is_relevant.astype(np.float32)

    # ---------------------------------------------------------
    # Calculate final average dissimilarity
    # ---------------------------------------------------------
    distance_matrix = np.divide(
        distance_sum,
        weight_sum,
        out=np.zeros_like(distance_sum),
        where=weight_sum != 0
    )

    np.fill_diagonal(distance_matrix, 0.0)

    # ---------------------------------------------------------
    # Validation checks
    # ---------------------------------------------------------
    if not np.allclose(distance_matrix, distance_matrix.T):
        raise ValueError("Distance matrix is not symmetric.")

    if distance_matrix.min() < 0 or distance_matrix.max() > 1:
        raise ValueError(
            "Distance matrix contains values outside the expected 0-to-1 range."
        )

    upper_triangle = distance_matrix[
        np.triu_indices(n_rows, k=1)
    ]

    print("\nDistance matrix successfully created.")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Memory used: {distance_matrix.nbytes / (1024 ** 2):.2f} MB")
    print(f"Minimum pairwise distance: {upper_triangle.min():.4f}")
    print(f"Median pairwise distance: {np.median(upper_triangle):.4f}")
    print(f"Maximum pairwise distance: {upper_triangle.max():.4f}")

    if skipped_numeric_features:
        print(
            "Numeric features skipped because they had no variation: "
            f"{skipped_numeric_features}"
        )

    return distance_matrix


def assign_clusters(
    distance_matrix: np.ndarray,
    medoid_indices: list[int],
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Assign every record to its nearest medoid.

    Returns:
        cluster_labels:
            Cluster assignment for each sampled incident.

        closest_distances:
            Distance from each incident to its assigned medoid.

        total_cost:
            Sum of all distances to assigned medoids.
    """

    distances_to_medoids = distance_matrix[:, medoid_indices]

    cluster_labels = np.argmin(
        distances_to_medoids,
        axis=1
    )

    closest_distances = distances_to_medoids[
        np.arange(len(distance_matrix)),
        cluster_labels
    ]

    total_cost = float(closest_distances.sum())

    return cluster_labels, closest_distances, total_cost


# def run_pam(
#     distance_matrix: np.ndarray,
#     k: int,
#     max_iterations: int = 100,
# ) -> tuple[np.ndarray, list[int], float]:
#     """
#     Run Partitioning Around Medoids (PAM) on a precomputed distance matrix.

#     The algorithm has two major phases:

#     1. BUILD:
#        Select an initial set of representative incidents called medoids.

#     2. SWAP:
#        Attempt to replace current medoids with non-medoid incidents
#        whenever doing so reduces total clustering cost.

#     Returns:
#         cluster_labels:
#             Cluster assignment for each sampled incident.

#         medoid_indices:
#             Row positions of the representative incidents.

#         total_cost:
#             Sum of distances from all incidents to their assigned medoid.
#     """

#     if distance_matrix.ndim != 2:
#         raise ValueError("Distance matrix must be two-dimensional.")

#     n_rows, n_columns = distance_matrix.shape

#     if n_rows != n_columns:
#         raise ValueError("Distance matrix must be square.")

#     if k < 2:
#         raise ValueError("Use at least two clusters for this experiment.")

#     if k >= n_rows:
#         raise ValueError("Number of clusters must be smaller than the sample size.")

#     start_time = time.perf_counter()

#     # ---------------------------------------------------------
#     # BUILD phase
#     # ---------------------------------------------------------

#     # The first medoid is the incident with the smallest total
#     # dissimilarity to all other sampled incidents.
#     total_distance_per_record = distance_matrix.sum(axis=1)

#     first_medoid = int(np.argmin(total_distance_per_record))

#     medoid_indices = [first_medoid]

#     # Add remaining medoids one at a time.
#     # Each new medoid is the candidate that gives the lowest
#     # total assignment cost when added to the current medoids.
#     while len(medoid_indices) < k:
#         best_candidate = None
#         best_candidate_cost = np.inf

#         current_medoids = set(medoid_indices)

#         for candidate in range(n_rows):
#             if candidate in current_medoids:
#                 continue

#             trial_medoids = medoid_indices + [candidate]

#             _, _, trial_cost = assign_clusters(
#                 distance_matrix=distance_matrix,
#                 medoid_indices=trial_medoids
#             )

#             if trial_cost < best_candidate_cost:
#                 best_candidate = candidate
#                 best_candidate_cost = trial_cost

#         medoid_indices.append(best_candidate)

#     cluster_labels, closest_distances, total_cost = assign_clusters(
#         distance_matrix=distance_matrix,
#         medoid_indices=medoid_indices
#     )

#     print(f"\nPAM BUILD phase complete for k={k}.")
#     print(f"Initial medoid indices: {medoid_indices}")
#     print(f"Initial total cost: {total_cost:.4f}")

#     # ---------------------------------------------------------
#     # SWAP phase
#     # ---------------------------------------------------------

#     iterations_completed = 0

#     for iteration in range(1, max_iterations + 1):
#         best_swap_cost = total_cost
#         best_swap_medoids = None

#         current_medoids = set(medoid_indices)

#         for medoid_position in range(k):
#             for candidate in range(n_rows):
#                 if candidate in current_medoids:
#                     continue

#                 trial_medoids = medoid_indices.copy()
#                 trial_medoids[medoid_position] = candidate

#                 _, _, trial_cost = assign_clusters(
#                     distance_matrix=distance_matrix,
#                     medoid_indices=trial_medoids
#                 )

#                 if trial_cost < best_swap_cost:
#                     best_swap_cost = trial_cost
#                     best_swap_medoids = trial_medoids

#         # Stop when no swap improves the total cost.
#         if best_swap_medoids is None:
#             break

#         medoid_indices = best_swap_medoids
#         total_cost = best_swap_cost
#         iterations_completed = iteration

#         cluster_labels, closest_distances, total_cost = assign_clusters(
#             distance_matrix=distance_matrix,
#             medoid_indices=medoid_indices
#         )

#         print(
#             f"SWAP iteration {iteration}: "
#             f"total cost reduced to {total_cost:.4f}"
#         )

#     elapsed_seconds = time.perf_counter() - start_time

#     cluster_labels, closest_distances, total_cost = assign_clusters(
#         distance_matrix=distance_matrix,
#         medoid_indices=medoid_indices
#     )

#     cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

#     print(f"\nPAM complete for k={k}.")
#     print(f"Final medoid indices: {medoid_indices}")
#     print(f"Final total cost: {total_cost:.4f}")
#     print(f"Completed swap iterations: {iterations_completed}")
#     print(f"Runtime: {elapsed_seconds:.2f} seconds")
#     print("\nCluster sizes:")
#     print(cluster_sizes)

#     return cluster_labels, medoid_indices, total_cost


def run_pam(
    distance_matrix: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run the original PAM algorithm using the precomputed
    Gower-style distance matrix.
    """

    result = kmedoids.pam(
        diss=distance_matrix,
        medoids=k,
        init="build",
        random_state=RANDOM_SEED
    )

    cluster_labels = result.labels
    medoid_indices = result.medoids
    total_cost = float(result.loss)

    print(f"\nLibrary PAM complete for k={k}.")
    print(f"Medoid indices: {medoid_indices.tolist()}")
    print(f"Total cost: {total_cost:.4f}")
    print(f"Iterations: {result.n_iter}")
    print(f"Swaps performed: {result.n_swap}")

    print("\nCluster sizes:")
    print(pd.Series(cluster_labels).value_counts().sort_index())

    return cluster_labels, medoid_indices, total_cost


def evaluate_k_values(
    distance_matrix: np.ndarray,
    k_values=K_VALUES,
) -> pd.DataFrame:
    """
    Run PAM for several k values and record baseline clustering quality.

    Silhouette score:
        Higher is better. Measures how well-separated the clusters are.

    Total cost:
        Lower is better, but it will naturally decrease as k increases.
    """

    evaluation_rows = []

    for k in k_values:
        cluster_labels, medoid_indices, total_cost = run_pam(
            distance_matrix=distance_matrix,
            k=k
        )

        silhouette = silhouette_score(
            distance_matrix,
            cluster_labels,
            metric="precomputed"
        )

        cluster_sizes = pd.Series(cluster_labels).value_counts()

        evaluation_rows.append({
            "k": k,
            "Silhouette Score": silhouette,
            "Total Cost": total_cost,
            "Smallest Cluster Size": cluster_sizes.min(),
            "Largest Cluster Size": cluster_sizes.max(),
            "Medoid Indices": medoid_indices.tolist()
        })

    evaluation_df = pd.DataFrame(evaluation_rows)

    print("\nPAM evaluation summary:")
    print(
        evaluation_df[
            [
                "k",
                "Silhouette Score",
                "Total Cost",
                "Smallest Cluster Size",
                "Largest Cluster Size"
            ]
        ]
    )

    return evaluation_df


def plot_k_evaluation(evaluation_df: pd.DataFrame) -> None:
    """
    Plot silhouette score and total PAM cost across candidate k values.
    """

    plt.figure(figsize=(8, 5))
    plt.plot(
        evaluation_df["k"],
        evaluation_df["Silhouette Score"],
        marker="o"
    )
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Silhouette Score")
    plt.title("PAM Silhouette Score by Number of Clusters")
    plt.xticks(evaluation_df["k"])
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(8, 5))
    plt.plot(
        evaluation_df["k"],
        evaluation_df["Total Cost"],
        marker="o"
    )
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("Total Dissimilarity Cost")
    plt.title("PAM Total Cost by Number of Clusters")
    plt.xticks(evaluation_df["k"])
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def profile_clusters(
    model_sample: pd.DataFrame,
    profile_sample: pd.DataFrame,
    distance_matrix: np.ndarray,
    k: int,
    top_n: int = 8,
) -> dict[str, pd.DataFrame]:
    """
    Profile the clusters created by PAM.

    This function identifies:
        - Numeric differences between clusters
        - Categorical values overrepresented in each cluster
        - Warning-device indicators that distinguish each cluster
        - Outcome rates for post-cluster description only
    """

    cluster_labels, medoid_indices, total_cost = run_pam(
        distance_matrix=distance_matrix,
        k=k
    )

    clustered_model = model_sample.copy()
    clustered_model["cluster"] = cluster_labels

    clustered_profile = profile_sample.copy()
    clustered_profile["cluster"] = cluster_labels

    # ---------------------------------------------------------
    # Cluster sizes
    # ---------------------------------------------------------
    cluster_sizes = (
        clustered_model["cluster"]
        .value_counts()
        .sort_index()
        .rename("Cluster Size")
        .to_frame()
    )

    # ---------------------------------------------------------
    # Numeric feature medians
    # ---------------------------------------------------------
    numeric_profile = (
        clustered_model
        .groupby("cluster")[NUMERIC_FEATURES]
        .median()
        .round(2)
        .T
    )

    numeric_profile.insert(
        0,
        "Overall Median",
        model_sample[NUMERIC_FEATURES].median().round(2)
    )

    # ---------------------------------------------------------
    # Categorical feature differences
    # ---------------------------------------------------------
    categorical_rows = []

    for cluster in sorted(clustered_model["cluster"].unique()):
        cluster_df = clustered_model[clustered_model["cluster"] == cluster]

        for feature in CATEGORICAL_FEATURES:
            overall_rates = model_sample[feature].value_counts(normalize=True)
            cluster_rates = cluster_df[feature].value_counts(normalize=True)

            for value in cluster_rates.index:
                cluster_percent = cluster_rates[value] * 100
                overall_percent = overall_rates.get(value, 0) * 100

                categorical_rows.append({
                    "Cluster": cluster,
                    "Feature": feature,
                    "Value": value,
                    "Cluster Percent": cluster_percent,
                    "Overall Percent": overall_percent,
                    "Difference Points": cluster_percent - overall_percent
                })

    categorical_profile = pd.DataFrame(categorical_rows)

    # ---------------------------------------------------------
    # Binary warning-device differences
    # ---------------------------------------------------------
    binary_rows = []

    for cluster in sorted(clustered_model["cluster"].unique()):
        cluster_df = clustered_model[clustered_model["cluster"] == cluster]

        for feature in BINARY_FEATURES:
            cluster_percent = cluster_df[feature].mean() * 100
            overall_percent = model_sample[feature].mean() * 100

            binary_rows.append({
                "Cluster": cluster,
                "Feature": feature,
                "Cluster Percent": cluster_percent,
                "Overall Percent": overall_percent,
                "Difference Points": cluster_percent - overall_percent,
                "Absolute Difference": abs(cluster_percent - overall_percent)
            })

    binary_profile = pd.DataFrame(binary_rows)

    # ---------------------------------------------------------
    # Representative medoid incidents
    # ---------------------------------------------------------
    medoid_profile = profile_sample.iloc[medoid_indices].copy()
    medoid_profile.insert(0, "cluster", cluster_labels[medoid_indices])
    medoid_profile.insert(1, "Medoid Sample Index", medoid_indices)

    # ---------------------------------------------------------
    # Outcomes: description only, not cluster formation
    # ---------------------------------------------------------
    outcome_profile = (
        clustered_profile
        .groupby("cluster")[["fatality_present", "injury_present"]]
        .mean()
        .mul(100)
        .round(2)
    )

    # ---------------------------------------------------------
    # Print readable diagnostic summary
    # ---------------------------------------------------------
    print(f"\n================ k={k} CLUSTER PROFILE ================")

    print("\nCluster sizes:")
    print(cluster_sizes)

    print("\nNumeric feature medians:")
    print(numeric_profile)

    for cluster in sorted(clustered_model["cluster"].unique()):
        print(f"\nCluster {cluster}: most overrepresented categorical values")

        top_categories = (
            categorical_profile[categorical_profile["Cluster"] == cluster]
            .sort_values("Difference Points", ascending=False)
            .head(top_n)
        )

        print(
            top_categories[
                [
                    "Feature",
                    "Value",
                    "Cluster Percent",
                    "Overall Percent",
                    "Difference Points"
                ]
            ].round(2).to_string(index=False)
        )

        print(f"\nCluster {cluster}: strongest warning-device differences")

        top_binary = (
            binary_profile[binary_profile["Cluster"] == cluster]
            .sort_values("Absolute Difference", ascending=False)
            .head(top_n)
        )

        print(
            top_binary[
                [
                    "Feature",
                    "Cluster Percent",
                    "Overall Percent",
                    "Difference Points"
                ]
            ].round(2).to_string(index=False)
        )

    print("\nOutcome percentages by cluster, for interpretation only:")
    print(outcome_profile)

    return {
        "cluster_sizes": cluster_sizes,
        "numeric_profile": numeric_profile,
        "categorical_profile": categorical_profile,
        "binary_profile": binary_profile,
        "medoid_profile": medoid_profile,
        "outcome_profile": outcome_profile
    }


def run_sensitivity_checks(
    model_sample: pd.DataFrame,
    model_df: pd.DataFrame,
    k: int = 3,
) -> pd.DataFrame:
    """
    Test whether the k=3 result is overly dependent on:
        - warning-device indicators
        - time/light variables that may overlap
    """

    configurations = {
        "Baseline - all features": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "binary_features": BINARY_FEATURES,
        },

        "No warning-device indicators": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "binary_features": [],
        },

        "Remove time_of_day": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": [
                feature for feature in CATEGORICAL_FEATURES
                if feature != "time_of_day"
            ],
            "binary_features": BINARY_FEATURES,
        },

        "Remove Visibility": {
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": [
                feature for feature in CATEGORICAL_FEATURES
                if feature != "Visibility"
            ],
            "binary_features": BINARY_FEATURES,
        },
    }

    results = []

    for name, feature_groups in configurations.items():
        print(f"\n================ {name} ================")

        distance_matrix = compute_distance_matrix(
            model_sample=model_sample,
            numeric_reference_df=model_df,
            numeric_features=feature_groups["numeric_features"],
            categorical_features=feature_groups["categorical_features"],
            binary_features=feature_groups["binary_features"],
        )

        cluster_labels, medoid_indices, total_cost = run_pam(
            distance_matrix=distance_matrix,
            k=k
        )

        silhouette = silhouette_score(
            distance_matrix,
            cluster_labels,
            metric="precomputed"
        )

        cluster_sizes = (
            pd.Series(cluster_labels)
            .value_counts()
            .sort_index()
            .tolist()
        )

        results.append({
            "Configuration": name,
            "k": k,
            "Silhouette Score": silhouette,
            "Total Cost": total_cost,
            "Cluster Sizes": cluster_sizes,
            "Medoid Indices": medoid_indices.tolist()
        })

    sensitivity_df = pd.DataFrame(results)

    print("\nSensitivity check summary:")
    print(
        sensitivity_df[
            ["Configuration", "k", "Silhouette Score", "Total Cost", "Cluster Sizes"]
        ].to_string(index=False)
    )

    return sensitivity_df


def main() -> None:
    df = load_and_validate_data(DATA_PATH)

    model_df, profile_df, audit_df = prepare_kmedoids_data(df)

    model_sample, profile_sample = create_pam_sample(
        model_df=model_df,
        profile_df=profile_df
    )

    # Final revised feature space:
    # remove time_of_day from fitting, keep warning-device indicators.
    distance_matrix = compute_distance_matrix(
        model_sample=model_sample,
        numeric_reference_df=model_df,
        numeric_features=NUMERIC_FEATURES,
        categorical_features=KMEDOIDS_CATEGORICAL_FEATURES,
        binary_features=BINARY_FEATURES
    )

    # Final Version 1 PAM choice based on highest silhouette score.
    final_profile = profile_clusters(
        model_sample=model_sample,
        profile_sample=profile_sample,
        distance_matrix=distance_matrix,
        k=2
    )


if __name__ == "__main__":
    main()