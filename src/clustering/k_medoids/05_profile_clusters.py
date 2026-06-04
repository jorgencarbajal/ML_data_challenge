import json

import numpy as np
import pandas as pd

from common import (
    BINARY_FEATURES,
    CLUSTER_ASSIGNMENTS_PATH,
    CLUSTER_BINARY_PROFILE_PATH,
    CLUSTER_CATEGORICAL_PROFILE_PATH,
    CLUSTER_NUMERIC_PROFILE_PATH,
    CLUSTER_OUTCOME_PROFILE_PATH,
    CLUSTER_SUMMARY_PATH,
    FITTED_CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    SELECTED_K_PATH,
    load_distance_matrix,
    load_sampled_features,
    print_section,
    run_pam,
)


def load_selected_k() -> int:
    if not SELECTED_K_PATH.exists():
        raise FileNotFoundError("Selected k file not found. Run 04_evaluate_k.py first.")

    with SELECTED_K_PATH.open("r", encoding="utf-8") as file:
        selected_k_info = json.load(file)

    return int(selected_k_info["selected_k"])


def make_cluster_assignments(
    profile_sample: pd.DataFrame,
    distance_matrix: np.ndarray,
    cluster_labels,
    medoid_indices,
) -> pd.DataFrame:
    medoid_indices = np.asarray(medoid_indices)
    distances_to_medoids = distance_matrix[:, medoid_indices]
    nearest_medoid_position = np.argmin(distances_to_medoids, axis=1)

    assignments = profile_sample.copy()
    assignments["cluster"] = cluster_labels
    assignments["assigned_medoid_sample_index"] = medoid_indices[nearest_medoid_position]
    assignments["distance_to_medoid"] = distances_to_medoids[
        np.arange(len(distance_matrix)),
        nearest_medoid_position,
    ]
    assignments["is_medoid"] = np.isin(np.arange(len(distance_matrix)), medoid_indices)

    return assignments


def build_top_categorical_profile(
    clustered_model: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    rows = []

    for cluster in sorted(clustered_model["cluster"].unique()):
        cluster_df = clustered_model[clustered_model["cluster"] == cluster]

        for feature in FITTED_CATEGORICAL_FEATURES:
            overall_rates = clustered_model[feature].value_counts(normalize=True)
            cluster_rates = cluster_df[feature].value_counts(normalize=True)

            for value, cluster_rate in cluster_rates.items():
                overall_rate = overall_rates.get(value, 0.0)
                rows.append({
                    "cluster": int(cluster),
                    "feature": feature,
                    "value": value,
                    "cluster_percent": cluster_rate * 100,
                    "overall_percent": overall_rate * 100,
                    "difference_points": (cluster_rate - overall_rate) * 100,
                })

    profile_df = pd.DataFrame(rows)

    return (
        profile_df
        .sort_values(["cluster", "difference_points"], ascending=[True, False])
        .groupby("cluster")
        .head(top_n)
        .reset_index(drop=True)
    )


def build_time_of_day_profile(assignments: pd.DataFrame) -> pd.DataFrame:
    rows = []

    for cluster in sorted(assignments["cluster"].unique()):
        cluster_df = assignments[assignments["cluster"] == cluster]
        rates = cluster_df["time_of_day"].value_counts(normalize=True).sort_index()

        for value, rate in rates.items():
            rows.append({
                "cluster": int(cluster),
                "time_of_day": value,
                "cluster_percent": rate * 100,
            })

    return pd.DataFrame(rows)


def print_profile_summary(
    selected_k: int,
    pam_result: dict,
    cluster_sizes: pd.Series,
    numeric_profile: pd.DataFrame,
    binary_profile: pd.DataFrame,
    top_categorical_profile: pd.DataFrame,
    time_of_day_profile: pd.DataFrame,
    outcome_profile: pd.DataFrame,
) -> None:
    print_section("FINAL PAM CLUSTERING")
    print(f"Selected k: {selected_k}")
    print(f"Total cost: {pam_result['total_cost']:.4f}")
    print(f"Medoid sample indices: {pam_result['medoids'].tolist()}")

    print_section("CLUSTER SIZES")
    print(cluster_sizes.to_string())

    print_section("NUMERIC MEDIANS BY CLUSTER")
    print(numeric_profile.round(2).to_string())

    binary_spread = binary_profile.max(axis=0) - binary_profile.min(axis=0)
    top_binary_features = binary_spread.sort_values(ascending=False).head(8).index

    print_section("STRONGEST WARNING-DEVICE RATE DIFFERENCES")
    print(binary_profile[top_binary_features].round(2).to_string())

    print_section("TOP OVERREPRESENTED FITTED CATEGORICAL VALUES")
    print(
        top_categorical_profile[
            [
                "cluster",
                "feature",
                "value",
                "cluster_percent",
                "overall_percent",
                "difference_points",
            ]
        ].round(2).to_string(index=False)
    )

    print_section("POST-HOC TIME OF DAY PROFILE")
    print(time_of_day_profile.round(2).to_string(index=False))

    print_section("POST-HOC OUTCOME RATES BY CLUSTER")
    print("Description only: these outcome columns were not used to form clusters.")
    print(outcome_profile.round(2).to_string())


def save_outputs(
    assignments: pd.DataFrame,
    numeric_profile: pd.DataFrame,
    binary_profile: pd.DataFrame,
    top_categorical_profile: pd.DataFrame,
    outcome_profile: pd.DataFrame,
    summary: dict,
) -> None:
    assignments.to_csv(CLUSTER_ASSIGNMENTS_PATH, index=False)
    numeric_profile.to_csv(CLUSTER_NUMERIC_PROFILE_PATH)
    binary_profile.to_csv(CLUSTER_BINARY_PROFILE_PATH)
    top_categorical_profile.to_csv(CLUSTER_CATEGORICAL_PROFILE_PATH, index=False)
    outcome_profile.to_csv(CLUSTER_OUTCOME_PROFILE_PATH)

    with CLUSTER_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print_section("FILES WRITTEN")
    print(f"Cluster assignments: {CLUSTER_ASSIGNMENTS_PATH}")
    print(f"Numeric profile: {CLUSTER_NUMERIC_PROFILE_PATH}")
    print(f"Binary warning-device profile: {CLUSTER_BINARY_PROFILE_PATH}")
    print(f"Top categorical profile: {CLUSTER_CATEGORICAL_PROFILE_PATH}")
    print(f"Post-hoc outcome rates: {CLUSTER_OUTCOME_PROFILE_PATH}")
    print(f"Cluster summary: {CLUSTER_SUMMARY_PATH}")


def main() -> None:
    selected_k = load_selected_k()
    distance_matrix = load_distance_matrix()
    model_sample, profile_sample = load_sampled_features()

    pam_result = run_pam(distance_matrix=distance_matrix, k=selected_k)
    cluster_labels = pam_result["labels"]

    clustered_model = model_sample.copy()
    clustered_model["cluster"] = cluster_labels

    assignments = make_cluster_assignments(
        profile_sample=profile_sample,
        distance_matrix=distance_matrix,
        cluster_labels=cluster_labels,
        medoid_indices=pam_result["medoids"],
    )

    cluster_sizes = (
        pd.Series(cluster_labels)
        .value_counts()
        .sort_index()
        .rename("cluster_size")
    )
    numeric_profile = clustered_model.groupby("cluster")[NUMERIC_FEATURES].median()
    binary_profile = clustered_model.groupby("cluster")[BINARY_FEATURES].mean().mul(100)
    top_categorical_profile = build_top_categorical_profile(clustered_model)
    time_of_day_profile = build_time_of_day_profile(assignments)
    outcome_profile = (
        assignments
        .groupby("cluster")[["fatality_present", "injury_present"]]
        .mean()
        .mul(100)
    )

    summary = {
        "selected_k": selected_k,
        "total_cost": float(pam_result["total_cost"]),
        "medoid_sample_indices": [int(value) for value in pam_result["medoids"]],
        "cluster_sizes": {
            str(cluster): int(size)
            for cluster, size in cluster_sizes.to_dict().items()
        },
        "outcome_note": (
            "fatality_present and injury_present are post-hoc descriptions only; "
            "they were not used to form clusters."
        ),
    }

    print_profile_summary(
        selected_k=selected_k,
        pam_result=pam_result,
        cluster_sizes=cluster_sizes,
        numeric_profile=numeric_profile,
        binary_profile=binary_profile,
        top_categorical_profile=top_categorical_profile,
        time_of_day_profile=time_of_day_profile,
        outcome_profile=outcome_profile,
    )

    save_outputs(
        assignments=assignments,
        numeric_profile=numeric_profile,
        binary_profile=binary_profile,
        top_categorical_profile=top_categorical_profile,
        outcome_profile=outcome_profile,
        summary=summary,
    )


if __name__ == "__main__":
    main()
