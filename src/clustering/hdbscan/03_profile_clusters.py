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
    load_hdbscan_inputs,
    load_selected_model,
    print_section,
    run_hdbscan,
)


def label_display_name(label: int) -> str:
    if label == -1:
        return "noise"

    return f"cluster_{label}"


def make_cluster_assignments(
    profile_sample: pd.DataFrame,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    assignments = profile_sample.copy()
    assignments["cluster"] = labels
    assignments["cluster_label"] = [
        label_display_name(int(label))
        for label in labels
    ]
    assignments["is_noise"] = assignments["cluster"] == -1
    assignments["cluster_probability"] = probabilities

    return assignments


def build_top_categorical_profile(
    clustered_model: pd.DataFrame,
    top_n: int = 8,
) -> pd.DataFrame:
    rows = []

    for label in sorted(clustered_model["cluster"].unique()):
        group_df = clustered_model[clustered_model["cluster"] == label]

        for feature in FITTED_CATEGORICAL_FEATURES:
            overall_rates = clustered_model[feature].value_counts(normalize=True)
            group_rates = group_df[feature].value_counts(normalize=True)

            for value, group_rate in group_rates.items():
                overall_rate = overall_rates.get(value, 0.0)
                rows.append({
                    "cluster": int(label),
                    "cluster_label": label_display_name(int(label)),
                    "feature": feature,
                    "value": value,
                    "cluster_percent": group_rate * 100,
                    "overall_percent": overall_rate * 100,
                    "difference_points": (group_rate - overall_rate) * 100,
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

    for label in sorted(assignments["cluster"].unique()):
        group_df = assignments[assignments["cluster"] == label]
        rates = group_df["time_of_day"].value_counts(normalize=True).sort_index()

        for value, rate in rates.items():
            rows.append({
                "cluster": int(label),
                "cluster_label": label_display_name(int(label)),
                "time_of_day": value,
                "cluster_percent": rate * 100,
            })

    return pd.DataFrame(rows)


def print_profile_summary(
    selected_model: dict,
    cluster_sizes: pd.Series,
    numeric_profile: pd.DataFrame,
    binary_profile: pd.DataFrame,
    top_categorical_profile: pd.DataFrame,
    time_of_day_profile: pd.DataFrame,
    outcome_profile: pd.DataFrame,
    probability_profile: pd.DataFrame,
) -> None:
    print_section("FINAL HDBSCAN CLUSTERING")
    print("Implementation: sklearn.cluster.HDBSCAN")
    print("Metric: precomputed Gower-style distance")
    print(f"min_cluster_size: {selected_model['min_cluster_size']}")
    print(f"min_samples: {selected_model['min_samples']}")
    print(f"cluster_selection_method: {selected_model['cluster_selection_method']}")
    print(f"Cluster count excluding noise: {selected_model['cluster_count']}")
    print(f"Noise percent: {selected_model['noise_percent']:.2f}%")
    print(
        "Clustered-point silhouette: "
        f"{selected_model['silhouette_clustered_only']}"
    )

    print_section("CLUSTER AND NOISE SIZES")
    print(cluster_sizes.to_string())

    print_section("CLUSTER PROBABILITY SUMMARY")
    print(probability_profile.round(4).to_string())

    print_section("NUMERIC MEDIANS BY GROUP")
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
                "cluster_label",
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

    print_section("POST-HOC OUTCOME RATES BY GROUP")
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
    model_sample, profile_sample, distance_matrix = load_hdbscan_inputs()
    selected_model = load_selected_model()

    result = run_hdbscan(
        distance_matrix=distance_matrix,
        min_cluster_size=selected_model["min_cluster_size"],
        min_samples=selected_model["min_samples"],
        cluster_selection_method=selected_model["cluster_selection_method"],
    )

    labels = result["labels"]
    probabilities = result["probabilities"]

    clustered_model = model_sample.copy()
    clustered_model["cluster"] = labels
    clustered_model["cluster_label"] = [
        label_display_name(int(label))
        for label in labels
    ]

    assignments = make_cluster_assignments(
        profile_sample=profile_sample,
        labels=labels,
        probabilities=probabilities,
    )

    cluster_sizes = (
        assignments["cluster_label"]
        .value_counts()
        .rename("group_size")
    )

    numeric_profile = clustered_model.groupby("cluster_label")[NUMERIC_FEATURES].median()
    binary_profile = (
        clustered_model
        .groupby("cluster_label")[BINARY_FEATURES]
        .mean()
        .mul(100)
    )
    top_categorical_profile = build_top_categorical_profile(clustered_model)
    time_of_day_profile = build_time_of_day_profile(assignments)
    outcome_profile = (
        assignments
        .groupby("cluster_label")[["fatality_present", "injury_present"]]
        .mean()
        .mul(100)
    )
    probability_profile = (
        assignments
        .groupby("cluster_label")["cluster_probability"]
        .agg(["count", "mean", "median", "min", "max"])
    )

    summary = {
        "method": "HDBSCAN",
        "implementation": "sklearn.cluster.HDBSCAN",
        "metric": "precomputed Gower-style distance",
        "selected_parameters": {
            "min_cluster_size": selected_model["min_cluster_size"],
            "min_samples": selected_model["min_samples"],
            "cluster_selection_method": selected_model["cluster_selection_method"],
        },
        "cluster_count_excluding_noise": selected_model["cluster_count"],
        "noise_count": selected_model["noise_count"],
        "noise_percent": selected_model["noise_percent"],
        "clustered_count": selected_model["clustered_count"],
        "clustered_percent": selected_model["clustered_percent"],
        "silhouette_clustered_only": selected_model["silhouette_clustered_only"],
        "group_sizes": {
            str(label): int(size)
            for label, size in cluster_sizes.to_dict().items()
        },
        "outcome_note": (
            "fatality_present and injury_present are post-hoc descriptions only; "
            "they were not used to form clusters."
        ),
    }

    print_profile_summary(
        selected_model=selected_model,
        cluster_sizes=cluster_sizes,
        numeric_profile=numeric_profile,
        binary_profile=binary_profile,
        top_categorical_profile=top_categorical_profile,
        time_of_day_profile=time_of_day_profile,
        outcome_profile=outcome_profile,
        probability_profile=probability_profile,
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
