import json

import numpy as np
import pandas as pd
from sklearn.metrics import silhouette_score

from common import (
    HDBSCAN_EVALUATION_PATH,
    HDBSCAN_GRID,
    MAX_NOISE_PERCENT_FOR_SELECTION,
    MIN_CLUSTER_SIZE_FOR_SELECTION,
    OUTPUT_DIR,
    SELECTED_MODEL_PATH,
    load_hdbscan_inputs,
    print_section,
    run_hdbscan,
)


def evaluate_labels(
    distance_matrix: np.ndarray,
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> dict:
    labels = np.asarray(labels)
    probabilities = np.asarray(probabilities)
    clustered_mask = labels != -1
    clustered_labels = labels[clustered_mask]
    cluster_sizes = pd.Series(clustered_labels).value_counts().sort_index()

    cluster_count = int(len(cluster_sizes))
    noise_count = int((labels == -1).sum())
    clustered_count = int(clustered_mask.sum())
    noise_percent = noise_count / len(labels) * 100
    clustered_percent = clustered_count / len(labels) * 100

    silhouette_clustered_only = np.nan

    if cluster_count >= 2 and clustered_count > cluster_count:
        clustered_distances = distance_matrix[np.ix_(clustered_mask, clustered_mask)]
        silhouette_clustered_only = silhouette_score(
            clustered_distances,
            clustered_labels,
            metric="precomputed",
        )

    if clustered_count > 0:
        clustered_probabilities = probabilities[clustered_mask]
        mean_probability = float(clustered_probabilities.mean())
        median_probability = float(np.median(clustered_probabilities))
        smallest_cluster_size = int(cluster_sizes.min())
        largest_cluster_size = int(cluster_sizes.max())
        largest_cluster_percent = largest_cluster_size / clustered_count * 100
    else:
        mean_probability = np.nan
        median_probability = np.nan
        smallest_cluster_size = 0
        largest_cluster_size = 0
        largest_cluster_percent = np.nan

    cluster_size_dict = {
        str(int(cluster)): int(size)
        for cluster, size in cluster_sizes.to_dict().items()
    }

    return {
        "cluster_count": cluster_count,
        "noise_count": noise_count,
        "noise_percent": noise_percent,
        "clustered_count": clustered_count,
        "clustered_percent": clustered_percent,
        "smallest_cluster_size": smallest_cluster_size,
        "largest_cluster_size": largest_cluster_size,
        "largest_cluster_percent_of_clustered": largest_cluster_percent,
        "silhouette_clustered_only": silhouette_clustered_only,
        "mean_cluster_probability": mean_probability,
        "median_cluster_probability": median_probability,
        "cluster_sizes": cluster_size_dict,
    }


def evaluate_grid(distance_matrix: np.ndarray) -> pd.DataFrame:
    rows = []

    for index, parameters in enumerate(HDBSCAN_GRID, start=1):
        print(
            f"Running HDBSCAN {index}/{len(HDBSCAN_GRID)}: "
            f"min_cluster_size={parameters['min_cluster_size']}, "
            f"min_samples={parameters['min_samples']}, "
            f"method={parameters['cluster_selection_method']}"
        )

        result = run_hdbscan(
            distance_matrix=distance_matrix,
            min_cluster_size=parameters["min_cluster_size"],
            min_samples=parameters["min_samples"],
            cluster_selection_method=parameters["cluster_selection_method"],
        )

        row = {
            **parameters,
            **evaluate_labels(
                distance_matrix=distance_matrix,
                labels=result["labels"],
                probabilities=result["probabilities"],
            ),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def choose_selected_model(evaluation_df: pd.DataFrame) -> dict:
    selection_df = evaluation_df.copy()
    selection_df["valid_for_selection"] = (
        (selection_df["cluster_count"] >= 2)
        & (selection_df["noise_percent"] <= MAX_NOISE_PERCENT_FOR_SELECTION)
        & (selection_df["smallest_cluster_size"] >= MIN_CLUSTER_SIZE_FOR_SELECTION)
        & selection_df["silhouette_clustered_only"].notna()
    )

    valid_df = selection_df[selection_df["valid_for_selection"]].copy()

    if valid_df.empty:
        ranked_df = selection_df[
            selection_df["silhouette_clustered_only"].notna()
        ].copy()
        selection_rule = (
            "fallback: no configuration met the noise and minimum-cluster-size "
            "filters, so the highest clustered-point silhouette was selected"
        )
    else:
        ranked_df = valid_df
        selection_rule = (
            "highest clustered-point silhouette among configurations with at "
            f"least 2 clusters, noise <= {MAX_NOISE_PERCENT_FOR_SELECTION:.0f}%, "
            f"and smallest cluster >= {MIN_CLUSTER_SIZE_FOR_SELECTION}"
        )

    if ranked_df.empty:
        ranked_df = selection_df.copy()
        selection_rule = (
            "fallback: all configurations were all-noise or single-cluster; "
            "selected the configuration with the lowest noise percent"
        )
        sort_columns = ["noise_percent", "min_cluster_size"]
        ascending = [True, True]
    else:
        sort_columns = [
            "silhouette_clustered_only",
            "noise_percent",
            "cluster_count",
            "min_cluster_size",
        ]
        ascending = [False, True, False, True]

    best_row = ranked_df.sort_values(sort_columns, ascending=ascending).iloc[0]

    selected = {
        "selection_rule": selection_rule,
        "min_cluster_size": int(best_row["min_cluster_size"]),
        "min_samples": (
            None
            if pd.isna(best_row["min_samples"])
            else int(best_row["min_samples"])
        ),
        "cluster_selection_method": str(best_row["cluster_selection_method"]),
        "cluster_count": int(best_row["cluster_count"]),
        "noise_count": int(best_row["noise_count"]),
        "noise_percent": float(best_row["noise_percent"]),
        "clustered_count": int(best_row["clustered_count"]),
        "clustered_percent": float(best_row["clustered_percent"]),
        "smallest_cluster_size": int(best_row["smallest_cluster_size"]),
        "largest_cluster_size": int(best_row["largest_cluster_size"]),
        "silhouette_clustered_only": (
            None
            if pd.isna(best_row["silhouette_clustered_only"])
            else float(best_row["silhouette_clustered_only"])
        ),
        "cluster_sizes": best_row["cluster_sizes"],
    }

    evaluation_df["valid_for_selection"] = selection_df["valid_for_selection"]

    return selected


def print_evaluation_summary(
    evaluation_df: pd.DataFrame,
    selected_model: dict,
) -> None:
    print_section("HDBSCAN GRID SUMMARY")

    summary_columns = [
        "min_cluster_size",
        "min_samples",
        "cluster_selection_method",
        "cluster_count",
        "noise_percent",
        "clustered_count",
        "smallest_cluster_size",
        "largest_cluster_size",
        "silhouette_clustered_only",
        "valid_for_selection",
    ]

    sorted_summary = evaluation_df.sort_values(
        ["valid_for_selection", "silhouette_clustered_only", "noise_percent"],
        ascending=[False, False, True],
    )

    print(sorted_summary[summary_columns].round(4).to_string(index=False))

    print_section("SELECTED HDBSCAN CONFIGURATION")
    print(f"Selection rule: {selected_model['selection_rule']}")
    print(f"min_cluster_size: {selected_model['min_cluster_size']}")
    print(f"min_samples: {selected_model['min_samples']}")
    print(f"cluster_selection_method: {selected_model['cluster_selection_method']}")
    print(f"Cluster count excluding noise: {selected_model['cluster_count']}")
    print(f"Noise percent: {selected_model['noise_percent']:.2f}%")
    print(f"Clustered rows: {selected_model['clustered_count']:,}")
    print(f"Smallest cluster size: {selected_model['smallest_cluster_size']}")
    print(f"Largest cluster size: {selected_model['largest_cluster_size']}")
    print(
        "Clustered-point silhouette: "
        f"{selected_model['silhouette_clustered_only']}"
    )
    print(f"Cluster sizes: {selected_model['cluster_sizes']}")


def save_outputs(evaluation_df: pd.DataFrame, selected_model: dict) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    evaluation_df.to_csv(HDBSCAN_EVALUATION_PATH, index=False)

    with SELECTED_MODEL_PATH.open("w", encoding="utf-8") as file:
        json.dump(selected_model, file, indent=2)

    print_section("FILES WRITTEN")
    print(f"HDBSCAN evaluation: {HDBSCAN_EVALUATION_PATH}")
    print(f"Selected HDBSCAN model: {SELECTED_MODEL_PATH}")


def main() -> None:
    _, _, distance_matrix = load_hdbscan_inputs()

    print_section("HDBSCAN GRID EVALUATION")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Grid configurations: {len(HDBSCAN_GRID)}")
    print("Metric: precomputed Gower-style distance")
    print("Primary quality measure: silhouette on non-noise clustered points")
    print("Additional measures: noise percent, cluster count, cluster sizes")

    evaluation_df = evaluate_grid(distance_matrix)
    selected_model = choose_selected_model(evaluation_df)

    print_evaluation_summary(
        evaluation_df=evaluation_df,
        selected_model=selected_model,
    )
    save_outputs(
        evaluation_df=evaluation_df,
        selected_model=selected_model,
    )


if __name__ == "__main__":
    main()
