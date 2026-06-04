import json

import pandas as pd
from sklearn.metrics import silhouette_score

from common import (
    K_EVALUATION_PATH,
    K_VALUES,
    SELECTED_K_PATH,
    load_distance_matrix,
    print_section,
    run_pam,
)


def evaluate_k_values(distance_matrix) -> pd.DataFrame:
    rows = []

    for k in K_VALUES:
        print(f"Running PAM for k={k}...")

        pam_result = run_pam(distance_matrix=distance_matrix, k=k)
        cluster_labels = pam_result["labels"]
        cluster_sizes = pd.Series(cluster_labels).value_counts().sort_index()

        silhouette = silhouette_score(
            distance_matrix,
            cluster_labels,
            metric="precomputed",
        )

        rows.append({
            "k": k,
            "silhouette_score": float(silhouette),
            "total_cost": pam_result["total_cost"],
            "smallest_cluster_size": int(cluster_sizes.min()),
            "largest_cluster_size": int(cluster_sizes.max()),
            "cluster_sizes": cluster_sizes.to_dict(),
            "medoid_indices": pam_result["medoids"].tolist(),
            "iterations": pam_result["iterations"],
            "swaps": pam_result["swaps"],
        })

    return pd.DataFrame(rows)


def choose_selected_k(evaluation_df: pd.DataFrame) -> dict:
    sorted_results = evaluation_df.sort_values(
        ["silhouette_score", "k"],
        ascending=[False, True],
    )
    best_row = sorted_results.iloc[0]

    return {
        "selected_k": int(best_row["k"]),
        "selection_rule": "highest silhouette score; smaller k breaks ties",
        "silhouette_score": float(best_row["silhouette_score"]),
        "total_cost": float(best_row["total_cost"]),
        "smallest_cluster_size": int(best_row["smallest_cluster_size"]),
        "largest_cluster_size": int(best_row["largest_cluster_size"]),
        "medoid_indices": best_row["medoid_indices"],
    }


def save_outputs(evaluation_df: pd.DataFrame, selected_k_info: dict) -> None:
    evaluation_df.to_csv(K_EVALUATION_PATH, index=False)

    with SELECTED_K_PATH.open("w", encoding="utf-8") as file:
        json.dump(selected_k_info, file, indent=2)

    print_section("FILES WRITTEN")
    print(f"k evaluation results: {K_EVALUATION_PATH}")
    print(f"Selected k metadata: {SELECTED_K_PATH}")


def main() -> None:
    distance_matrix = load_distance_matrix()

    print_section("PAM K EVALUATION")
    print(f"Distance matrix shape: {distance_matrix.shape}")
    print(f"Candidate k values: {list(K_VALUES)}")

    evaluation_df = evaluate_k_values(distance_matrix)
    selected_k_info = choose_selected_k(evaluation_df)

    print_section("K RESULTS TABLE")
    print(
        evaluation_df[
            [
                "k",
                "silhouette_score",
                "total_cost",
                "smallest_cluster_size",
                "largest_cluster_size",
            ]
        ].round(4).to_string(index=False)
    )

    print_section("SELECTED BASELINE K")
    print(f"Selected k: {selected_k_info['selected_k']}")
    print(f"Selection rule: {selected_k_info['selection_rule']}")
    print(f"Silhouette score: {selected_k_info['silhouette_score']:.4f}")
    print(f"Total cost: {selected_k_info['total_cost']:.4f}")
    print(f"Smallest cluster size: {selected_k_info['smallest_cluster_size']}")
    print(f"Largest cluster size: {selected_k_info['largest_cluster_size']}")

    save_outputs(evaluation_df, selected_k_info)


if __name__ == "__main__":
    main()
