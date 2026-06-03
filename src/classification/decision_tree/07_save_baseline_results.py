import json
from pathlib import Path

import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
)

from common import (
    DATA_PATH,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    TREE_MAX_DEPTH,
    TREE_MIN_SAMPLES_LEAF,
    VALIDATION_SIZE,
    build_tree_model,
    find_best_f1_threshold,
    get_feature_groups,
    get_predictor_columns,
    load_data,
    make_splits,
    print_section,
)


OUTPUT_PATH = Path("data/decision_tree_baseline_results.json")


def make_metric_row(
    y_true: pd.Series,
    probabilities,
    threshold: float,
) -> dict[str, float | int]:
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


def main() -> None:
    df = load_data()

    print_section("DATASET LOADED")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    predictors = get_predictor_columns(df)
    feature_groups = get_feature_groups(df, predictors)
    X = df[predictors].copy()
    y = df[TARGET].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    model = build_tree_model(feature_groups)
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    best_threshold = find_best_f1_threshold(y_val, val_probabilities)
    threshold = float(best_threshold["threshold"])

    results = {
        "model": "DecisionTreeClassifier",
        "target": TARGET,
        "random_seed": RANDOM_SEED,
        "data_path": str(DATA_PATH),
        "split": {
            "test_size": TEST_SIZE,
            "validation_size_of_remaining_data": VALIDATION_SIZE,
            "train_rows": len(X_train),
            "validation_rows": len(X_val),
            "test_rows": len(X_test),
        },
        "features": {
            "raw_predictor_count": len(predictors),
            "numeric_count": len(feature_groups["numeric"]),
            "categorical_count": len(feature_groups["categorical"]),
            "binary_count": len(feature_groups["binary"]),
        },
        "tree_parameters": {
            "max_depth": TREE_MAX_DEPTH,
            "min_samples_leaf": TREE_MIN_SAMPLES_LEAF,
            "class_weight": None,
        },
        "threshold_selection": {
            "source": "validation",
            "threshold": round(threshold, 4),
            "validation_f1_at_threshold": round(float(best_threshold["f1"]), 4),
        },
        "metrics": {
            "validation": make_metric_row(y_val, val_probabilities, threshold),
            "test": make_metric_row(y_test, test_probabilities, threshold),
        },
    }

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")

    print_section("RESULTS SAVED")
    print(f"Output path: {OUTPUT_PATH}")
    print(f"Frozen threshold: {threshold:.4f}")
    print("\nValidation metrics:")
    print(pd.Series(results["metrics"]["validation"]).to_string())
    print("\nTest metrics:")
    print(pd.Series(results["metrics"]["test"]).to_string())


if __name__ == "__main__":
    main()
