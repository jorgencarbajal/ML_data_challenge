import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from common import (
    DATA_PATH,
    RANDOM_SEED,
    TARGET,
    TREE_MAX_DEPTH,
    TREE_MIN_SAMPLES_LEAF,
    build_tree_model,
    find_best_f1_threshold,
    get_feature_groups,
    get_predictor_columns,
    load_data,
    make_splits,
    print_section,
)


def print_model_setup() -> None:
    print_section("MODEL SETUP")
    print("Model: sklearn DecisionTreeClassifier")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"max_depth: {TREE_MAX_DEPTH}")
    print(f"min_samples_leaf: {TREE_MIN_SAMPLES_LEAF}")
    print("class_weight: None")


def print_validation_results(
    y_val: pd.Series,
    val_probabilities: np.ndarray,
    best_threshold: pd.Series,
) -> None:
    print_section("VALIDATION RESULTS")

    default_predictions = (val_probabilities >= 0.50).astype(int)
    tuned_predictions = (
        val_probabilities >= best_threshold["threshold"]
    ).astype(int)

    print(f"Validation PR-AUC / Average Precision: {average_precision_score(y_val, val_probabilities):.4f}")

    print("\nDefault threshold results:")
    print("Threshold: 0.5000")
    print(f"F1: {f1_score(y_val, default_predictions, zero_division=0):.4f}")
    print(f"Precision: {precision_score(y_val, default_predictions, zero_division=0):.4f}")
    print(f"Recall: {recall_score(y_val, default_predictions, zero_division=0):.4f}")
    print(f"Predicted injury cases: {int(default_predictions.sum()):,}")

    print("\nBest validation F1 threshold:")
    print(f"Threshold: {best_threshold['threshold']:.4f}")
    print(f"F1: {best_threshold['f1']:.4f}")
    print(f"Precision: {best_threshold['precision']:.4f}")
    print(f"Recall: {best_threshold['recall']:.4f}")
    print(f"Predicted injury cases: {int(best_threshold['predicted_positive_count']):,}")

    print("\nValidation confusion matrix at selected threshold:")
    print("Rows = actual class, columns = predicted class")
    print(confusion_matrix(y_val, tuned_predictions))


def print_tree_summary(model: Pipeline) -> None:
    tree = model.named_steps["tree"]

    print_section("TREE SUMMARY")
    print(f"Actual tree depth: {tree.get_depth()}")
    print(f"Number of leaves: {tree.get_n_leaves()}")


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

    X_train, X_val, _, y_train, y_val, _ = make_splits(X, y)

    print_section("TRAINING DATA")
    print(f"Train rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Predictors before preprocessing: {len(predictors)}")

    print_model_setup()

    model = build_tree_model(feature_groups)
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    best_threshold = find_best_f1_threshold(y_val, val_probabilities)

    print_validation_results(y_val, val_probabilities, best_threshold)
    print_tree_summary(model)

    print_section("NEXT STEP")
    print(
        "The selected threshold came from validation data only. "
        "Do not change it when evaluating the held-out test set."
    )


if __name__ == "__main__":
    main()
