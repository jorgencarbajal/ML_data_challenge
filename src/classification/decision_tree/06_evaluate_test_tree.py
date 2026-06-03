import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from common import (
    DATA_PATH,
    TARGET,
    build_tree_model,
    class_balance_row,
    find_best_f1_threshold,
    get_feature_groups,
    get_predictor_columns,
    load_data,
    make_splits,
    print_section,
)


def metric_row(
    split_name: str,
    y_true: pd.Series,
    probabilities,
    threshold: float,
) -> dict[str, float | int | str]:
    predictions = (probabilities >= threshold).astype(int)

    return {
        "split": split_name,
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
    }


def print_split_balance(
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    print_section("SPLIT CLASS BALANCE")

    balance = pd.DataFrame([
        class_balance_row("train", y_train),
        class_balance_row("validation", y_val),
        class_balance_row("test", y_test),
    ])
    print(balance.to_string(index=False))


def print_final_results(
    y_val: pd.Series,
    y_test: pd.Series,
    val_probabilities,
    test_probabilities,
    threshold: float,
) -> None:
    print_section("FINAL METRICS")

    results = pd.DataFrame([
        metric_row("validation", y_val, val_probabilities, threshold),
        metric_row("test", y_test, test_probabilities, threshold),
    ])

    print(f"Frozen threshold selected on validation data: {threshold:.4f}")
    print("\nOfficial comparison metrics:")
    print(results.to_string(index=False))

    print("\nTest confusion matrix at frozen threshold:")
    print("Rows = actual class, columns = predicted class")
    print(confusion_matrix(y_test, test_probabilities >= threshold))


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

    print_split_balance(y_train, y_val, y_test)

    model = build_tree_model(feature_groups)
    model.fit(X_train, y_train)

    val_probabilities = model.predict_proba(X_val)[:, 1]
    test_probabilities = model.predict_proba(X_test)[:, 1]

    best_threshold = find_best_f1_threshold(y_val, val_probabilities)
    threshold = float(best_threshold["threshold"])

    print_section("THRESHOLD SELECTION")
    print("Threshold source: validation set only")
    print(f"Chosen threshold: {threshold:.4f}")
    print(f"Validation F1 at chosen threshold: {best_threshold['f1']:.4f}")
    print(f"Validation precision: {best_threshold['precision']:.4f}")
    print(f"Validation recall: {best_threshold['recall']:.4f}")

    print_final_results(
        y_val,
        y_test,
        val_probabilities,
        test_probabilities,
        threshold,
    )

    print_section("TEST SET RULE")
    print("The held-out test set was not used for fitting or threshold selection.")


if __name__ == "__main__":
    main()
