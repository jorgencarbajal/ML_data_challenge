import pandas as pd

from common import (
    DATA_PATH,
    RANDOM_SEED,
    TEST_SIZE,
    TARGET,
    VALIDATION_SIZE,
    class_balance_row,
    get_feature_groups,
    get_predictor_columns,
    load_data,
    make_splits,
    print_section,
)


def print_feature_summary(feature_types: dict[str, list[str]]) -> None:
    print_section("PREDICTOR SUMMARY")

    total_predictors = sum(len(columns) for columns in feature_types.values())
    print(f"Predictors before preprocessing: {total_predictors}")
    print(f"Numeric predictors: {len(feature_types['numeric'])}")
    print(f"Categorical predictors: {len(feature_types['categorical'])}")
    print(f"Binary predictors: {len(feature_types['binary'])}")


def print_split_strategy() -> None:
    print_section("SPLIT STRATEGY")

    train_fraction = (1 - TEST_SIZE) * (1 - VALIDATION_SIZE)
    validation_fraction = (1 - TEST_SIZE) * VALIDATION_SIZE

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Test size: {TEST_SIZE:.0%} of full dataset")
    print(
        "Validation size: "
        f"{VALIDATION_SIZE:.0%} of the post-test modeling set"
    )
    print(f"Effective train size: {train_fraction:.0%} of full dataset")
    print(f"Effective validation size: {validation_fraction:.0%} of full dataset")
    print(f"Effective test size: {TEST_SIZE:.0%} of full dataset")


def print_split_balance(
    y: pd.Series,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> None:
    print_section("SPLIT CLASS BALANCE")

    balance_rows = [
        class_balance_row("full", y),
        class_balance_row("train", y_train),
        class_balance_row("validation", y_validation),
        class_balance_row("test", y_test),
    ]

    balance = pd.DataFrame(balance_rows)
    print(balance.to_string(index=False))


def validate_split_sizes(
    X: pd.DataFrame,
    y: pd.Series,
    X_train: pd.DataFrame,
    X_validation: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_validation: pd.Series,
    y_test: pd.Series,
) -> None:
    print_section("SPLIT CHECKS")

    total_split_rows = len(X_train) + len(X_validation) + len(X_test)
    total_split_targets = len(y_train) + len(y_validation) + len(y_test)

    if total_split_rows != len(X):
        raise ValueError("Split feature row counts do not sum to the full dataset.")

    if total_split_targets != len(y):
        raise ValueError("Split target row counts do not sum to the full dataset.")

    if len(X_train) != len(y_train):
        raise ValueError("Train X/y row counts do not match.")

    if len(X_validation) != len(y_validation):
        raise ValueError("Validation X/y row counts do not match.")

    if len(X_test) != len(y_test):
        raise ValueError("Test X/y row counts do not match.")

    train_index = set(X_train.index)
    validation_index = set(X_validation.index)
    test_index = set(X_test.index)

    if train_index.intersection(validation_index):
        raise ValueError("Train and validation indices overlap.")
    if train_index.intersection(test_index):
        raise ValueError("Train and test indices overlap.")
    if validation_index.intersection(test_index):
        raise ValueError("Validation and test indices overlap.")

    print("Passed: train, validation, and test row counts are consistent.")
    print("Passed: train, validation, and test indices do not overlap.")
    print("Passed: test data is held out before creating train/validation splits.")


def main() -> None:
    df = load_data()

    print_section("DATASET LOADED")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    included_columns = get_predictor_columns(df)
    feature_types = get_feature_groups(df, included_columns)
    print_feature_summary(feature_types)

    X = df[included_columns].copy()
    y = df[TARGET].copy()
    print_split_strategy()

    X_train, X_validation, X_test, y_train, y_validation, y_test = (
        make_splits(X, y)
    )

    print_split_balance(y, y_train, y_validation, y_test)
    validate_split_sizes(
        X,
        y,
        X_train,
        X_validation,
        X_test,
        y_train,
        y_validation,
        y_test,
    )


if __name__ == "__main__":
    main()
