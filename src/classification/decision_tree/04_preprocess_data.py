import pandas as pd
from scipy import sparse

from common import (
    DATA_PATH,
    TARGET,
    build_preprocessor,
    get_feature_groups,
    get_predictor_columns,
    load_data,
    make_splits,
    print_section,
)


def print_split_rows(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    print_section("SPLIT ROWS")
    print(f"Train rows: {len(X_train):,}")
    print(f"Validation rows: {len(X_val):,}")
    print(f"Test rows: {len(X_test):,}")


def print_feature_groups(feature_groups: dict[str, list[str]]) -> None:
    print_section("RAW FEATURE GROUPS")
    print(f"Numeric predictors: {len(feature_groups['numeric'])}")
    print(f"Categorical predictors: {len(feature_groups['categorical'])}")
    print(f"Binary predictors: {len(feature_groups['binary'])}")
    print(f"Total predictors before preprocessing: {sum(len(v) for v in feature_groups.values())}")


def print_preprocessing_summary(
    preprocessor,
    X_train_processed,
    X_val_processed,
    X_test_processed,
    feature_groups: dict[str, list[str]],
) -> None:
    print_section("PREPROCESSING SUMMARY")

    feature_names = preprocessor.get_feature_names_out()
    categorical_encoder = (
        preprocessor
        .named_transformers_["categorical"]
        .named_steps["encoder"]
    )
    categorical_feature_names = categorical_encoder.get_feature_names_out(
        feature_groups["categorical"]
    )

    print("Passed: preprocessing was fitted on the training set only.")
    print(f"Output matrix type: {'sparse' if sparse.issparse(X_train_processed) else 'dense'}")
    print(f"Train shape after preprocessing: {X_train_processed.shape}")
    print(f"Validation shape after preprocessing: {X_val_processed.shape}")
    print(f"Test shape after preprocessing: {X_test_processed.shape}")

    print("\nEncoded feature counts:")
    print(f"Numeric output features: {len(feature_groups['numeric'])}")
    print(f"Categorical output features: {len(categorical_feature_names)}")
    print(f"Binary output features: {len(feature_groups['binary'])}")
    print(f"Total output features: {len(feature_names)}")

    print("\nFirst 20 output feature names:")
    for name in feature_names[:20]:
        print(f"  - {name}")

    print("\nLast 10 output feature names:")
    for name in feature_names[-10:]:
        print(f"  - {name}")


def validate_preprocessing_shapes(
    X_train,
    X_val,
    X_test,
    X_train_processed,
    X_val_processed,
    X_test_processed,
) -> None:
    print_section("PREPROCESSING CHECKS")

    if X_train_processed.shape[0] != len(X_train):
        raise ValueError("Train row count changed during preprocessing.")

    if X_val_processed.shape[0] != len(X_val):
        raise ValueError("Validation row count changed during preprocessing.")

    if X_test_processed.shape[0] != len(X_test):
        raise ValueError("Test row count changed during preprocessing.")

    output_feature_count = X_train_processed.shape[1]
    if X_val_processed.shape[1] != output_feature_count:
        raise ValueError("Validation output feature count does not match train.")

    if X_test_processed.shape[1] != output_feature_count:
        raise ValueError("Test output feature count does not match train.")

    print("Passed: preprocessing kept all split row counts unchanged.")
    print("Passed: train, validation, and test have the same output feature count.")


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

    X_train, X_val, X_test, _, _, _ = make_splits(X, y)

    print_split_rows(X_train, X_val, X_test)
    print_feature_groups(feature_groups)

    preprocessor = build_preprocessor(feature_groups)
    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print_preprocessing_summary(
        preprocessor,
        X_train_processed,
        X_val_processed,
        X_test_processed,
        feature_groups,
    )
    validate_preprocessing_shapes(
        X_train,
        X_val,
        X_test,
        X_train_processed,
        X_val_processed,
        X_test_processed,
    )


if __name__ == "__main__":
    main()
