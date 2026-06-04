import inspect

import pandas as pd

from common import (
    DATA_PATH,
    OUTPUT_DIR,
    RANDOM_SEED,
    TARGET,
    TEST_SIZE,
    VALIDATION_SIZE,
    class_balance_row,
    format_percent,
    get_excluded_columns,
    get_feature_groups,
    get_predictor_columns,
    load_data,
    make_splits,
    print_section,
)


def print_dependency_check() -> None:
    print_section("DEPENDENCY CHECK")

    import torch
    import pytorch_tabnet
    from pytorch_tabnet.tab_model import TabNetClassifier

    print(f"torch version: {torch.__version__}")
    print(f"pytorch_tabnet version: {getattr(pytorch_tabnet, '__version__', 'not exposed')}")
    print("TabNetClassifier import: passed")
    print(f"TabNetClassifier signature: {inspect.signature(TabNetClassifier)}")


def print_dataset_summary(df: pd.DataFrame) -> None:
    print_section("DATASET LOADED")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def print_target_summary(df: pd.DataFrame) -> None:
    print_section("TARGET CHECK")

    if TARGET not in df.columns:
        raise ValueError(f"Missing target column: {TARGET}")

    target = df[TARGET]
    counts = target.value_counts(dropna=False).sort_index()
    target_summary = counts.rename_axis("value").reset_index(name="count")
    target_summary["percent"] = (
        target_summary["count"] / len(df) * 100
    ).map(format_percent)

    print(f"Target column: {TARGET}")
    print(f"Target dtype: {target.dtype}")
    print(f"Missing target values: {target.isna().sum():,}")
    print("\nTarget distribution:")
    print(target_summary.to_string(index=False))

    observed_values = set(target.dropna().unique().tolist())
    if observed_values != {0, 1}:
        raise ValueError(
            f"{TARGET} must be binary with values {{0, 1}}. "
            f"Observed values: {sorted(observed_values)}"
        )

    if target.isna().any():
        raise ValueError(f"{TARGET} contains missing values.")

    print("\nPassed: target exists, has no missing values, and is binary 0/1.")


def print_feature_selection(
    df: pd.DataFrame,
    predictors: list[str],
    excluded_columns: list[str],
    feature_groups: dict[str, list[str]],
) -> None:
    print_section("FEATURE SELECTION")

    print(f"Excluded columns: {len(excluded_columns)}")
    for column in excluded_columns:
        print(f"  - {column}")

    print(f"\nIncluded predictors: {len(predictors)}")
    print(f"Numeric predictors: {len(feature_groups['numeric'])}")
    print(f"Categorical predictors: {len(feature_groups['categorical'])}")
    print(f"Binary predictors: {len(feature_groups['binary'])}")

    accounted_columns = set(predictors).union(excluded_columns)
    unaccounted_columns = sorted(set(df.columns).difference(accounted_columns))
    if unaccounted_columns:
        raise ValueError(f"Columns not included or excluded: {unaccounted_columns}")

    print("\nPassed: every column is either included or explicitly excluded.")
    print("Passed: leakage scan did not find risky predictor names.")


def print_tabnet_input_plan(feature_groups: dict[str, list[str]]) -> None:
    print_section("TABNET INPUT PLAN")
    print("Numeric predictors: median imputation fitted on training data.")
    print("Binary predictors: most-frequent imputation fitted on training data.")
    print("Categorical predictors: string fill + integer encoding fitted on training data.")
    print("Unknown validation/test categories will use a reserved unknown category code.")
    print("TabNet will receive cat_idxs and cat_dims for categorical embeddings.")

    print("\nFirst categorical predictors:")
    for column in feature_groups["categorical"][:10]:
        print(f"  - {column}")

    if len(feature_groups["categorical"]) > 10:
        print(f"  ... {len(feature_groups['categorical']) - 10} more")


def print_split_summary(
    y: pd.Series,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
) -> None:
    print_section("SPLIT STRATEGY")

    train_fraction = (1 - TEST_SIZE) * (1 - VALIDATION_SIZE)
    validation_fraction = (1 - TEST_SIZE) * VALIDATION_SIZE

    print(f"Random seed: {RANDOM_SEED}")
    print(f"Effective train size: {train_fraction:.0%}")
    print(f"Effective validation size: {validation_fraction:.0%}")
    print(f"Effective test size: {TEST_SIZE:.0%}")

    balance = pd.DataFrame([
        class_balance_row("full", y),
        class_balance_row("train", y_train),
        class_balance_row("validation", y_val),
        class_balance_row("test", y_test),
    ])

    print("\nClass balance:")
    print(balance.to_string(index=False))


def main() -> None:
    print_dependency_check()

    df = load_data()
    print_dataset_summary(df)
    print_target_summary(df)

    predictors = get_predictor_columns(df)
    excluded_columns = get_excluded_columns(df)
    feature_groups = get_feature_groups(df, predictors)

    print_feature_selection(
        df=df,
        predictors=predictors,
        excluded_columns=excluded_columns,
        feature_groups=feature_groups,
    )
    print_tabnet_input_plan(feature_groups)

    X = df[predictors].copy()
    y = df[TARGET].copy()
    _, _, _, y_train, y_val, y_test = make_splits(X, y)
    print_split_summary(y, y_train, y_val, y_test)

    print_section("NEXT STEP")
    print(f"TabNet artifacts will be written under: {OUTPUT_DIR}")
    print("Next script will prepare numeric arrays and categorical embedding metadata.")


if __name__ == "__main__":
    main()
