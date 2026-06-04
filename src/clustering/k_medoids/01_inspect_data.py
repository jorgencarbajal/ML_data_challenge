import pandas as pd

from common import (
    ALL_CATEGORICAL_FEATURES,
    BINARY_FEATURES,
    DATA_PATH,
    FITTED_CATEGORICAL_FEATURES,
    FITTED_FEATURES,
    NUMERIC_FEATURES,
    PROFILE_COLUMNS,
    REQUIRED_COLUMNS,
    format_percent,
    get_missing_columns,
    load_data,
    print_section,
)


def print_dataset_shape(df: pd.DataFrame) -> None:
    print_section("DATASET LOADED")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")


def print_required_column_check(df: pd.DataFrame) -> None:
    print_section("REQUIRED COLUMN CHECK")

    missing_columns = get_missing_columns(df, REQUIRED_COLUMNS)

    print(f"Required columns: {len(REQUIRED_COLUMNS)}")
    print(f"Missing required columns: {len(missing_columns)}")

    if missing_columns:
        for column in missing_columns:
            print(f"  - {column}")
        raise ValueError("Missing required columns. Stop before clustering.")

    print("Passed: all required clustering/profile columns are present.")


def print_feature_setup() -> None:
    print_section("FEATURE SETUP")
    print(f"Numeric fitted features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical features listed: {len(ALL_CATEGORICAL_FEATURES)}")
    print(f"Categorical fitted features: {len(FITTED_CATEGORICAL_FEATURES)}")
    print("Excluded from fitting but kept for profiling: time_of_day")
    print(f"Binary warning-device fitted features: {len(BINARY_FEATURES)}")
    print(f"Total fitted features: {len(FITTED_FEATURES)}")
    print(f"Profile-only columns: {PROFILE_COLUMNS}")


def print_basic_column_audit(df: pd.DataFrame) -> None:
    print_section("REQUIRED COLUMN AUDIT")

    audit = pd.DataFrame({
        "column": REQUIRED_COLUMNS,
        "dtype": [str(df[column].dtype) for column in REQUIRED_COLUMNS],
        "missing": [int(df[column].isna().sum()) for column in REQUIRED_COLUMNS],
        "missing_percent": [
            format_percent(df[column].isna().mean() * 100)
            for column in REQUIRED_COLUMNS
        ],
        "unique_nonmissing": [
            int(df[column].nunique(dropna=True))
            for column in REQUIRED_COLUMNS
        ],
    })

    print(audit.to_string(index=False))


def main() -> None:
    df = load_data()
    print_dataset_shape(df)
    print_required_column_check(df)
    print_feature_setup()
    print_basic_column_audit(df)


if __name__ == "__main__":
    main()
