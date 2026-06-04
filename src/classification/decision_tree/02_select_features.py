import pandas as pd

from common import (
    DATA_PATH,
    TARGET,
    get_excluded_columns,
    get_feature_groups,
    get_predictor_columns,
    load_data,
    print_section,
)


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def print_target_summary(df: pd.DataFrame) -> None:
    print_section("TARGET SUMMARY")

    counts = df[TARGET].value_counts(dropna=False).sort_index()
    summary = counts.rename_axis("value").reset_index(name="count")
    summary["percent"] = (
        summary["count"] / len(df) * 100
    ).map(format_percent)

    print(summary.to_string(index=False))


def print_excluded_columns(
    df: pd.DataFrame,
    excluded_columns: list[str],
) -> None:
    print_section("EXCLUDED COLUMNS")

    print(f"Total excluded columns: {len(excluded_columns)}")

    for column in excluded_columns:
        missing = df[column].isna().sum()
        unique = df[column].nunique(dropna=True)
        print(
            f"  - {column} "
            f"(dtype={df[column].dtype}, missing={missing:,}, unique={unique:,})"
        )


def print_included_columns(
    df: pd.DataFrame,
    included_columns: list[str],
    feature_types: dict[str, list[str]],
) -> None:
    print_section("INCLUDED PREDICTOR COLUMNS")

    print(f"Total included predictors: {len(included_columns)}")

    for group_name in ["numeric", "categorical", "binary"]:
        columns = feature_types[group_name]
        print(f"\n{group_name}: {len(columns)} column(s)")

        for column in columns:
            missing = df[column].isna().sum()
            unique = df[column].nunique(dropna=True)
            print(
                f"  - {column} "
                f"(dtype={df[column].dtype}, missing={missing:,}, unique={unique:,})"
            )


def validate_feature_selection(
    df: pd.DataFrame,
    included_columns: list[str],
    excluded_columns: list[str],
) -> None:
    print_section("FEATURE SELECTION CHECKS")

    overlap = sorted(set(included_columns).intersection(excluded_columns))
    if overlap:
        raise ValueError(f"Columns are both included and excluded: {overlap}")

    accounted_columns = set(included_columns).union(excluded_columns)
    unaccounted_columns = sorted(set(df.columns).difference(accounted_columns))
    if unaccounted_columns:
        raise ValueError(f"Columns were not included or excluded: {unaccounted_columns}")

    print("Passed: every dataset column is either included or explicitly excluded.")
    print("Passed: no excluded columns entered the predictor list.")
    print(f"Rows available for modeling: {len(df):,}")
    print(f"Predictors available before preprocessing: {len(included_columns)}")


def main() -> None:
    df = load_data()

    print_section("DATASET LOADED")
    print(f"Path: {DATA_PATH}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    print_target_summary(df)

    included_columns = get_predictor_columns(df)
    excluded_columns = get_excluded_columns(df)
    feature_types = get_feature_groups(df, included_columns)

    print_excluded_columns(df, excluded_columns)
    print_included_columns(df, included_columns, feature_types)
    validate_feature_selection(df, included_columns, excluded_columns)


if __name__ == "__main__":
    main()
