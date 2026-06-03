from pathlib import Path

import pandas as pd


DATA_PATH = Path("data/v1.csv")
TARGET = "injury_present"
ID_COLUMNS = ["Report Key"]


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def format_percent(value: float) -> str:
    return f"{value:.2f}%"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    df = pd.read_csv(path, low_memory=False)
    print_section("DATASET LOADED")
    print(f"Path: {path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    return df


def validate_target(df: pd.DataFrame) -> None:
    print_section("TARGET CHECK")

    if TARGET not in df.columns:
        raise ValueError(f"Required target column is missing: {TARGET}")

    target = df[TARGET]
    print(f"Target column: {TARGET}")
    print(f"Target dtype: {target.dtype}")
    print(f"Missing target values: {target.isna().sum():,}")

    counts = target.value_counts(dropna=False).sort_index()
    target_summary = (
        counts.rename_axis("value")
        .reset_index(name="count")
    )
    target_summary["percent"] = (
        target_summary["count"] / len(df) * 100
    ).map(format_percent)
    print("\nTarget distribution:")
    print(target_summary.to_string(index=False))

    observed_values = set(target.dropna().unique().tolist())
    if observed_values != {0, 1}:
        raise ValueError(
            f"{TARGET} must be binary with values {{0, 1}}. "
            f"Observed non-missing values: {sorted(observed_values)}"
        )

    if target.isna().any():
        raise ValueError(f"{TARGET} contains missing values.")

    print("\nPassed: target exists, has no missing values, and is binary 0/1.")


def inspect_columns(df: pd.DataFrame) -> None:
    print_section("COLUMN AUDIT")

    audit = pd.DataFrame({
        "column": df.columns,
        "dtype": [str(df[column].dtype) for column in df.columns],
        "missing": [df[column].isna().sum() for column in df.columns],
        "missing_percent": [
            format_percent(df[column].isna().mean() * 100)
            for column in df.columns
        ],
        "unique_nonmissing": [
            df[column].nunique(dropna=True)
            for column in df.columns
        ],
    })

    print(audit.to_string(index=False))


def inspect_identifiers(df: pd.DataFrame) -> None:
    print_section("IDENTIFIER CHECK")

    for column in ID_COLUMNS:
        if column not in df.columns:
            print(f"{column}: not present")
            continue

        duplicate_count = int(df[column].duplicated().sum())
        unique_count = int(df[column].nunique(dropna=True))
        print(f"{column}:")
        print(f"  Unique non-missing values: {unique_count:,}")
        print(f"  Duplicated rows after first occurrence: {duplicate_count:,}")


def group_name_based_leakage_candidates(df: pd.DataFrame) -> None:
    print_section("NAME-BASED LEAKAGE CANDIDATES")

    leakage_keywords_by_reason = {
        "identifier": ["key", "id", "report"],
        "exact date/time": ["date", "time"],
        "injury outcome or total": ["injur"],
        "fatality outcome or total": ["fatal", "kill"],
        "post-incident consequence": ["damage", "cost"],
    }

    grouped_columns: dict[str, list[str]] = {}
    for reason, keywords in leakage_keywords_by_reason.items():
        matches = [
            column for column in df.columns
            if any(keyword in column.lower() for keyword in keywords)
        ]
        grouped_columns[reason] = matches

    for reason, columns in grouped_columns.items():
        print(f"\n{reason}: {len(columns)} column(s)")
        if columns:
            for column in columns:
                marker = " <-- target" if column == TARGET else ""
                print(f"  - {column}{marker}")
        else:
            print("  (none)")

    print(
        "\nNote: this is a conservative name-based scan. "
        "We will turn it into an explicit exclusion function in the next step."
    )


def main() -> None:
    df = load_data(DATA_PATH)
    validate_target(df)
    inspect_columns(df)
    inspect_identifiers(df)
    group_name_based_leakage_candidates(df)


if __name__ == "__main__":
    main()
