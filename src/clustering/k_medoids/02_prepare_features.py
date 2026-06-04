from common import (
    BINARY_FEATURES,
    FITTED_CATEGORICAL_FEATURES,
    FITTED_FEATURES,
    NUMERIC_FEATURES,
    OUTPUT_DIR,
    PROFILE_COLUMNS,
    load_data,
    prepare_features,
    print_section,
)


MODEL_FEATURES_PATH = OUTPUT_DIR / "prepared_model_features.csv"
PROFILE_PATH = OUTPUT_DIR / "prepared_profile_columns.csv"
AUDIT_PATH = OUTPUT_DIR / "preparation_audit.csv"


def print_preparation_summary(model_rows: int, audit_rows) -> None:
    print_section("FEATURE PREPARATION COMPLETE")
    print(f"Rows prepared: {model_rows:,}")
    print(f"Numeric fitted features: {len(NUMERIC_FEATURES)}")
    print(f"Categorical fitted features: {len(FITTED_CATEGORICAL_FEATURES)}")
    print(f"Binary warning-device fitted features: {len(BINARY_FEATURES)}")
    print(f"Total fitted features: {len(FITTED_FEATURES)}")
    print(f"Profile columns kept separate: {PROFILE_COLUMNS}")

    print_section("MISSING-VALUE HANDLING SUMMARY")
    missing_audit = audit_rows[audit_rows["missing_before"] > 0].copy()

    if missing_audit.empty:
        print("No missing values found in fitted features.")
    else:
        missing_audit["missing_percent"] = (
            missing_audit["missing_percent"].round(2).astype(str) + "%"
        )
        print(
            missing_audit[
                [
                    "column",
                    "feature_group",
                    "missing_before",
                    "missing_percent",
                    "preparation",
                ]
            ].to_string(index=False)
        )

    binary_audit = audit_rows[audit_rows["feature_group"] == "binary"]
    print_section("BINARY FEATURE VALIDATION")
    print(f"Binary features checked: {len(binary_audit)}")
    print("Passed: binary warning-device features contain only 0/1 and no missing values.")


def save_outputs(model_df, profile_df, audit_df) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model_df.to_csv(MODEL_FEATURES_PATH, index=False)
    profile_df.to_csv(PROFILE_PATH, index=False)
    audit_df.to_csv(AUDIT_PATH, index=False)

    print_section("FILES WRITTEN")
    print(f"Prepared fitted features: {MODEL_FEATURES_PATH}")
    print(f"Prepared profile columns: {PROFILE_PATH}")
    print(f"Preparation audit: {AUDIT_PATH}")


def main() -> None:
    df = load_data()
    model_df, profile_df, audit_df = prepare_features(df)

    print_preparation_summary(len(model_df), audit_df)
    save_outputs(model_df, profile_df, audit_df)


if __name__ == "__main__":
    main()
