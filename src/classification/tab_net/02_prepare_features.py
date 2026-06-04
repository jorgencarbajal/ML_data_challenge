from __future__ import annotations

import pandas as pd

from common import (
    CATEGORICAL_MISSING_VALUE,
    CATEGORICAL_UNKNOWN_VALUE,
    FEATURE_METADATA_PATH,
    OUTPUT_DIR,
    PREPARED_TEST_PATH,
    PREPARED_TRAIN_PATH,
    PREPARED_VALIDATION_PATH,
    TARGET,
    TEST_TARGET_PATH,
    TRAIN_TARGET_PATH,
    VALIDATION_TARGET_PATH,
    get_feature_groups,
    get_predictor_columns,
    load_data,
    make_splits,
    print_section,
    save_json,
)


def fit_numeric_imputers(
    X_train: pd.DataFrame,
    numeric_features: list[str],
) -> dict[str, float]:
    medians = {}

    for column in numeric_features:
        values = pd.to_numeric(X_train[column], errors="coerce")
        median = values.median()

        if pd.isna(median):
            raise ValueError(f"Numeric feature has no usable training values: {column}")

        medians[column] = float(median)

    return medians


def fit_binary_imputers(
    X_train: pd.DataFrame,
    binary_features: list[str],
) -> dict[str, int]:
    modes = {}

    for column in binary_features:
        values = pd.to_numeric(X_train[column], errors="coerce")
        observed_values = set(values.dropna().unique().tolist())
        invalid_values = observed_values - {0, 1}

        if invalid_values:
            raise ValueError(
                f"Binary feature {column} contains values outside 0/1: "
                f"{sorted(invalid_values)}"
            )

        mode_values = values.dropna().mode()

        if mode_values.empty:
            raise ValueError(f"Binary feature has no usable training values: {column}")

        modes[column] = int(mode_values.iloc[0])

    return modes


def fit_categorical_encoders(
    X_train: pd.DataFrame,
    categorical_features: list[str],
) -> dict[str, dict]:
    encoders = {}

    for column in categorical_features:
        train_values = (
            X_train[column]
            .astype("string")
            .fillna(CATEGORICAL_MISSING_VALUE)
        )
        categories = sorted(train_values.unique().tolist())
        mapping = {
            category: index
            for index, category in enumerate(categories)
        }
        unknown_code = len(categories)

        encoders[column] = {
            "mapping": mapping,
            "unknown_code": unknown_code,
            "cardinality": unknown_code + 1,
        }

    return encoders


def transform_features(
    X: pd.DataFrame,
    feature_groups: dict[str, list[str]],
    numeric_medians: dict[str, float],
    binary_modes: dict[str, int],
    categorical_encoders: dict[str, dict],
    ordered_features: list[str],
) -> pd.DataFrame:
    transformed = pd.DataFrame(index=X.index)

    for column in feature_groups["numeric"]:
        transformed[column] = (
            pd.to_numeric(X[column], errors="coerce")
            .fillna(numeric_medians[column])
            .astype("float32")
        )

    for column in feature_groups["categorical"]:
        encoder = categorical_encoders[column]
        values = (
            X[column]
            .astype("string")
            .fillna(CATEGORICAL_MISSING_VALUE)
        )
        transformed[column] = (
            values
            .map(encoder["mapping"])
            .fillna(encoder["unknown_code"])
            .astype("int64")
        )

    for column in feature_groups["binary"]:
        transformed[column] = (
            pd.to_numeric(X[column], errors="coerce")
            .fillna(binary_modes[column])
            .astype("int64")
        )

    transformed = transformed[ordered_features]

    if transformed.isna().any().any():
        missing_columns = transformed.columns[transformed.isna().any()].tolist()
        raise ValueError(f"Missing values remain after TabNet preparation: {missing_columns}")

    return transformed


def build_metadata(
    predictors: list[str],
    feature_groups: dict[str, list[str]],
    ordered_features: list[str],
    numeric_medians: dict[str, float],
    binary_modes: dict[str, int],
    categorical_encoders: dict[str, dict],
    X_val_prepared: pd.DataFrame,
    X_test_prepared: pd.DataFrame,
) -> dict:
    cat_idxs = [
        ordered_features.index(column)
        for column in feature_groups["categorical"]
    ]
    cat_dims = [
        categorical_encoders[column]["cardinality"]
        for column in feature_groups["categorical"]
    ]

    categorical_summary = {}

    for column in feature_groups["categorical"]:
        encoder = categorical_encoders[column]
        unknown_code = encoder["unknown_code"]
        categorical_summary[column] = {
            "training_categories": len(encoder["mapping"]),
            "cardinality_including_unknown": encoder["cardinality"],
            "unknown_code": unknown_code,
            "validation_unknown_count": int((X_val_prepared[column] == unknown_code).sum()),
            "test_unknown_count": int((X_test_prepared[column] == unknown_code).sum()),
        }

    return {
        "target": TARGET,
        "raw_predictor_count": len(predictors),
        "ordered_features": ordered_features,
        "feature_groups": feature_groups,
        "tabnet": {
            "cat_idxs": cat_idxs,
            "cat_dims": cat_dims,
            "cat_emb_dim": 1,
        },
        "numeric_medians": numeric_medians,
        "binary_modes": binary_modes,
        "categorical_missing_value": CATEGORICAL_MISSING_VALUE,
        "categorical_unknown_value": CATEGORICAL_UNKNOWN_VALUE,
        "categorical_summary": categorical_summary,
    }


def print_preparation_summary(
    X_train_prepared: pd.DataFrame,
    X_val_prepared: pd.DataFrame,
    X_test_prepared: pd.DataFrame,
    metadata: dict,
) -> None:
    print_section("TABNET PREPARATION SUMMARY")
    print(f"Train shape: {X_train_prepared.shape}")
    print(f"Validation shape: {X_val_prepared.shape}")
    print(f"Test shape: {X_test_prepared.shape}")
    print(f"Prepared feature count: {len(metadata['ordered_features'])}")
    print(f"Categorical embedding features: {len(metadata['tabnet']['cat_idxs'])}")

    print("\nFeature groups:")
    for group_name, columns in metadata["feature_groups"].items():
        print(f"  {group_name}: {len(columns)}")

    print("\nFirst 10 ordered features:")
    for column in metadata["ordered_features"][:10]:
        print(f"  - {column}")

    print("\nFirst 10 categorical embedding positions:")
    for column, cat_idx, cat_dim in zip(
        metadata["feature_groups"]["categorical"][:10],
        metadata["tabnet"]["cat_idxs"][:10],
        metadata["tabnet"]["cat_dims"][:10],
    ):
        print(f"  - {column}: cat_idx={cat_idx}, cat_dim={cat_dim}")

    unknown_rows = []
    for column, summary in metadata["categorical_summary"].items():
        if summary["validation_unknown_count"] or summary["test_unknown_count"]:
            unknown_rows.append({
                "column": column,
                "validation_unknown_count": summary["validation_unknown_count"],
                "test_unknown_count": summary["test_unknown_count"],
            })

    print_section("UNKNOWN CATEGORY CHECK")
    if unknown_rows:
        print(pd.DataFrame(unknown_rows).to_string(index=False))
    else:
        print("No validation/test categories fell outside the training mappings.")

    print_section("MISSING-VALUE CHECK")
    print(f"Train missing values after preparation: {int(X_train_prepared.isna().sum().sum())}")
    print(f"Validation missing values after preparation: {int(X_val_prepared.isna().sum().sum())}")
    print(f"Test missing values after preparation: {int(X_test_prepared.isna().sum().sum())}")


def save_outputs(
    X_train_prepared: pd.DataFrame,
    X_val_prepared: pd.DataFrame,
    X_test_prepared: pd.DataFrame,
    y_train: pd.Series,
    y_val: pd.Series,
    y_test: pd.Series,
    metadata: dict,
) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    X_train_prepared.to_csv(PREPARED_TRAIN_PATH, index=False)
    X_val_prepared.to_csv(PREPARED_VALIDATION_PATH, index=False)
    X_test_prepared.to_csv(PREPARED_TEST_PATH, index=False)
    y_train.to_frame(TARGET).to_csv(TRAIN_TARGET_PATH, index=False)
    y_val.to_frame(TARGET).to_csv(VALIDATION_TARGET_PATH, index=False)
    y_test.to_frame(TARGET).to_csv(TEST_TARGET_PATH, index=False)
    save_json(metadata, FEATURE_METADATA_PATH)

    print_section("FILES WRITTEN")
    print(f"Prepared train features: {PREPARED_TRAIN_PATH}")
    print(f"Prepared validation features: {PREPARED_VALIDATION_PATH}")
    print(f"Prepared test features: {PREPARED_TEST_PATH}")
    print(f"Train target: {TRAIN_TARGET_PATH}")
    print(f"Validation target: {VALIDATION_TARGET_PATH}")
    print(f"Test target: {TEST_TARGET_PATH}")
    print(f"Feature metadata: {FEATURE_METADATA_PATH}")


def main() -> None:
    df = load_data()
    predictors = get_predictor_columns(df)
    feature_groups = get_feature_groups(df, predictors)

    X = df[predictors].copy()
    y = df[TARGET].copy()

    X_train, X_val, X_test, y_train, y_val, y_test = make_splits(X, y)

    ordered_features = (
        feature_groups["numeric"]
        + feature_groups["categorical"]
        + feature_groups["binary"]
    )

    numeric_medians = fit_numeric_imputers(X_train, feature_groups["numeric"])
    binary_modes = fit_binary_imputers(X_train, feature_groups["binary"])
    categorical_encoders = fit_categorical_encoders(
        X_train,
        feature_groups["categorical"],
    )

    X_train_prepared = transform_features(
        X_train,
        feature_groups,
        numeric_medians,
        binary_modes,
        categorical_encoders,
        ordered_features,
    )
    X_val_prepared = transform_features(
        X_val,
        feature_groups,
        numeric_medians,
        binary_modes,
        categorical_encoders,
        ordered_features,
    )
    X_test_prepared = transform_features(
        X_test,
        feature_groups,
        numeric_medians,
        binary_modes,
        categorical_encoders,
        ordered_features,
    )

    metadata = build_metadata(
        predictors=predictors,
        feature_groups=feature_groups,
        ordered_features=ordered_features,
        numeric_medians=numeric_medians,
        binary_modes=binary_modes,
        categorical_encoders=categorical_encoders,
        X_val_prepared=X_val_prepared,
        X_test_prepared=X_test_prepared,
    )

    print_preparation_summary(
        X_train_prepared=X_train_prepared,
        X_val_prepared=X_val_prepared,
        X_test_prepared=X_test_prepared,
        metadata=metadata,
    )
    save_outputs(
        X_train_prepared=X_train_prepared,
        X_val_prepared=X_val_prepared,
        X_test_prepared=X_test_prepared,
        y_train=y_train,
        y_val=y_val,
        y_test=y_test,
        metadata=metadata,
    )


if __name__ == "__main__":
    main()
