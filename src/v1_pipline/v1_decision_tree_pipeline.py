from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    make_scorer,
    precision_score,
    recall_score,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_text
from sklearn.model_selection import (
    StratifiedKFold,
    cross_validate,
    train_test_split,
)

import pandas as pd
from pandas.api.types import is_numeric_dtype


# ============================================================
# Global configuration
# ============================================================

DATA_PATH = Path("data/v1.csv")
TARGET = "injury_present"

TEST_SIZE = 0.20
RANDOM_STATE = 42

TREE_MAX_DEPTH = 8
TREE_MIN_SAMPLES_LEAF = 100
TREE_CLASS_WEIGHT = {0: 1, 1: 2.0}

NUMERIC_FEATURES = [
    "Train Speed",
    "Estimated Vehicle Speed",
    "Number Vehicle Occupants",
    "Number of Cars",
    "Temperature",
]


CATEGORICAL_FEATURES = [
    "season",
    "time_of_day",
    "Highway User",
    "Highway User Position",
    "Equipment Involved",
    "Equipment Struck",
    "Equipment Type",
    "Track Type",
    "Warning Connected To Signal",
    "Crossing Illuminated",
    "Visibility",
    "Weather Condition",
    "View Obstruction",
    "Highway User Action",
    "Driver Passed Vehicle",
    "Driver In Vehicle",
]


BINARY_FEATURES = [
    "has_gate",
    "has_cantilever_fls",
    "has_standard_fls",
    "has_wig_wags",
    "has_highway_traffic_signals",
    "has_audible",
    "has_crossbucks",
    "has_stop_signs",
    "has_watchman",
    "has_flagged_by_crew",
    "has_other_warning",
    # "has_no_warning_device",
]


LEAKAGE_OR_EXCLUDED_FEATURES = [
    "Report Key",
    "Date",
    "year",
    "Crossing Users Killed",
    "Crossing Users Injured",
    "Employees Killed",
    "Employees Injured",
    "Passengers Killed",
    "Passengers Injured",
    "Total Killed Form 55A",
    "Total Injured Form 55A",
    "Total Killed Form 57",
    "Total Injured Form 57",
    "fatality_present",
    "injury_present",
    "Vehicle Damage Cost",
]


SELECTED_FEATURES = (
    NUMERIC_FEATURES
    + CATEGORICAL_FEATURES
    + BINARY_FEATURES
)


# ============================================================
# Data loading and required-column validation
# ============================================================

def load_and_validate_data(data_path: Path = DATA_PATH) -> pd.DataFrame:
    """Load the saved Version 1 dataset and confirm required fields exist."""

    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found: {data_path}")

    df = pd.read_csv(data_path, low_memory=False)

    print("=" * 75)
    print("DATASET LOADED")
    print("=" * 75)
    print(f"File: {data_path}")
    print(f"Rows: {len(df):,}")
    print(f"Columns: {len(df.columns):,}")

    # Check whether a feature was accidentally listed more than once.
    duplicate_selected_features = (
        pd.Series(SELECTED_FEATURES)
        .loc[pd.Series(SELECTED_FEATURES).duplicated()]
        .tolist()
    )

    if duplicate_selected_features:
        raise ValueError(
            f"Duplicate selected predictors found: {duplicate_selected_features}"
        )

    required_columns = [TARGET] + SELECTED_FEATURES
    missing_required_columns = [
        column for column in required_columns if column not in df.columns
    ]

    print("\nRequired classification columns:")
    print(f"Expected: {len(required_columns)}")
    print(f"Missing: {len(missing_required_columns)}")

    if missing_required_columns:
        print("Missing required columns:")
        for column in missing_required_columns:
            print(f"  - {column}")

        raise ValueError(
            "Required target or predictor columns are missing from v1.csv."
        )

    overlapping_leakage_features = sorted(
        set(SELECTED_FEATURES).intersection(LEAKAGE_OR_EXCLUDED_FEATURES)
    )

    print("\nPredictor leakage overlap check:")
    if overlapping_leakage_features:
        print("FAILED: excluded fields appear in the predictor list:")
        for column in overlapping_leakage_features:
            print(f"  - {column}")

        raise ValueError("Leakage fields must not be used as predictors.")
    else:
        print("Passed: no listed leakage/excluded fields are selected predictors.")

    return df


# ============================================================
# Classification feature-space audit
# ============================================================

def audit_classification_feature_space(df: pd.DataFrame) -> None:
    """Print target, predictor, missingness, and exclusion diagnostics."""

    print("\n" + "=" * 75)
    print("CLASSIFICATION FEATURE-SPACE AUDIT")
    print("=" * 75)

    print("\nApproved predictor counts before encoding:")
    print(f"Numeric predictors:     {len(NUMERIC_FEATURES)}")
    print(f"Categorical predictors: {len(CATEGORICAL_FEATURES)}")
    print(f"Binary predictors:      {len(BINARY_FEATURES)}")
    print(f"Total predictors:       {len(SELECTED_FEATURES)}")

    # --------------------------------------------------------
    # Target distribution
    # --------------------------------------------------------
    print("\n" + "-" * 75)
    print(f"TARGET DISTRIBUTION: {TARGET}")
    print("-" * 75)

    target_summary = (
        df[TARGET]
        .value_counts(dropna=False)
        .rename_axis("Target Value")
        .reset_index(name="Count")
    )

    target_summary["Percent"] = (
        target_summary["Count"] / len(df) * 100
    ).round(2)

    print(target_summary.to_string(index=False))

    missing_target_count = df[TARGET].isna().sum()

    if missing_target_count > 0:
        print(
            f"\nWARNING: target contains {missing_target_count:,} missing values. "
            "Do not model until this is resolved."
        )
    else:
        print("\nPassed: target has no missing values.")

    # --------------------------------------------------------
    # Excluded fields present in the saved dataset
    # --------------------------------------------------------
    print("\n" + "-" * 75)
    print("EXCLUDED / LEAKAGE FIELD CHECK")
    print("-" * 75)

    present_excluded_fields = [
        column for column in LEAKAGE_OR_EXCLUDED_FEATURES
        if column in df.columns
    ]

    missing_excluded_fields = [
        column for column in LEAKAGE_OR_EXCLUDED_FEATURES
        if column not in df.columns
    ]

    print("Excluded fields present in v1.csv:")
    for column in present_excluded_fields:
        print(f"  - {column}")

    if missing_excluded_fields:
        print("\nExcluded fields not present in v1.csv:")
        for column in missing_excluded_fields:
            print(f"  - {column}")

    # --------------------------------------------------------
    # Predictor dtypes and missingness
    # --------------------------------------------------------
    print("\n" + "-" * 75)
    print("PREDICTOR DATA TYPES AND MISSINGNESS")
    print("-" * 75)

    feature_groups = {
        "Numeric": NUMERIC_FEATURES,
        "Categorical": CATEGORICAL_FEATURES,
        "Binary": BINARY_FEATURES,
    }

    audit_rows = []

    for group_name, feature_list in feature_groups.items():
        for feature in feature_list:
            missing_count = df[feature].isna().sum()

            audit_rows.append({
                "Group": group_name,
                "Feature": feature,
                "Dtype": str(df[feature].dtype),
                "Missing": missing_count,
                "Missing %": round(missing_count / len(df) * 100, 2),
                "Unique Nonmissing": df[feature].nunique(dropna=True),
            })

    feature_audit = pd.DataFrame(audit_rows)

    print(feature_audit.to_string(index=False))

    # --------------------------------------------------------
    # Numeric and binary type checks
    # --------------------------------------------------------
    print("\n" + "-" * 75)
    print("NUMERIC / BINARY FORMAT CHECK")
    print("-" * 75)

    nonnumeric_expected_fields = [
        feature
        for feature in NUMERIC_FEATURES + BINARY_FEATURES
        if not is_numeric_dtype(df[feature])
    ]

    if nonnumeric_expected_fields:
        print("Review needed: these expected numeric/binary fields are not numeric:")
        for feature in nonnumeric_expected_fields:
            print(f"  - {feature}: dtype={df[feature].dtype}")
    else:
        print("Passed: all numeric and binary predictors have numeric dtypes.")

    print("\nObserved values in binary warning-device predictors:")
    for feature in BINARY_FEATURES:
        observed_values = df[feature].value_counts(dropna=False).to_dict()
        print(f"  {feature}: {observed_values}")

    # --------------------------------------------------------
    # Columns not included in this model
    # --------------------------------------------------------
    print("\n" + "-" * 75)
    print("COLUMNS NOT SELECTED AS PREDICTORS")
    print("-" * 75)

    selected_or_target = set(SELECTED_FEATURES + [TARGET])

    not_selected_columns = [
        column for column in df.columns
        if column not in selected_or_target
    ]

    print(
        "These columns will not enter X. Review them only to make sure "
        "we did not overlook a legitimate predictor or hidden leakage field."
    )

    for column in not_selected_columns:
        label = (
            "EXCLUDED / KNOWN"
            if column in LEAKAGE_OR_EXCLUDED_FEATURES
            else "NOT CURRENTLY SELECTED"
        )
        print(f"  - {column} [{label}]")


def audit_categorical_levels(
    df: pd.DataFrame,
    rare_percent_threshold: float = 0.50,
    very_small_count_threshold: int = 100,
) -> None:
    """Inspect categorical predictor levels for sparse categories."""

    print("\n" + "=" * 75)
    print("CATEGORICAL LEVEL FREQUENCY AUDIT")
    print("=" * 75)

    summary_rows = []
    rare_level_rows = []

    for feature in CATEGORICAL_FEATURES:
        values_with_missing = df[feature].astype("string").fillna("<MISSING>")
        level_counts = values_with_missing.value_counts(dropna=False)

        smallest_level_count = level_counts.min()
        smallest_level_percent = smallest_level_count / len(df) * 100

        rare_levels = level_counts[
            (level_counts / len(df) * 100) < rare_percent_threshold
        ]

        very_small_levels = level_counts[
            level_counts < very_small_count_threshold
        ]

        summary_rows.append({
            "Feature": feature,
            "Levels Including Missing": len(level_counts),
            "Smallest Level Count": smallest_level_count,
            "Smallest Level %": round(smallest_level_percent, 4),
            f"Levels Below {rare_percent_threshold}%": len(rare_levels),
            f"Levels Below {very_small_count_threshold} Rows": len(very_small_levels),
        })

        for level, count in rare_levels.items():
            rare_level_rows.append({
                "Feature": feature,
                "Level": level,
                "Count": count,
                "Percent": round(count / len(df) * 100, 4),
                "Below 100 Rows": count < very_small_count_threshold,
            })

    summary_table = pd.DataFrame(summary_rows)

    print("\nCategorical feature summary:")
    print(summary_table.to_string(index=False))

    print("\nLevels representing less than 0.50% of all rows:")

    if rare_level_rows:
        rare_level_table = pd.DataFrame(rare_level_rows).sort_values(
            by=["Percent", "Feature"]
        )

        print(rare_level_table.to_string(index=False))
    else:
        print("No categorical levels fall below the threshold.")


def define_X_and_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Create the approved predictor matrix and binary target vector."""

    X = df[SELECTED_FEATURES].copy()
    y = df[TARGET].copy()

    print("\n" + "=" * 75)
    print("X AND y DEFINED")
    print("=" * 75)
    print(f"X rows: {X.shape[0]:,}")
    print(f"X predictors before encoding: {X.shape[1]}")
    print(f"y rows: {len(y):,}")

    unexpected_columns = [
        column for column in X.columns
        if column in LEAKAGE_OR_EXCLUDED_FEATURES
    ]

    if unexpected_columns:
        raise ValueError(
            f"Leakage fields unexpectedly entered X: {unexpected_columns}"
        )

    print("Passed: X contains only approved predictors.")

    return X, y


def split_training_and_test_data(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Create a reproducible stratified train/test split."""

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    print("\n" + "=" * 75)
    print("STRATIFIED TRAIN / TEST SPLIT")
    print("=" * 75)
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Test proportion: {TEST_SIZE:.0%}")
    print(f"Training rows: {len(X_train):,}")
    print(f"Testing rows: {len(X_test):,}")

    split_summaries = []

    for split_name, split_y in [
        ("Full dataset", y),
        ("Training set", y_train),
        ("Testing set", y_test),
    ]:
        injury_count = int((split_y == 1).sum())
        noninjury_count = int((split_y == 0).sum())
        injury_percent = injury_count / len(split_y) * 100
        majority_baseline = max(injury_count, noninjury_count) / len(split_y) * 100

        split_summaries.append({
            "Split": split_name,
            "Rows": len(split_y),
            "Injury Cases": injury_count,
            "No-Injury Cases": noninjury_count,
            "Injury %": round(injury_percent, 2),
            "Majority Baseline Accuracy %": round(majority_baseline, 2),
        })

    split_summary_table = pd.DataFrame(split_summaries)

    print("\nTarget balance by split:")
    print(split_summary_table.to_string(index=False))

    return X_train, X_test, y_train, y_test


def build_preprocessing_pipeline() -> ColumnTransformer:
    """Create preprocessing steps that will later be fitted on training data only."""

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
            ("binary", "passthrough", BINARY_FEATURES),
        ]
    )

    return preprocessor


def audit_preprocessing_output(
    preprocessor: ColumnTransformer,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
) -> None:
    """Fit preprocessing on training data only and inspect transformed output."""

    print("\n" + "=" * 75)
    print("PREPROCESSING PIPELINE AUDIT")
    print("=" * 75)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_test_processed = preprocessor.transform(X_test)

    feature_names = preprocessor.get_feature_names_out()

    print("Passed: preprocessing was fitted on the training set only.")
    print(f"Training rows after preprocessing: {X_train_processed.shape[0]:,}")
    print(f"Testing rows after preprocessing:  {X_test_processed.shape[0]:,}")
    print(f"Predictors before encoding:        {X_train.shape[1]}")
    print(f"Predictors after encoding:         {X_train_processed.shape[1]}")

    print("\nEncoded feature counts by group:")
    print(f"Numeric output features:     {len(NUMERIC_FEATURES)}")
    print(f"Binary output features:      {len(BINARY_FEATURES)}")
    print(
        f"Categorical encoded features: "
        f"{X_train_processed.shape[1] - len(NUMERIC_FEATURES) - len(BINARY_FEATURES)}"
    )

    print("\nFirst 25 transformed feature names:")
    for feature_name in feature_names[:25]:
        print(f"  - {feature_name}")

    print("\nLast 15 transformed feature names:")
    for feature_name in feature_names[-15:]:
        print(f"  - {feature_name}")


def run_decision_tree_baseline(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> Pipeline:
    """Fit an initial constrained Decision Tree inside a preprocessing pipeline."""

    model_pipeline = Pipeline(
        steps=[
            ("preprocessor", build_preprocessing_pipeline()),
            (
                "classifier",
                DecisionTreeClassifier(
                    criterion="gini",
                    max_depth=TREE_MAX_DEPTH,
                    min_samples_leaf=TREE_MIN_SAMPLES_LEAF,
                    class_weight=TREE_CLASS_WEIGHT,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )

    print("\n" + "=" * 75)
    print("DECISION TREE BASELINE TRAINING")
    print("=" * 75)
    print("Criterion: gini")
    print(f"Maximum depth: {TREE_MAX_DEPTH}")
    print(f"Minimum samples per leaf: {TREE_MIN_SAMPLES_LEAF}")
    print(f"Random seed: {RANDOM_STATE}")
    print(f"Class weighting: {TREE_CLASS_WEIGHT}")

    model_pipeline.fit(X_train, y_train)

    print("\nPassed: Decision Tree baseline fitted using training data only.")

    return model_pipeline


def evaluate_predictions(
    model_pipeline: Pipeline,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Evaluate training and test performance for the initial baseline."""

    print("\n" + "=" * 75)
    print("DECISION TREE BASELINE EVALUATION")
    print("=" * 75)

    evaluation_rows = []

    for split_name, X_split, y_split in [
        ("Training", X_train, y_train),
        ("Testing", X_test, y_test),
    ]:
        predicted_classes = model_pipeline.predict(X_split)
        predicted_probabilities = model_pipeline.predict_proba(X_split)[:, 1]

        evaluation_rows.append({
            "Split": split_name,
            "Accuracy": round(accuracy_score(y_split, predicted_classes), 4),
            "Precision": round(
                precision_score(y_split, predicted_classes, zero_division=0), 4
            ),
            "Recall": round(
                recall_score(y_split, predicted_classes, zero_division=0), 4
            ),
            "F1": round(
                f1_score(y_split, predicted_classes, zero_division=0), 4
            ),
            "Average Precision": round(
                average_precision_score(y_split, predicted_probabilities), 4
            ),
        })

        matrix = confusion_matrix(y_split, predicted_classes)

        print(f"\n{split_name} confusion matrix:")
        print("Rows = actual class, columns = predicted class")
        print(matrix)

    evaluation_table = pd.DataFrame(evaluation_rows)

    print("\nPerformance summary:")
    print(evaluation_table.to_string(index=False))

    test_majority_accuracy = max(
        (y_test == 0).mean(),
        (y_test == 1).mean(),
    )

    test_no_skill_average_precision = (y_test == 1).mean()

    print("\nNo-skill reference values for the testing set:")
    print(f"Majority-class accuracy: {test_majority_accuracy:.4f}")
    print(f"No-skill average precision: {test_no_skill_average_precision:.4f}")

    fitted_tree = model_pipeline.named_steps["classifier"]

    print("\nFitted tree behavior:")
    print(f"Actual tree depth: {fitted_tree.get_depth()}")
    print(f"Number of leaves: {fitted_tree.get_n_leaves()}")


def inspect_tree_behavior(model_pipeline: Pipeline) -> None:
    """Inspect which processed features the baseline tree used."""

    print("\n" + "=" * 75)
    print("DECISION TREE BEHAVIOR INSPECTION")
    print("=" * 75)

    fitted_preprocessor = model_pipeline.named_steps["preprocessor"]
    fitted_tree = model_pipeline.named_steps["classifier"]

    feature_names = fitted_preprocessor.get_feature_names_out()
    feature_importances = fitted_tree.feature_importances_

    importance_table = pd.DataFrame({
        "Feature": feature_names,
        "Importance": feature_importances,
    })

    used_features = (
        importance_table[importance_table["Importance"] > 0]
        .sort_values(by="Importance", ascending=False)
        .reset_index(drop=True)
    )

    print(f"Processed features available to tree: {len(feature_names)}")
    print(f"Features actually used in splits:     {len(used_features)}")

    print("\nFeatures used by the tree, ordered by impurity importance:")
    print(used_features.to_string(index=False))

    print("\nTop tree rules, truncated after depth 3 for readability:")
    tree_rules = export_text(
        fitted_tree,
        feature_names=list(feature_names),
        max_depth=3,
        decimals=2,
    )

    print(tree_rules)


def audit_key_predictor_target_relationships(df: pd.DataFrame) -> None:
    """Inspect target prevalence for predictors strongly used by the baseline tree."""

    print("\n" + "=" * 75)
    print("KEY PREDICTOR / TARGET RELATIONSHIP AUDIT")
    print("=" * 75)

    key_features = [
        "Driver In Vehicle",
        "Highway User",
        "Highway User Position",
    ]

    for feature in key_features:
        print("\n" + "-" * 75)
        print(f"{feature} by injury outcome")
        print("-" * 75)

        working_values = df[feature].astype("string").fillna("<MISSING>")

        summary = (
            pd.DataFrame({
                feature: working_values,
                TARGET: df[TARGET],
            })
            .groupby(feature, dropna=False)[TARGET]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .rename(columns={
                "count": "Incidents",
                "sum": "Injury Cases",
                "mean": "Injury Rate",
            })
        )

        summary["Injury Rate %"] = (summary["Injury Rate"] * 100).round(2)
        summary = summary.drop(columns="Injury Rate")
        summary = summary.sort_values("Injury Rate %", ascending=False)

        print(summary.to_string(index=False))


def compare_tree_variations(
    X_train: pd.DataFrame,
    y_train: pd.Series,
) -> None:
    """Compare a small set of Decision Tree settings using training data only."""

    print("\n" + "=" * 75)
    print("CONTROLLED DECISION TREE COMPARISON")
    print("=" * 75)
    print("Evaluation method: 3-fold stratified cross-validation on training data only")
    print("Testing set is not used during this comparison.")

    candidate_settings = [
    {
        "Model": "Current baseline",
        "max_depth": 5,
        "min_samples_leaf": 100,
        "class_weight": None,
    },
    {
        "Model": "Depth 8 unweighted",
        "max_depth": 8,
        "min_samples_leaf": 100,
        "class_weight": None,
    },
    {
        "Model": "Depth 8 weight 1.5",
        "max_depth": 8,
        "min_samples_leaf": 100,
        "class_weight": {0: 1, 1: 1.5},
    },
    {
        "Model": "Depth 8 weight 2.0",
        "max_depth": 8,
        "min_samples_leaf": 100,
        "class_weight": {0: 1, 1: 2.0},
    },
    {
        "Model": "Depth 8 balanced",
        "max_depth": 8,
        "min_samples_leaf": 100,
        "class_weight": "balanced",
    },
]

    cv = StratifiedKFold(
        n_splits=3,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": make_scorer(precision_score, zero_division=0),
        "recall": make_scorer(recall_score, zero_division=0),
        "f1": make_scorer(f1_score, zero_division=0),
        "average_precision": "average_precision",
    }

    comparison_rows = []

    for settings in candidate_settings:
        model_pipeline = Pipeline(
            steps=[
                ("preprocessor", build_preprocessing_pipeline()),
                (
                    "classifier",
                    DecisionTreeClassifier(
                        criterion="gini",
                        max_depth=settings["max_depth"],
                        min_samples_leaf=settings["min_samples_leaf"],
                        class_weight=settings["class_weight"],
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        )

        cv_results = cross_validate(
            model_pipeline,
            X_train,
            y_train,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
        )

        comparison_rows.append({
            "Model": settings["Model"],
            "Max Depth": settings["max_depth"],
            "Min Leaf": settings["min_samples_leaf"],
            "Class Weight": str(settings["class_weight"]),
            "Accuracy": round(cv_results["test_accuracy"].mean(), 4),
            "Precision": round(cv_results["test_precision"].mean(), 4),
            "Recall": round(cv_results["test_recall"].mean(), 4),
            "F1": round(cv_results["test_f1"].mean(), 4),
            "Average Precision": round(
                cv_results["test_average_precision"].mean(), 4
            ),
        })

    comparison_table = pd.DataFrame(comparison_rows)

    print("\nCross-validation performance summary:")
    print(comparison_table.to_string(index=False))

    print("\nReference values from the training target distribution:")
    print(f"Majority-class accuracy: {(y_train == 0).mean():.4f}")
    print(f"No-skill average precision: {(y_train == 1).mean():.4f}")


# ============================================================
# Main
# ============================================================

def main() -> None:
    df = load_and_validate_data()

    X, y = define_X_and_y(df)
    X_train, X_test, y_train, y_test = split_training_and_test_data(X, y)

    # Already completed using training data only:
    # compare_tree_variations(X_train, y_train)

    model_pipeline = run_decision_tree_baseline(X_train, y_train)

    evaluate_predictions(
        model_pipeline,
        X_train,
        X_test,
        y_train,
        y_test,
    )

    inspect_tree_behavior(model_pipeline)


if __name__ == "__main__":
    main()