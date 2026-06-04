import pandas as pd
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import confusion_matrix

from common import (
    RANDOM_SEED,
    TABNET_FEATURE_IMPORTANCE_PATH,
    TABNET_MODEL_ZIP_PATH,
    TABNET_RESULTS_PATH,
    TABNET_VALIDATION_RESULTS_PATH,
    TARGET,
    TEST_SIZE,
    VALIDATION_SIZE,
    load_json,
    load_prepared_tabnet_data,
    make_metric_row,
    print_section,
    save_json,
)


def to_model_array(X: pd.DataFrame):
    return X.to_numpy(dtype="float32")


def load_trained_model() -> TabNetClassifier:
    if not TABNET_MODEL_ZIP_PATH.exists():
        raise FileNotFoundError(
            "Saved TabNet model not found. Run 03_train_validate_tabnet.py first. "
            f"Expected: {TABNET_MODEL_ZIP_PATH}"
        )

    model = TabNetClassifier()
    model.load_model(str(TABNET_MODEL_ZIP_PATH))

    return model


def print_final_results(
    validation_results: dict,
    y_val: pd.Series,
    y_test: pd.Series,
    val_probabilities,
    test_probabilities,
    threshold: float,
) -> dict:
    print_section("FINAL TABNET METRICS")

    results = {
        "validation": make_metric_row(y_val, val_probabilities, threshold),
        "test": make_metric_row(y_test, test_probabilities, threshold),
    }

    comparison_table = pd.DataFrame([
        {"split": "validation", **results["validation"]},
        {"split": "test", **results["test"]},
    ])

    print(f"Frozen threshold selected on validation data: {threshold:.4f}")
    print(
        "Validation F1 stored from training script: "
        f"{validation_results['threshold_selection']['validation_f1_at_threshold']:.4f}"
    )
    print("\nOfficial comparison metrics:")
    print(comparison_table.to_string(index=False))

    print("\nValidation confusion matrix at frozen threshold:")
    print("Rows = actual class, columns = predicted class")
    print(confusion_matrix(y_val, val_probabilities >= threshold))

    print("\nTest confusion matrix at frozen threshold:")
    print("Rows = actual class, columns = predicted class")
    print(confusion_matrix(y_test, test_probabilities >= threshold))

    return results


def save_outputs(
    metadata: dict,
    validation_results: dict,
    final_metrics: dict,
    threshold: float,
    train_rows: int,
) -> None:
    results = {
        "model": "TabNetClassifier",
        "target": TARGET,
        "random_seed": RANDOM_SEED,
        "split": {
            "test_size": TEST_SIZE,
            "validation_size_of_remaining_data": VALIDATION_SIZE,
            "train_rows": train_rows,
            "validation_rows": final_metrics["validation"]["rows"],
            "test_rows": final_metrics["test"]["rows"],
        },
        "features": {
            "raw_predictor_count": metadata["raw_predictor_count"],
            "numeric_count": len(metadata["feature_groups"]["numeric"]),
            "categorical_count": len(metadata["feature_groups"]["categorical"]),
            "binary_count": len(metadata["feature_groups"]["binary"]),
            "cat_idxs": metadata["tabnet"]["cat_idxs"],
            "cat_dims": metadata["tabnet"]["cat_dims"],
        },
        "threshold_selection": {
            "source": "validation",
            "threshold": round(threshold, 4),
            "validation_f1_at_threshold": (
                validation_results["threshold_selection"]["validation_f1_at_threshold"]
            ),
        },
        "training": {
            "training_seconds": validation_results["training_seconds"],
            "tabnet_params": validation_results["tabnet_params"],
            "tabnet_fit_params": validation_results["tabnet_fit_params"],
        },
        "metrics": final_metrics,
    }

    save_json(results, TABNET_RESULTS_PATH)

    print_section("FILES WRITTEN")
    print(f"Final TabNet results: {TABNET_RESULTS_PATH}")
    print(f"Feature importance file: {TABNET_FEATURE_IMPORTANCE_PATH}")


def main() -> None:
    X_train, X_val, X_test, _, y_val, y_test, metadata = load_prepared_tabnet_data()
    validation_results = load_json(TABNET_VALIDATION_RESULTS_PATH)
    threshold = float(validation_results["threshold_selection"]["threshold"])

    print_section("PREPARED DATA LOADED")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")
    print(f"Validation target rows: {len(y_val):,}")
    print(f"Test target rows: {len(y_test):,}")

    print_section("FROZEN MODEL AND THRESHOLD")
    print(f"Model path: {TABNET_MODEL_ZIP_PATH}")
    print(f"Threshold source: validation")
    print(f"Frozen threshold: {threshold:.4f}")

    model = load_trained_model()
    val_probabilities = model.predict_proba(to_model_array(X_val))[:, 1]
    test_probabilities = model.predict_proba(to_model_array(X_test))[:, 1]

    final_metrics = print_final_results(
        validation_results=validation_results,
        y_val=y_val,
        y_test=y_test,
        val_probabilities=val_probabilities,
        test_probabilities=test_probabilities,
        threshold=threshold,
    )
    save_outputs(
        metadata=metadata,
        validation_results=validation_results,
        final_metrics=final_metrics,
        threshold=threshold,
        train_rows=len(X_train),
    )

    print_section("TEST SET RULE")
    print("The held-out test set was not used for fitting or threshold selection.")


if __name__ == "__main__":
    main()
