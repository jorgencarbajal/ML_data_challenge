import time

import numpy as np
import pandas as pd
import torch
from pytorch_tabnet.tab_model import TabNetClassifier
from sklearn.metrics import average_precision_score, confusion_matrix

from common import (
    RANDOM_SEED,
    TABNET_FEATURE_IMPORTANCE_PATH,
    TABNET_FIT_PARAMS,
    TABNET_MODEL_PATH,
    TABNET_PARAMS,
    TABNET_VALIDATION_RESULTS_PATH,
    TARGET,
    find_best_f1_threshold,
    load_prepared_tabnet_data,
    make_metric_row,
    print_section,
    save_json,
)


def to_model_arrays(
    X: pd.DataFrame,
    y: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    return X.to_numpy(dtype=np.float32), y.to_numpy(dtype=np.int64)


def build_tabnet_classifier(metadata: dict) -> TabNetClassifier:
    tabnet_metadata = metadata["tabnet"]

    return TabNetClassifier(
        **TABNET_PARAMS,
        cat_idxs=tabnet_metadata["cat_idxs"],
        cat_dims=tabnet_metadata["cat_dims"],
    )


def print_training_setup(metadata: dict) -> None:
    print_section("TABNET TRAINING SETUP")
    print(f"Target: {TARGET}")
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Torch version: {torch.__version__}")
    print("Device: CPU")

    print("\nModel parameters:")
    for key, value in TABNET_PARAMS.items():
        print(f"  {key}: {value}")

    print("\nFit parameters:")
    for key, value in TABNET_FIT_PARAMS.items():
        print(f"  {key}: {value}")

    print("\nEmbedding metadata:")
    print(f"  categorical features: {len(metadata['tabnet']['cat_idxs'])}")
    print(f"  cat_idxs: {metadata['tabnet']['cat_idxs']}")
    print(f"  cat_dims: {metadata['tabnet']['cat_dims']}")


def print_validation_summary(
    y_val: pd.Series,
    val_probabilities: np.ndarray,
    best_threshold: pd.Series,
) -> None:
    threshold = float(best_threshold["threshold"])
    predictions = (val_probabilities >= threshold).astype(int)

    print_section("VALIDATION RESULTS")
    print(f"Validation PR-AUC / Average Precision: {average_precision_score(y_val, val_probabilities):.4f}")

    print("\nBest validation F1 threshold:")
    print(f"Threshold: {threshold:.4f}")
    print(f"F1: {best_threshold['f1']:.4f}")
    print(f"Precision: {best_threshold['precision']:.4f}")
    print(f"Recall: {best_threshold['recall']:.4f}")
    print(f"Predicted injury cases: {int(best_threshold['predicted_positive_count']):,}")

    print("\nValidation confusion matrix at selected threshold:")
    print("Rows = actual class, columns = predicted class")
    print(confusion_matrix(y_val, predictions))


def save_training_outputs(
    model: TabNetClassifier,
    X_train: pd.DataFrame,
    y_val: pd.Series,
    val_probabilities: np.ndarray,
    best_threshold: pd.Series,
    elapsed_seconds: float,
) -> None:
    threshold = float(best_threshold["threshold"])
    validation_results = {
        "model": "TabNetClassifier",
        "target": TARGET,
        "training_seconds": round(elapsed_seconds, 2),
        "threshold_selection": {
            "source": "validation",
            "threshold": round(threshold, 4),
            "validation_f1_at_threshold": round(float(best_threshold["f1"]), 4),
            "validation_precision_at_threshold": round(float(best_threshold["precision"]), 4),
            "validation_recall_at_threshold": round(float(best_threshold["recall"]), 4),
        },
        "metrics": {
            "validation": make_metric_row(y_val, val_probabilities, threshold),
        },
        "tabnet_params": TABNET_PARAMS,
        "tabnet_fit_params": TABNET_FIT_PARAMS,
        "history": {
            key: [float(value) for value in values]
            for key, values in model.history.history.items()
        },
    }

    save_json(validation_results, TABNET_VALIDATION_RESULTS_PATH)
    model.save_model(str(TABNET_MODEL_PATH))

    importance = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)
    importance.to_csv(TABNET_FEATURE_IMPORTANCE_PATH, index=False)

    print_section("FILES WRITTEN")
    print(f"Validation results: {TABNET_VALIDATION_RESULTS_PATH}")
    print(f"Saved model path prefix: {TABNET_MODEL_PATH}")
    print(f"Feature importance: {TABNET_FEATURE_IMPORTANCE_PATH}")


def main() -> None:
    X_train, X_val, _, y_train, y_val, _, metadata = load_prepared_tabnet_data()

    print_section("PREPARED DATA LOADED")
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Train target rows: {len(y_train):,}")
    print(f"Validation target rows: {len(y_val):,}")

    print_training_setup(metadata)

    X_train_array, y_train_array = to_model_arrays(X_train, y_train)
    X_val_array, y_val_array = to_model_arrays(X_val, y_val)

    model = build_tabnet_classifier(metadata)

    start_time = time.perf_counter()
    model.fit(
        X_train=X_train_array,
        y_train=y_train_array,
        eval_set=[(X_val_array, y_val_array)],
        eval_name=["validation"],
        **TABNET_FIT_PARAMS,
    )
    elapsed_seconds = time.perf_counter() - start_time

    print_section("TRAINING COMPLETE")
    print(f"Elapsed training seconds: {elapsed_seconds:.2f}")

    val_probabilities = model.predict_proba(X_val_array)[:, 1]
    best_threshold = find_best_f1_threshold(y_val, val_probabilities)

    print_validation_summary(
        y_val=y_val,
        val_probabilities=val_probabilities,
        best_threshold=best_threshold,
    )
    save_training_outputs(
        model=model,
        X_train=X_train,
        y_val=y_val,
        val_probabilities=val_probabilities,
        best_threshold=best_threshold,
        elapsed_seconds=elapsed_seconds,
    )

    print_section("NEXT STEP")
    print("The selected threshold came from validation data only.")
    print("Run 04_evaluate_test_tabnet.py next to evaluate the frozen model on test data.")


if __name__ == "__main__":
    main()
