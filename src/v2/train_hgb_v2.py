"""
Train the DriverGuardianAI V2 Histogram Gradient Boosting model.

The script uses participant-aware train, calibration, and test files.
It fits preprocessing and the model on training participants only,
selects a Fatigue threshold on the calibration participant only, and
performs final evaluation on untouched test participants.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight


PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "v2" / "train.csv"
CALIBRATION_PATH = PROJECT_ROOT / "data" / "splits" / "v2" / "calibration.csv"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "v2" / "test.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "v2"
MODEL_PATH = MODEL_DIRECTORY / "driver_guardian_hgb_v2.joblib"

RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "hgb_v2"

TARGET_COLUMN = "fatigue_level"
PARTICIPANT_COLUMN = "participant_id"
SESSION_COLUMN = "session_id"

FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count",
]

NUMERIC_FEATURES = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count",
]

CATEGORICAL_FEATURES = ["condition"]
CLASS_NAMES = ["Alert", "Fatigue"]

TARGET_MAPPING = {
    "Alert": 0,
    "Mild Fatigue": 1,
    "Moderate Fatigue": 1,
}

RANDOM_STATE = 42
THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01
FALSE_POSITIVE_PENALTY = 0.10
MINIMUM_DESIRED_SENSITIVITY = 0.60
SENSITIVITY_SHORTFALL_PENALTY = 0.30


def save_json(data: Dict, filepath: Path) -> None:
    with filepath.open("w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)


def convert_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
    mapping = {
        "true": 1.0,
        "false": 0.0,
        "yes": 1.0,
        "no": 0.0,
        "1": 1.0,
        "0": 0.0,
        "1.0": 1.0,
        "0.0": 0.0,
    }

    converted = text.map(mapping)
    converted = converted.fillna(pd.to_numeric(series, errors="coerce"))

    invalid = converted.isna() & series.notna()
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique().tolist()
        raise ValueError(
            f"{column_name} contains unsupported values: {examples[:10]}"
        )

    return converted.astype("float64")


def prepare_split(dataframe: pd.DataFrame, split_name: str) -> pd.DataFrame:
    required = [
        TARGET_COLUMN,
        PARTICIPANT_COLUMN,
        SESSION_COLUMN,
        *FEATURE_COLUMNS,
    ]

    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"{split_name} split is missing columns: {missing}")

    if dataframe.empty:
        raise ValueError(f"{split_name} split contains no rows.")

    dataframe = dataframe.copy()

    unknown = set(
        dataframe[TARGET_COLUMN].dropna().astype(str).unique()
    ).difference(TARGET_MAPPING.keys())

    if unknown:
        raise ValueError(
            f"{split_name} split contains unsupported labels: {sorted(unknown)}"
        )

    dataframe["binary_target"] = (
        dataframe[TARGET_COLUMN].astype(str).map(TARGET_MAPPING).astype(int)
    )

    dataframe["hands_detected"] = convert_boolean_series(
        dataframe["hands_detected"], "hands_detected"
    )
    dataframe["low_light"] = convert_boolean_series(
        dataframe["low_light"], "low_light"
    )

    for feature in [
        "ear",
        "yawn_score",
        "head_tilt",
        "face_confidence",
        "blink_count",
    ]:
        dataframe[feature] = pd.to_numeric(dataframe[feature], errors="coerce")

    dataframe["condition"] = (
        dataframe["condition"].astype(str).str.strip().str.lower()
    )

    return dataframe


def load_splits() -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    for filepath in [TRAIN_PATH, CALIBRATION_PATH, TEST_PATH]:
        if not filepath.exists():
            raise FileNotFoundError(f"Required split file not found: {filepath}")

    train = prepare_split(pd.read_csv(TRAIN_PATH), "train")
    calibration = prepare_split(pd.read_csv(CALIBRATION_PATH), "calibration")
    test = prepare_split(pd.read_csv(TEST_PATH), "test")
    return train, calibration, test


def validate_split_separation(
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
) -> None:
    splits = {
        "train": train,
        "calibration": calibration,
        "test": test,
    }

    pairs = [
        ("train", "calibration"),
        ("train", "test"),
        ("calibration", "test"),
    ]

    for first, second in pairs:
        participant_overlap = set(
            splits[first][PARTICIPANT_COLUMN].astype(str)
        ) & set(splits[second][PARTICIPANT_COLUMN].astype(str))

        if participant_overlap:
            raise RuntimeError(
                f"Participant leakage between {first} and {second}: "
                f"{sorted(participant_overlap)}"
            )

        session_overlap = set(
            splits[first][SESSION_COLUMN].astype(str)
        ) & set(splits[second][SESSION_COLUMN].astype(str))

        if session_overlap:
            raise RuntimeError(
                f"Session leakage between {first} and {second}: "
                f"{sorted(session_overlap)}"
            )


def build_pipeline() -> Pipeline:
    numeric_pipeline = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            (
                "encoder",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
            ),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, NUMERIC_FEATURES),
            ("categorical", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )

    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_iter=350,
        max_leaf_nodes=31,
        min_samples_leaf=30,
        l2_regularization=1.0,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
        random_state=RANDOM_STATE,
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier),
        ]
    )


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = [int(value) for value in matrix.ravel()]

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0.0

    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_fatigue": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_fatigue": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(
            average_precision_score(y_true, probabilities)
        ),
        "false_positive_rate": float(false_positive_rate),
        "true_negative": tn,
        "false_positive": fp,
        "false_negative": fn,
        "true_positive": tp,
    }


def search_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[float, pd.DataFrame]:
    rows = []
    best_threshold = None
    best_score = -np.inf

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + THRESHOLD_STEP,
        THRESHOLD_STEP,
    )

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        metrics = calculate_metrics(
            y_true,
            predictions,
            probabilities,
            threshold,
        )

        shortfall = max(
            0.0,
            MINIMUM_DESIRED_SENSITIVITY - metrics["recall_sensitivity"],
        )

        score = (
            metrics["balanced_accuracy"]
            - FALSE_POSITIVE_PENALTY * metrics["false_positive_rate"]
            - SENSITIVITY_SHORTFALL_PENALTY * shortfall
        )

        metrics["sensitivity_shortfall"] = float(shortfall)
        metrics["selection_score"] = float(score)
        rows.append(metrics)

        if score > best_score:
            best_score = score
            best_threshold = float(threshold)

    if best_threshold is None:
        raise RuntimeError("Threshold search failed.")

    results = pd.DataFrame(rows).sort_values(
        ["selection_score", "balanced_accuracy", "f1_fatigue"],
        ascending=False,
    )

    return best_threshold, results.reset_index(drop=True)


def save_confusion_matrix_plot(
    matrix: np.ndarray,
    filepath: Path,
    title: str,
) -> None:
    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)

    for row_index in range(2):
        for column_index in range(2):
            plt.text(
                column_index,
                row_index,
                str(matrix[row_index, column_index]),
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filepath, dpi=300)
    plt.close()


def save_threshold_plot(
    threshold_results: pd.DataFrame,
    selected_threshold: float,
) -> None:
    ordered = threshold_results.sort_values("threshold")

    plt.figure(figsize=(10, 6))
    plt.plot(
        ordered["threshold"],
        ordered["recall_sensitivity"],
        label="Fatigue sensitivity",
    )
    plt.plot(
        ordered["threshold"],
        ordered["specificity"],
        label="Alert specificity",
    )
    plt.plot(
        ordered["threshold"],
        ordered["balanced_accuracy"],
        label="Balanced accuracy",
    )
    plt.plot(
        ordered["threshold"],
        ordered["f1_fatigue"],
        label="Fatigue F1",
    )
    plt.axvline(
        selected_threshold,
        linestyle="--",
        label=f"Selected threshold {selected_threshold:.2f}",
    )
    plt.xlabel("Fatigue probability threshold")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.05)
    plt.title("DriverGuardianAI V2 Threshold Trade-off")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIRECTORY / "threshold_tradeoff.png", dpi=300)
    plt.close()


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Histogram Gradient Boosting")
    print("=" * 72)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    train, calibration, test = load_splits()
    validate_split_separation(train, calibration, test)

    print(f"\nTraining samples: {len(train)}")
    print(f"Calibration samples: {len(calibration)}")
    print(f"Test samples: {len(test)}")

    training_participants = sorted(
        train[PARTICIPANT_COLUMN].astype(str).unique().tolist()
    )
    calibration_participants = sorted(
        calibration[PARTICIPANT_COLUMN].astype(str).unique().tolist()
    )
    test_participants = sorted(
        test[PARTICIPANT_COLUMN].astype(str).unique().tolist()
    )

    print(f"\nTraining participants: {training_participants}")
    print(f"Calibration participants: {calibration_participants}")
    print(f"Test participants: {test_participants}")

    X_train = train[FEATURE_COLUMNS].copy()
    y_train = train["binary_target"].to_numpy(dtype=int)

    X_calibration = calibration[FEATURE_COLUMNS].copy()
    y_calibration = calibration["binary_target"].to_numpy(dtype=int)

    X_test = test[FEATURE_COLUMNS].copy()
    y_test = test["binary_target"].to_numpy(dtype=int)

    unique_classes, counts = np.unique(y_train, return_counts=True)
    print("\nTraining binary class distribution:")
    for class_id, count in zip(unique_classes, counts):
        print(f"{CLASS_NAMES[int(class_id)]}: {int(count)}")

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    pipeline = build_pipeline()

    print("\nTraining V2 Histogram Gradient Boosting...")
    pipeline.fit(
        X_train,
        y_train,
        classifier__sample_weight=sample_weights,
    )

    transformed_feature_names = (
        pipeline.named_steps["preprocessor"].get_feature_names_out().tolist()
    )

    save_json(
        {
            "raw_feature_columns": FEATURE_COLUMNS,
            "transformed_feature_columns": transformed_feature_names,
        },
        RESULTS_DIRECTORY / "feature_names.json",
    )

    print(f"\nTransformed features: {transformed_feature_names}")

    calibration_probabilities = pipeline.predict_proba(X_calibration)[:, 1]
    selected_threshold, threshold_results = search_threshold(
        y_calibration,
        calibration_probabilities,
    )

    threshold_results.to_csv(
        RESULTS_DIRECTORY / "threshold_search.csv",
        index=False,
    )
    save_threshold_plot(threshold_results, selected_threshold)

    calibration_predictions = (
        calibration_probabilities >= selected_threshold
    ).astype(int)
    calibration_metrics = calculate_metrics(
        y_calibration,
        calibration_predictions,
        calibration_probabilities,
        selected_threshold,
    )
    save_json(
        calibration_metrics,
        RESULTS_DIRECTORY / "calibration_metrics.json",
    )

    print(f"\nSelected Fatigue threshold: {selected_threshold:.2f}")
    print(
        "Calibration balanced accuracy: "
        f"{calibration_metrics['balanced_accuracy']:.3f}"
    )
    print(
        "Calibration sensitivity: "
        f"{calibration_metrics['recall_sensitivity']:.3f}"
    )
    print(
        "Calibration specificity: "
        f"{calibration_metrics['specificity']:.3f}"
    )

    test_probabilities = pipeline.predict_proba(X_test)[:, 1]
    default_predictions = (test_probabilities >= 0.50).astype(int)
    calibrated_predictions = (
        test_probabilities >= selected_threshold
    ).astype(int)

    default_metrics = calculate_metrics(
        y_test,
        default_predictions,
        test_probabilities,
        0.50,
    )
    calibrated_metrics = calculate_metrics(
        y_test,
        calibrated_predictions,
        test_probabilities,
        selected_threshold,
    )

    save_json(
        default_metrics,
        RESULTS_DIRECTORY / "test_metrics_default.json",
    )
    save_json(
        calibrated_metrics,
        RESULTS_DIRECTORY / "test_metrics_calibrated.json",
    )

    default_report = classification_report(
        y_test,
        default_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )
    calibrated_report = classification_report(
        y_test,
        calibrated_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    (RESULTS_DIRECTORY / "classification_report_default.txt").write_text(
        default_report,
        encoding="utf-8",
    )
    (
        RESULTS_DIRECTORY / "classification_report_calibrated.txt"
    ).write_text(
        calibrated_report,
        encoding="utf-8",
    )

    default_matrix = confusion_matrix(
        y_test,
        default_predictions,
        labels=[0, 1],
    )
    calibrated_matrix = confusion_matrix(
        y_test,
        calibrated_predictions,
        labels=[0, 1],
    )

    save_confusion_matrix_plot(
        default_matrix,
        RESULTS_DIRECTORY / "confusion_matrix_default.png",
        "V2 HGB — Default Threshold 0.50",
    )
    save_confusion_matrix_plot(
        calibrated_matrix,
        RESULTS_DIRECTORY / "confusion_matrix_calibrated.png",
        f"V2 HGB — Calibrated Threshold {selected_threshold:.2f}",
    )

    test_predictions = test[
        [
            "timestamp",
            PARTICIPANT_COLUMN,
            SESSION_COLUMN,
            "source_file",
            TARGET_COLUMN,
            *FEATURE_COLUMNS,
        ]
    ].copy()
    test_predictions["actual_binary_class"] = y_test
    test_predictions["fatigue_probability"] = test_probabilities
    test_predictions["default_prediction"] = default_predictions
    test_predictions["calibrated_prediction"] = calibrated_predictions
    test_predictions.to_csv(
        RESULTS_DIRECTORY / "test_predictions.csv",
        index=False,
    )

    summary = {
        "project": "DriverGuardianAI V2",
        "model_type": "HistGradientBoostingClassifier",
        "task": "Alert vs Fatigue",
        "feature_columns": FEATURE_COLUMNS,
        "transformed_feature_columns": transformed_feature_names,
        "training_samples": int(len(train)),
        "calibration_samples": int(len(calibration)),
        "test_samples": int(len(test)),
        "training_participants": training_participants,
        "calibration_participants": calibration_participants,
        "test_participants": test_participants,
        "selected_fatigue_threshold": float(selected_threshold),
        "calibration_metrics": calibration_metrics,
        "test_metrics_default": default_metrics,
        "test_metrics_calibrated": calibrated_metrics,
        "important_notes": [
            "The model uses raw V2 features.",
            "Preprocessing was fitted on training participants only.",
            "The threshold was selected on calibration data only.",
            "Test participants were untouched until final evaluation.",
            "No V1 scaler or encoder was reused.",
        ],
    }
    save_json(summary, RESULTS_DIRECTORY / "training_summary.json")

    bundle = {
        "project_version": "v2",
        "model_name": "hist_gradient_boosting_v2",
        "pipeline": pipeline,
        "fatigue_threshold": float(selected_threshold),
        "feature_columns": FEATURE_COLUMNS,
        "transformed_feature_columns": transformed_feature_names,
        "class_names": CLASS_NAMES,
        "target_mapping": TARGET_MAPPING,
        "training_participants": training_participants,
        "calibration_participants": calibration_participants,
        "test_participants": test_participants,
    }
    joblib.dump(bundle, MODEL_PATH)

    print("\n" + "=" * 72)
    print("Untouched Test — Default Threshold 0.50")
    print("=" * 72)
    print("\nConfusion Matrix:")
    print(default_matrix)
    print("\nClassification Report:")
    print(default_report)
    print("\nMetrics:")
    for key, value in default_metrics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 72)
    print(
        "Untouched Test — Calibrated Threshold "
        f"{selected_threshold:.2f}"
    )
    print("=" * 72)
    print("\nConfusion Matrix:")
    print(calibrated_matrix)
    print("\nClassification Report:")
    print(calibrated_report)
    print("\nMetrics:")
    for key, value in calibrated_metrics.items():
        print(f"{key}: {value}")

    print("\n" + "=" * 72)
    print("V2 HGB training completed successfully.")
    print("=" * 72)
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Results saved to: {RESULTS_DIRECTORY}")


if __name__ == "__main__":
    main()