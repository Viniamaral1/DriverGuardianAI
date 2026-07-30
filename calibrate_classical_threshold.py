"""
Calibrate the Histogram Gradient Boosting fatigue threshold.

DriverGuardianAI — Experiment 13
--------------------------------
This experiment uses a participant-aware three-way split:

- Training participants: fit preprocessing and model
- Calibration participant: choose the probability threshold
- Test participants: final untouched evaluation

Binary target:
    0 = Alert
    1 = Fatigue

Fatigue combines:
- Mild Fatigue
- Moderate Fatigue
- Severe Fatigue

The threshold is selected using calibration data only. The final test
participants are never used to fit the model or select the threshold.

Outputs
-------
models/driver_guardian_calibrated_hgb.joblib

results/experiment13_calibrated_hgb/
    participant_split.json
    calibration_threshold_search.csv
    calibration_metrics.json
    test_metrics_default.json
    test_metrics_calibrated.json
    test_classification_report_default.txt
    test_classification_report_calibrated.txt
    test_predictions.csv
    confusion_matrix_default.png
    confusion_matrix_calibrated.png
    threshold_tradeoff.png
    probability_distribution.png
"""

import json
from pathlib import Path

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
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


# ============================================================
# PATHS
# ============================================================

DATASET_PATH = Path("data/dataset_exp3.csv")

EXPERIMENT_NAME = "experiment13_calibrated_hgb"

RESULTS_DIRECTORY = (
    Path("results") / EXPERIMENT_NAME
)

MODEL_PATH = Path(
    "models/driver_guardian_calibrated_hgb.joblib"
)

SPLIT_PATH = (
    RESULTS_DIRECTORY / "participant_split.json"
)

THRESHOLD_SEARCH_PATH = (
    RESULTS_DIRECTORY
    / "calibration_threshold_search.csv"
)

CALIBRATION_METRICS_PATH = (
    RESULTS_DIRECTORY
    / "calibration_metrics.json"
)

DEFAULT_TEST_METRICS_PATH = (
    RESULTS_DIRECTORY
    / "test_metrics_default.json"
)

CALIBRATED_TEST_METRICS_PATH = (
    RESULTS_DIRECTORY
    / "test_metrics_calibrated.json"
)

DEFAULT_REPORT_PATH = (
    RESULTS_DIRECTORY
    / "test_classification_report_default.txt"
)

CALIBRATED_REPORT_PATH = (
    RESULTS_DIRECTORY
    / "test_classification_report_calibrated.txt"
)

TEST_PREDICTIONS_PATH = (
    RESULTS_DIRECTORY
    / "test_predictions.csv"
)

DEFAULT_CONFUSION_MATRIX_PATH = (
    RESULTS_DIRECTORY
    / "confusion_matrix_default.png"
)

CALIBRATED_CONFUSION_MATRIX_PATH = (
    RESULTS_DIRECTORY
    / "confusion_matrix_calibrated.png"
)

THRESHOLD_TRADEOFF_PATH = (
    RESULTS_DIRECTORY
    / "threshold_tradeoff.png"
)

PROBABILITY_DISTRIBUTION_PATH = (
    RESULTS_DIRECTORY
    / "probability_distribution.png"
)


# ============================================================
# SETTINGS
# ============================================================

RANDOM_STATE = 42

TARGET_COLUMN = "fatigue_level"

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

CATEGORICAL_FEATURES = [
    "condition",
]

CLASS_NAMES = [
    "Alert",
    "Fatigue",
]

# First split:
# 75% of participants for development, 25% for final test.
TEST_PARTICIPANT_FRACTION = 0.25

# Second split inside development participants:
# one of six participants becomes calibration (~16.7%).
CALIBRATION_PARTICIPANT_FRACTION = 1 / 6

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.90
THRESHOLD_STEP = 0.01

# Threshold selection prioritises balanced performance.
# A small penalty discourages very high false-positive rates.
FALSE_POSITIVE_PENALTY = 0.10


# ============================================================
# DATA PREPARATION
# ============================================================

def extract_participant(source_file):
    """
    Extract participant name from source_file.
    """

    if pd.isna(source_file):
        raise ValueError(
            "A missing source_file value was found."
        )

    value = str(source_file).strip()

    if not value:
        raise ValueError(
            "An empty source_file value was found."
        )

    return value.split(
        "_",
        maxsplit=1
    )[0].lower()


def load_dataset():
    """
    Load the dataset and create the binary target.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    required_columns = {
        "source_file",
        TARGET_COLUMN,
        *FEATURE_COLUMNS,
    }

    missing_columns = required_columns.difference(
        dataframe.columns
    )

    if missing_columns:
        raise ValueError(
            "Dataset is missing columns: "
            f"{sorted(missing_columns)}"
        )

    dataframe = dataframe.copy()

    dataframe["participant"] = dataframe[
        "source_file"
    ].apply(extract_participant)

    target_mapping = {
        "Alert": 0,
        "Mild Fatigue": 1,
        "Moderate Fatigue": 1,
        "Severe Fatigue": 1,
    }

    source_targets = set(
        dataframe[TARGET_COLUMN]
        .dropna()
        .astype(str)
        .unique()
    )

    unknown_targets = source_targets.difference(
        target_mapping.keys()
    )

    if unknown_targets:
        raise ValueError(
            "Unknown target labels found: "
            f"{sorted(unknown_targets)}"
        )

    dataframe["binary_target"] = (
        dataframe[TARGET_COLUMN]
        .astype(str)
        .map(target_mapping)
    )

    dataframe["low_light"] = (
        dataframe["low_light"]
        .astype(str)
        .str.lower()
        .map(
            {
                "true": 1.0,
                "false": 0.0,
                "1": 1.0,
                "0": 0.0,
                "1.0": 1.0,
                "0.0": 0.0,
            }
        )
    )

    for feature in NUMERIC_FEATURES:
        dataframe[feature] = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

    dataframe["condition"] = (
        dataframe["condition"]
        .astype(str)
    )

    return dataframe


# ============================================================
# THREE-WAY PARTICIPANT SPLIT
# ============================================================

def create_three_way_split(dataframe):
    """
    Create training, calibration, and untouched test splits.
    """

    outer_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_PARTICIPANT_FRACTION,
        random_state=RANDOM_STATE,
    )

    development_indices, test_indices = next(
        outer_splitter.split(
            dataframe,
            groups=dataframe["participant"],
        )
    )

    development_dataframe = dataframe.iloc[
        development_indices
    ].copy()

    test_dataframe = dataframe.iloc[
        test_indices
    ].copy()

    inner_splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=CALIBRATION_PARTICIPANT_FRACTION,
        random_state=RANDOM_STATE,
    )

    train_indices, calibration_indices = next(
        inner_splitter.split(
            development_dataframe,
            groups=development_dataframe[
                "participant"
            ],
        )
    )

    train_dataframe = development_dataframe.iloc[
        train_indices
    ].copy()

    calibration_dataframe = development_dataframe.iloc[
        calibration_indices
    ].copy()

    train_participants = sorted(
        train_dataframe[
            "participant"
        ].unique().tolist()
    )

    calibration_participants = sorted(
        calibration_dataframe[
            "participant"
        ].unique().tolist()
    )

    test_participants = sorted(
        test_dataframe[
            "participant"
        ].unique().tolist()
    )

    all_lists = (
        train_participants
        + calibration_participants
        + test_participants
    )

    if len(all_lists) != len(set(all_lists)):
        raise RuntimeError(
            "Participant leakage was detected."
        )

    return (
        train_dataframe,
        calibration_dataframe,
        test_dataframe,
        train_participants,
        calibration_participants,
        test_participants,
    )


# ============================================================
# MODEL
# ============================================================

def build_preprocessor():
    """
    Build preprocessing for mixed tabular features.
    """

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median"),
            ),
            (
                "scaler",
                StandardScaler(),
            ),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent"
                ),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                NUMERIC_FEATURES,
            ),
            (
                "categorical",
                categorical_pipeline,
                CATEGORICAL_FEATURES,
            ),
        ],
        remainder="drop",
    )


def build_model():
    """
    Build the Histogram Gradient Boosting pipeline.
    """

    return Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(),
            ),
            (
                "classifier",
                HistGradientBoostingClassifier(
                    learning_rate=0.06,
                    max_iter=300,
                    max_leaf_nodes=31,
                    min_samples_leaf=30,
                    l2_regularization=1.0,
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    probabilities,
    threshold,
):
    """
    Calculate binary classification metrics.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    sensitivity = (
        true_positive
        / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    specificity = (
        true_negative
        / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )

    false_positive_rate = (
        false_positive
        / (false_positive + true_negative)
        if false_positive + true_negative > 0
        else 0.0
    )

    negative_predictive_value = (
        true_negative
        / (true_negative + false_negative)
        if true_negative + false_negative > 0
        else 0.0
    )

    return {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(y_true, y_pred)
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                y_pred,
            )
        ),
        "precision_fatigue": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "recall_sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "f1_fatigue": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0,
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                probabilities,
            )
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "negative_predictive_value": float(
            negative_predictive_value
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }


# ============================================================
# THRESHOLD SEARCH
# ============================================================

def search_threshold(calibration_true, calibration_probabilities):
    """
    Search thresholds using calibration participants only.

    Selection score:
        balanced_accuracy - penalty * false_positive_rate
    """

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX + THRESHOLD_STEP,
        THRESHOLD_STEP,
    )

    rows = []

    best_threshold = None
    best_score = -np.inf

    for threshold in thresholds:

        predictions = (
            calibration_probabilities
            >= threshold
        ).astype(int)

        metrics = calculate_metrics(
            calibration_true,
            predictions,
            calibration_probabilities,
            threshold,
        )

        selection_score = (
            metrics["balanced_accuracy"]
            - FALSE_POSITIVE_PENALTY
            * metrics["false_positive_rate"]
        )

        metrics["selection_score"] = float(
            selection_score
        )

        rows.append(metrics)

        if selection_score > best_score:
            best_score = selection_score
            best_threshold = float(threshold)

    results = pd.DataFrame(rows)

    results = results.sort_values(
        [
            "selection_score",
            "balanced_accuracy",
            "f1_fatigue",
        ],
        ascending=False,
    ).reset_index(drop=True)

    if best_threshold is None:
        raise RuntimeError(
            "Threshold search did not return a result."
        )

    return best_threshold, results


# ============================================================
# OUTPUT HELPERS
# ============================================================

def save_json(data, path):
    """
    Save a dictionary as JSON.
    """

    with path.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            data,
            file,
            indent=4,
        )


def save_confusion_matrix(
    matrix,
    path,
    title,
):
    """
    Save a confusion matrix image.
    """

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.colorbar()

    plt.xticks(
        [0, 1],
        CLASS_NAMES,
    )

    plt.yticks(
        [0, 1],
        CLASS_NAMES,
    )

    for row_index in range(2):
        for column_index in range(2):
            plt.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index
                    ]
                ),
                ha="center",
                va="center",
                fontsize=12,
            )

    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(title)

    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def save_threshold_plot(search_results):
    """
    Save threshold trade-off curves.
    """

    sorted_results = search_results.sort_values(
        "threshold"
    )

    plt.figure(figsize=(10, 6))

    plt.plot(
        sorted_results["threshold"],
        sorted_results["recall_sensitivity"],
        label="Fatigue sensitivity",
    )

    plt.plot(
        sorted_results["threshold"],
        sorted_results["specificity"],
        label="Alert specificity",
    )

    plt.plot(
        sorted_results["threshold"],
        sorted_results["balanced_accuracy"],
        label="Balanced accuracy",
    )

    plt.plot(
        sorted_results["threshold"],
        sorted_results["f1_fatigue"],
        label="Fatigue F1",
    )

    plt.xlabel("Fatigue probability threshold")
    plt.ylabel("Metric")
    plt.ylim(0.0, 1.05)

    plt.title(
        "Calibration Threshold Trade-off"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        THRESHOLD_TRADEOFF_PATH,
        dpi=300,
    )

    plt.close()


def save_probability_distribution(
    y_true,
    probabilities,
):
    """
    Plot test probabilities by actual class.
    """

    alert_probabilities = probabilities[
        y_true == 0
    ]

    fatigue_probabilities = probabilities[
        y_true == 1
    ]

    plt.figure(figsize=(9, 6))

    plt.hist(
        alert_probabilities,
        bins=30,
        density=True,
        alpha=0.55,
        label="Actual Alert",
    )

    plt.hist(
        fatigue_probabilities,
        bins=30,
        density=True,
        alpha=0.55,
        label="Actual Fatigue",
    )

    plt.xlabel("Predicted Fatigue probability")
    plt.ylabel("Density")

    plt.title(
        "Untouched Test Probability Distribution"
    )

    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PROBABILITY_DISTRIBUTION_PATH,
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run Experiment 13.
    """

    print("=" * 72)
    print("DriverGuardianAI")
    print("Experiment 13: Calibrated Histogram Gradient Boosting")
    print("=" * 72)

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    dataframe = load_dataset()

    print(
        f"\nDataset samples: {len(dataframe)}"
    )

    print(
        "Participants detected: "
        f"{dataframe['participant'].nunique()}"
    )

    (
        train_dataframe,
        calibration_dataframe,
        test_dataframe,
        train_participants,
        calibration_participants,
        test_participants,
    ) = create_three_way_split(dataframe)

    print("\nTraining participants:")

    for participant in train_participants:
        print(f"  - {participant}")

    print("\nCalibration participants:")

    for participant in calibration_participants:
        print(f"  - {participant}")

    print("\nUntouched test participants:")

    for participant in test_participants:
        print(f"  - {participant}")

    print(
        f"\nTraining samples: {len(train_dataframe)}"
    )

    print(
        "Calibration samples: "
        f"{len(calibration_dataframe)}"
    )

    print(
        f"Test samples: {len(test_dataframe)}"
    )

    split_information = {
        "experiment": EXPERIMENT_NAME,
        "training_participants": train_participants,
        "calibration_participants": (
            calibration_participants
        ),
        "test_participants": test_participants,
        "training_samples": int(
            len(train_dataframe)
        ),
        "calibration_samples": int(
            len(calibration_dataframe)
        ),
        "test_samples": int(
            len(test_dataframe)
        ),
    }

    save_json(
        split_information,
        SPLIT_PATH,
    )

    X_train = train_dataframe[
        FEATURE_COLUMNS
    ].copy()

    y_train = train_dataframe[
        "binary_target"
    ].astype(int)

    X_calibration = calibration_dataframe[
        FEATURE_COLUMNS
    ].copy()

    y_calibration = calibration_dataframe[
        "binary_target"
    ].astype(int).to_numpy()

    X_test = test_dataframe[
        FEATURE_COLUMNS
    ].copy()

    y_test = test_dataframe[
        "binary_target"
    ].astype(int).to_numpy()

    model = build_model()

    print(
        "\nTraining Histogram Gradient Boosting..."
    )

    model.fit(
        X_train,
        y_train,
    )

    calibration_probabilities = (
        model.predict_proba(
            X_calibration
        )[:, 1]
    )

    print(
        "\nSearching calibration threshold..."
    )

    (
        selected_threshold,
        threshold_results,
    ) = search_threshold(
        y_calibration,
        calibration_probabilities,
    )

    threshold_results.to_csv(
        THRESHOLD_SEARCH_PATH,
        index=False,
    )

    save_threshold_plot(
        threshold_results
    )

    calibration_predictions = (
        calibration_probabilities
        >= selected_threshold
    ).astype(int)

    calibration_metrics = calculate_metrics(
        y_calibration,
        calibration_predictions,
        calibration_probabilities,
        selected_threshold,
    )

    save_json(
        calibration_metrics,
        CALIBRATION_METRICS_PATH,
    )

    print(
        "\nSelected Fatigue threshold: "
        f"{selected_threshold:.2f}"
    )

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

    test_probabilities = model.predict_proba(
        X_test
    )[:, 1]

    default_predictions = (
        test_probabilities >= 0.50
    ).astype(int)

    calibrated_predictions = (
        test_probabilities
        >= selected_threshold
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
        DEFAULT_TEST_METRICS_PATH,
    )

    save_json(
        calibrated_metrics,
        CALIBRATED_TEST_METRICS_PATH,
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

    with DEFAULT_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(default_report)

    with CALIBRATED_REPORT_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        file.write(calibrated_report)

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

    save_confusion_matrix(
        default_matrix,
        DEFAULT_CONFUSION_MATRIX_PATH,
        "Untouched Test — Default Threshold",
    )

    save_confusion_matrix(
        calibrated_matrix,
        CALIBRATED_CONFUSION_MATRIX_PATH,
        "Untouched Test — Calibrated Threshold",
    )

    save_probability_distribution(
        y_test,
        test_probabilities,
    )

    test_output = test_dataframe[
        [
            "source_file",
            "participant",
            TARGET_COLUMN,
            *FEATURE_COLUMNS,
        ]
    ].copy()

    test_output[
        "actual_binary_class"
    ] = y_test

    test_output[
        "fatigue_probability"
    ] = test_probabilities

    test_output[
        "default_prediction"
    ] = default_predictions

    test_output[
        "calibrated_prediction"
    ] = calibrated_predictions

    test_output.to_csv(
        TEST_PREDICTIONS_PATH,
        index=False,
    )

    joblib.dump(
        {
            "model_name": "hist_gradient_boosting",
            "pipeline": model,
            "fatigue_threshold": float(
                selected_threshold
            ),
            "feature_columns": FEATURE_COLUMNS,
            "class_names": CLASS_NAMES,
            "training_participants": (
                train_participants
            ),
            "calibration_participants": (
                calibration_participants
            ),
            "test_participants": test_participants,
        },
        MODEL_PATH,
    )

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
    print(
        "Experiment 13 completed successfully."
    )
    print("=" * 72)

    print(
        f"\nCalibrated model saved to: {MODEL_PATH}"
    )

    print(
        f"Results saved to: {RESULTS_DIRECTORY}"
    )


if __name__ == "__main__":
    main()