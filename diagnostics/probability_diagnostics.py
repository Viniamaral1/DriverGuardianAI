"""
Probability and threshold diagnostics for DriverGuardianAI.

This script determines whether the poor live performance can be
improved through decision thresholds or whether model retraining
is required.

The three recorded sessions are divided chronologically:

- First half: threshold calibration
- Second half: holdout evaluation

This is a diagnostic experiment. The three sessions are too small
and too closely related to serve as a final production benchmark.

Run from the project root:

    python diagnostics/probability_diagnostics.py

Or in Jupyter:

    %run diagnostics/probability_diagnostics.py
"""

import json
from itertools import product
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score
)

from src.predict import Predictor


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIRECTORY = PROJECT_ROOT / "logs"

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "probability_diagnostics"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "driver_guardian_best.pth"
)

PREPROCESSING_PATH = (
    PROJECT_ROOT
    / "models"
    / "preprocessing.pkl"
)


# ============================================================
# DATA CONFIGURATION
# ============================================================

SESSION_FILES = {
    0: LOG_DIRECTORY / "alert.csv",
    1: LOG_DIRECTORY / "mild.csv",
    2: LOG_DIRECTORY / "moderate.csv"
}

CLASS_NAMES = [
    "Alert",
    "Mild Fatigue",
    "Moderate/Severe Fatigue"
]

FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count"
]


# ============================================================
# LOADING
# ============================================================

def load_sessions():
    """
    Load all labelled live sessions.
    """

    calibration_frames = []
    holdout_frames = []

    for actual_class, filepath in SESSION_FILES.items():

        if not filepath.exists():
            raise FileNotFoundError(
                f"Live session was not found: {filepath}"
            )

        dataframe = pd.read_csv(filepath)

        missing_columns = [
            column
            for column in FEATURE_COLUMNS
            if column not in dataframe.columns
        ]

        if missing_columns:
            raise ValueError(
                f"{filepath} is missing features: "
                f"{missing_columns}"
            )

        if dataframe.empty:
            raise ValueError(
                f"{filepath} contains no samples."
            )

        if "timestamp" in dataframe.columns:
            dataframe["timestamp"] = pd.to_datetime(
                dataframe["timestamp"],
                errors="coerce"
            )

            dataframe = dataframe.sort_values(
                "timestamp"
            ).reset_index(drop=True)

        dataframe["actual_class"] = actual_class
        dataframe["actual_label"] = CLASS_NAMES[
            actual_class
        ]

        split_index = max(
            1,
            len(dataframe) // 2
        )

        calibration = dataframe.iloc[
            :split_index
        ].copy()

        holdout = dataframe.iloc[
            split_index:
        ].copy()

        if holdout.empty:
            raise ValueError(
                f"{filepath} does not contain enough rows "
                "for calibration and holdout evaluation."
            )

        calibration["split"] = "calibration"
        holdout["split"] = "holdout"

        calibration_frames.append(calibration)
        holdout_frames.append(holdout)

        print(
            f"{CLASS_NAMES[actual_class]}: "
            f"{len(calibration)} calibration, "
            f"{len(holdout)} holdout"
        )

    calibration_data = pd.concat(
        calibration_frames,
        ignore_index=True
    )

    holdout_data = pd.concat(
        holdout_frames,
        ignore_index=True
    )

    return calibration_data, holdout_data


# ============================================================
# MODEL PROBABILITIES
# ============================================================

def add_model_probabilities(
    dataframe,
    predictor
):
    """
    Run samples through the production Predictor.
    """

    features = dataframe[
        FEATURE_COLUMNS
    ].copy()

    prediction_results = predictor.predict_batch(
        features
    )

    probabilities = []

    predicted_classes = []
    predicted_labels = []
    confidences = []

    for result in prediction_results:

        probability_dictionary = result[
            "probabilities"
        ]

        probabilities.append(
            [
                probability_dictionary["Alert"],
                probability_dictionary["Mild Fatigue"],
                probability_dictionary[
                    "Moderate/Severe Fatigue"
                ]
            ]
        )

        predicted_classes.append(
            result["predicted_class"]
        )

        predicted_labels.append(
            result["prediction"]
        )

        confidences.append(
            result["confidence"]
        )

    probabilities = np.asarray(
        probabilities,
        dtype=float
    )

    output = dataframe.copy()

    output["probability_alert"] = (
        probabilities[:, 0]
    )

    output["probability_mild"] = (
        probabilities[:, 1]
    )

    output["probability_moderate"] = (
        probabilities[:, 2]
    )

    output["argmax_prediction"] = (
        predicted_classes
    )

    output["argmax_label"] = (
        predicted_labels
    )

    output["argmax_confidence"] = (
        confidences
    )

    return output


# ============================================================
# THRESHOLD DECISION RULE
# ============================================================

def apply_thresholds(
    dataframe,
    alert_threshold,
    moderate_threshold
):
    """
    Apply a safety-oriented multiclass threshold rule.

    Decision order:
    1. Predict Moderate when Moderate probability reaches its threshold.
    2. Otherwise predict Alert when Alert probability reaches its threshold.
    3. Otherwise predict Mild.
    """

    alert_probabilities = dataframe[
        "probability_alert"
    ].to_numpy()

    moderate_probabilities = dataframe[
        "probability_moderate"
    ].to_numpy()

    predictions = np.full(
        len(dataframe),
        1,
        dtype=int
    )

    alert_mask = (
        alert_probabilities
        >= alert_threshold
    )

    predictions[alert_mask] = 0

    moderate_mask = (
        moderate_probabilities
        >= moderate_threshold
    )

    predictions[moderate_mask] = 2

    return predictions


def threshold_score(
    y_true,
    y_pred
):
    """
    Score thresholds using macro F1, while penalising
    total failure to recognise any class.
    """

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    predicted_classes = set(
        np.unique(y_pred).tolist()
    )

    class_coverage_penalty = 0.0

    if predicted_classes != {0, 1, 2}:
        class_coverage_penalty = 0.10

    return macro_f1 - class_coverage_penalty


def search_thresholds(
    calibration_data
):
    """
    Search Alert and Moderate thresholds.
    """

    y_true = calibration_data[
        "actual_class"
    ].astype(int).to_numpy()

    alert_thresholds = np.arange(
        0.05,
        0.91,
        0.025
    )

    moderate_thresholds = np.arange(
        0.20,
        0.96,
        0.025
    )

    rows = []

    best_result = None
    best_score = -np.inf

    for alert_threshold, moderate_threshold in product(
        alert_thresholds,
        moderate_thresholds
    ):

        predictions = apply_thresholds(
            calibration_data,
            alert_threshold,
            moderate_threshold
        )

        score = threshold_score(
            y_true,
            predictions
        )

        macro_f1 = f1_score(
            y_true,
            predictions,
            average="macro",
            zero_division=0
        )

        accuracy = accuracy_score(
            y_true,
            predictions
        )

        alert_recall = recall_score(
            y_true,
            predictions,
            labels=[0],
            average="macro",
            zero_division=0
        )

        moderate_recall = recall_score(
            y_true,
            predictions,
            labels=[2],
            average="macro",
            zero_division=0
        )

        row = {
            "alert_threshold": float(
                alert_threshold
            ),
            "moderate_threshold": float(
                moderate_threshold
            ),
            "selection_score": float(score),
            "accuracy": float(accuracy),
            "macro_f1": float(macro_f1),
            "alert_recall": float(alert_recall),
            "moderate_recall": float(
                moderate_recall
            )
        }

        rows.append(row)

        if score > best_score:
            best_score = score
            best_result = row

    results = pd.DataFrame(rows)

    results = results.sort_values(
        [
            "selection_score",
            "macro_f1",
            "accuracy"
        ],
        ascending=False
    ).reset_index(drop=True)

    return best_result, results


# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(
    y_true,
    y_pred
):
    """
    Calculate multiclass metrics.
    """

    return {
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),
        "precision_weighted": float(
            precision_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "recall_weighted": float(
            recall_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "f1_weighted": float(
            f1_score(
                y_true,
                y_pred,
                average="weighted",
                zero_division=0
            )
        ),
        "f1_macro": float(
            f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        )
    }


def evaluate_predictions(
    dataframe,
    predictions,
    name
):
    """
    Evaluate and print one prediction strategy.
    """

    y_true = dataframe[
        "actual_class"
    ].astype(int).to_numpy()

    metrics = calculate_metrics(
        y_true,
        predictions
    )

    report = classification_report(
        y_true,
        predictions,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1, 2]
    )

    print("\n" + "=" * 72)
    print(name)
    print("=" * 72)

    print("\nMetrics:")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    print("\nConfusion matrix:")
    print(matrix)

    print("\nClassification report:")
    print(report)

    return metrics, report, matrix


# ============================================================
# PLOTS
# ============================================================

def save_probability_boxplots(
    dataframe
):
    """
    Plot each model probability by actual session.
    """

    probability_columns = [
        (
            "probability_alert",
            "Alert probability"
        ),
        (
            "probability_mild",
            "Mild probability"
        ),
        (
            "probability_moderate",
            "Moderate probability"
        )
    ]

    for column, title in probability_columns:

        values = []

        for class_id in [0, 1, 2]:
            values.append(
                dataframe[
                    dataframe["actual_class"]
                    == class_id
                ][column].to_numpy()
            )

        plt.figure(figsize=(9, 6))

        plt.boxplot(
            values,
            tick_labels=CLASS_NAMES,
            showfliers=False
        )

        plt.ylabel(title)
        plt.xlabel("Actual recorded session")

        plt.title(
            f"{title} by Live Session"
        )

        plt.xticks(
            rotation=20,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            OUTPUT_DIRECTORY
            / f"{column}_by_session.png",
            dpi=300
        )

        plt.close()


def save_confusion_matrix_plot(
    matrix,
    filename,
    title
):
    """
    Save a confusion matrix image.
    """

    plt.figure(figsize=(8, 6))

    plt.imshow(matrix)
    plt.colorbar()

    plt.xticks(
        [0, 1, 2],
        CLASS_NAMES,
        rotation=25,
        ha="right"
    )

    plt.yticks(
        [0, 1, 2],
        CLASS_NAMES
    )

    for row_index in range(3):
        for column_index in range(3):

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
                va="center"
            )

    plt.xlabel("Predicted class")
    plt.ylabel("Actual session")
    plt.title(title)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIRECTORY / filename,
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run probability and threshold diagnostics.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("DriverGuardianAI Probability Diagnostics")
    print("=" * 72)

    (
        calibration_data,
        holdout_data
    ) = load_sessions()

    predictor = Predictor(
        model_path=str(MODEL_PATH),
        preprocessing_path=str(
            PREPROCESSING_PATH
        ),
        hidden_dims=[
            256,
            128,
            64
        ],
        dropout=0.30,
        num_classes=3
    )

    print(
        "\nCalculating calibration probabilities..."
    )

    calibration_data = add_model_probabilities(
        calibration_data,
        predictor
    )

    print(
        "Calculating holdout probabilities..."
    )

    holdout_data = add_model_probabilities(
        holdout_data,
        predictor
    )

    calibration_data.to_csv(
        OUTPUT_DIRECTORY
        / "calibration_probabilities.csv",
        index=False
    )

    holdout_data.to_csv(
        OUTPUT_DIRECTORY
        / "holdout_probabilities.csv",
        index=False
    )

    print("\nSearching thresholds...")

    (
        best_thresholds,
        threshold_results
    ) = search_thresholds(
        calibration_data
    )

    threshold_results.to_csv(
        OUTPUT_DIRECTORY
        / "threshold_search_results.csv",
        index=False
    )

    alert_threshold = best_thresholds[
        "alert_threshold"
    ]

    moderate_threshold = best_thresholds[
        "moderate_threshold"
    ]

    print("\nSelected thresholds:")

    print(
        f"Alert threshold: {alert_threshold:.3f}"
    )

    print(
        "Moderate threshold: "
        f"{moderate_threshold:.3f}"
    )

    holdout_true = holdout_data[
        "actual_class"
    ].astype(int).to_numpy()

    argmax_predictions = holdout_data[
        "argmax_prediction"
    ].astype(int).to_numpy()

    threshold_predictions = apply_thresholds(
        holdout_data,
        alert_threshold,
        moderate_threshold
    )

    (
        argmax_metrics,
        argmax_report,
        argmax_matrix
    ) = evaluate_predictions(
        holdout_data,
        argmax_predictions,
        "Holdout Evaluation — Original Argmax"
    )

    (
        threshold_metrics,
        threshold_report,
        threshold_matrix
    ) = evaluate_predictions(
        holdout_data,
        threshold_predictions,
        "Holdout Evaluation — Tuned Thresholds"
    )

    holdout_output = holdout_data.copy()

    holdout_output[
        "threshold_prediction"
    ] = threshold_predictions

    holdout_output[
        "threshold_label"
    ] = [
        CLASS_NAMES[class_id]
        for class_id in threshold_predictions
    ]

    holdout_output.to_csv(
        OUTPUT_DIRECTORY
        / "holdout_threshold_predictions.csv",
        index=False
    )

    summary = {
        "selected_thresholds": {
            "alert": float(
                alert_threshold
            ),
            "moderate": float(
                moderate_threshold
            )
        },
        "argmax_holdout_metrics": (
            argmax_metrics
        ),
        "threshold_holdout_metrics": (
            threshold_metrics
        ),
        "diagnostic_interpretation": {
            "thresholds_help": bool(
                threshold_metrics["f1_macro"]
                > argmax_metrics["f1_macro"] + 0.10
            ),
            "retraining_recommended": bool(
                threshold_metrics["f1_macro"]
                < 0.60
            )
        }
    }

    with (
        OUTPUT_DIRECTORY
        / "probability_diagnostics.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    with (
        OUTPUT_DIRECTORY
        / "argmax_holdout_report.txt"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(argmax_report)

    with (
        OUTPUT_DIRECTORY
        / "threshold_holdout_report.txt"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(threshold_report)

    save_probability_boxplots(
        pd.concat(
            [
                calibration_data,
                holdout_data
            ],
            ignore_index=True
        )
    )

    save_confusion_matrix_plot(
        argmax_matrix,
        "argmax_holdout_confusion_matrix.png",
        "Live Holdout Confusion Matrix — Argmax"
    )

    save_confusion_matrix_plot(
        threshold_matrix,
        "threshold_holdout_confusion_matrix.png",
        "Live Holdout Confusion Matrix — Tuned Thresholds"
    )

    print("\n" + "=" * 72)

    if summary[
        "diagnostic_interpretation"
    ]["thresholds_help"]:

        print(
            "Thresholding materially improved macro F1."
        )

    else:

        print(
            "Thresholding did not provide a large improvement."
        )

    if summary[
        "diagnostic_interpretation"
    ]["retraining_recommended"]:

        print(
            "Recommendation: retrain the model using "
            "live-aligned feature generation and validation."
        )

    else:

        print(
            "Recommendation: thresholds may be sufficient "
            "for the next prototype."
        )

    print(
        f"\nResults saved to: {OUTPUT_DIRECTORY}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()