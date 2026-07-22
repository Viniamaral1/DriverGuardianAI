"""
Evaluate the DriverGuardianAI V2 model on labelled live logs.

Inputs
------
models/v2/driver_guardian_hgb_v2.joblib
logs/hgb_live_alert.csv
logs/hgb_live_fatigue.csv

Outputs
-------
results/v2/live_log_evaluation/
"""

import json
from pathlib import Path
from typing import Any, Dict, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = PROJECT_ROOT / "models" / "v2" / "driver_guardian_hgb_v2.joblib"
LIVE_ALERT_PATH = PROJECT_ROOT / "logs" / "hgb_live_alert.csv"
LIVE_FATIGUE_PATH = PROJECT_ROOT / "logs" / "hgb_live_fatigue.csv"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "live_log_evaluation"

SUMMARY_PATH = RESULTS_DIRECTORY / "evaluation_summary.json"
SESSION_METRICS_PATH = RESULTS_DIRECTORY / "session_metrics.csv"
COMBINED_PREDICTIONS_PATH = RESULTS_DIRECTORY / "combined_predictions.csv"
DEFAULT_CONFUSION_MATRIX_PATH = RESULTS_DIRECTORY / "confusion_matrix_default.png"
CALIBRATED_CONFUSION_MATRIX_PATH = RESULTS_DIRECTORY / "confusion_matrix_calibrated.png"
PROBABILITY_DISTRIBUTION_PATH = RESULTS_DIRECTORY / "probability_distribution.png"
FEATURE_COMPARISON_PATH = RESULTS_DIRECTORY / "feature_comparison.csv"
FEATURE_COMPARISON_PLOT_PATH = RESULTS_DIRECTORY / "feature_comparison.png"

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

NUMERIC_FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count",
]

CLASS_NAMES = ["Alert", "Fatigue"]


def require_file(filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"Required file was not found: {filepath}")


def convert_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    text_values = series.astype(str).str.strip().str.lower()

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

    converted = text_values.map(mapping)
    converted = converted.fillna(pd.to_numeric(series, errors="coerce"))

    invalid_mask = converted.isna() & series.notna()

    if invalid_mask.any():
        examples = series.loc[invalid_mask].astype(str).unique().tolist()
        raise ValueError(
            f"{column_name} contains unsupported values: {examples[:10]}"
        )

    return converted.astype("float64")


def prepare_live_dataframe(
    dataframe: pd.DataFrame,
    actual_label: str,
    source_name: str,
) -> pd.DataFrame:
    missing_columns = [
        column for column in FEATURE_COLUMNS if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{source_name} is missing feature columns: {missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(f"{source_name} contains no rows.")

    dataframe = dataframe.copy()

    dataframe["hands_detected"] = convert_boolean_series(
        dataframe["hands_detected"],
        "hands_detected",
    )

    dataframe["low_light"] = convert_boolean_series(
        dataframe["low_light"],
        "low_light",
    )

    for feature in [
        "ear",
        "yawn_score",
        "head_tilt",
        "face_confidence",
        "blink_count",
    ]:
        dataframe[feature] = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        )

    dataframe["condition"] = (
        dataframe["condition"].astype(str).str.strip().str.lower()
    )

    dataframe["actual_label"] = actual_label
    dataframe["actual_class"] = 0 if actual_label == "Alert" else 1
    dataframe["live_session"] = source_name

    return dataframe


def load_model_bundle() -> Dict[str, Any]:
    require_file(MODEL_PATH)

    bundle = joblib.load(MODEL_PATH)

    required_keys = {
        "pipeline",
        "fatigue_threshold",
        "feature_columns",
        "class_names",
    }

    missing_keys = required_keys.difference(bundle.keys())

    if missing_keys:
        raise KeyError(
            f"V2 model bundle is missing keys: {sorted(missing_keys)}"
        )

    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(
            "V2 model feature order does not match the evaluation script."
        )

    return bundle


def load_live_logs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    require_file(LIVE_ALERT_PATH)
    require_file(LIVE_FATIGUE_PATH)

    alert = prepare_live_dataframe(
        pd.read_csv(LIVE_ALERT_PATH),
        actual_label="Alert",
        source_name="live_alert",
    )

    fatigue = prepare_live_dataframe(
        pd.read_csv(LIVE_FATIGUE_PATH),
        actual_label="Fatigue",
        source_name="live_fatigue",
    )

    return alert, fatigue


def calculate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    sensitivity = (
        true_positive / (true_positive + false_negative)
        if true_positive + false_negative > 0
        else 0.0
    )

    specificity = (
        true_negative / (true_negative + false_positive)
        if true_negative + false_positive > 0
        else 0.0
    )

    false_positive_rate = (
        false_positive / (false_positive + true_negative)
        if false_positive + true_negative > 0
        else 0.0
    )

    metrics = {
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
        "false_positive_rate": float(false_positive_rate),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }

    if len(np.unique(y_true)) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(y_true, probabilities)
        )
        metrics["average_precision"] = float(
            average_precision_score(y_true, probabilities)
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def create_session_metrics(
    dataframe: pd.DataFrame,
    prediction_column: str,
    threshold: float,
    threshold_type: str,
) -> pd.DataFrame:
    rows = []

    for session_name, group in dataframe.groupby("live_session", sort=True):
        actual_label = str(group["actual_label"].iloc[0])
        actual_class = int(group["actual_class"].iloc[0])

        predictions = group[prediction_column].to_numpy(dtype=int)
        probabilities = group["fatigue_probability_v2"].to_numpy(dtype=float)
        correct = predictions == actual_class

        rows.append(
            {
                "threshold_type": threshold_type,
                "live_session": session_name,
                "actual_label": actual_label,
                "threshold": float(threshold),
                "samples": int(len(group)),
                "correct_predictions": int(correct.sum()),
                "frame_accuracy": float(correct.mean()),
                "predicted_alert": int((predictions == 0).sum()),
                "predicted_fatigue": int((predictions == 1).sum()),
                "mean_fatigue_probability": float(probabilities.mean()),
                "minimum_fatigue_probability": float(probabilities.min()),
                "maximum_fatigue_probability": float(probabilities.max()),
            }
        )

    return pd.DataFrame(rows)


def create_feature_comparison(
    live_alert: pd.DataFrame,
    live_fatigue: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for feature in NUMERIC_FEATURE_COLUMNS:
        alert_values = (
            pd.to_numeric(live_alert[feature], errors="coerce")
            .astype("float64")
            .dropna()
        )

        fatigue_values = (
            pd.to_numeric(live_fatigue[feature], errors="coerce")
            .astype("float64")
            .dropna()
        )

        rows.append(
            {
                "feature": feature,
                "live_alert_mean": float(alert_values.mean()),
                "live_fatigue_mean": float(fatigue_values.mean()),
                "fatigue_minus_alert": float(
                    fatigue_values.mean() - alert_values.mean()
                ),
                "live_alert_std": float(alert_values.std(ddof=1)),
                "live_fatigue_std": float(fatigue_values.std(ddof=1)),
            }
        )

    return pd.DataFrame(rows)


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


def save_probability_distribution_plot(
    combined: pd.DataFrame,
    calibrated_threshold: float,
) -> None:
    alert_probabilities = combined.loc[
        combined["actual_class"] == 0,
        "fatigue_probability_v2",
    ]

    fatigue_probabilities = combined.loc[
        combined["actual_class"] == 1,
        "fatigue_probability_v2",
    ]

    plt.figure(figsize=(10, 6))

    plt.hist(
        alert_probabilities,
        bins=30,
        density=True,
        alpha=0.55,
        label="Actual live Alert",
    )

    plt.hist(
        fatigue_probabilities,
        bins=30,
        density=True,
        alpha=0.55,
        label="Actual live Fatigue",
    )

    plt.axvline(
        0.50,
        linestyle="--",
        label="Default threshold 0.50",
    )

    plt.axvline(
        calibrated_threshold,
        linestyle="--",
        label=f"Calibrated threshold {calibrated_threshold:.2f}",
    )

    plt.xlabel("V2 predicted Fatigue probability")
    plt.ylabel("Density")
    plt.title("V2 Model on Existing Live Sessions")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PROBABILITY_DISTRIBUTION_PATH, dpi=300)
    plt.close()


def save_feature_comparison_plot(
    comparison: pd.DataFrame,
) -> None:
    plot_data = comparison.copy()

    maximum_values = plot_data[
        ["live_alert_mean", "live_fatigue_mean"]
    ].abs().max(axis=1)

    maximum_values = maximum_values.replace(0.0, 1.0)

    plot_data["alert_normalised"] = (
        plot_data["live_alert_mean"] / maximum_values
    )

    plot_data["fatigue_normalised"] = (
        plot_data["live_fatigue_mean"] / maximum_values
    )

    axis = plot_data.set_index("feature")[
        ["alert_normalised", "fatigue_normalised"]
    ].plot(
        kind="bar",
        figsize=(11, 6),
    )

    axis.set_ylabel("Mean relative to feature maximum")
    axis.set_xlabel("Feature")
    axis.set_title("Live Alert vs Live Fatigue Feature Means")

    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(FEATURE_COMPARISON_PLOT_PATH, dpi=300)
    plt.close()


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Live Log Evaluation")
    print("=" * 72)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nLoading V2 model...")

    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    calibrated_threshold = float(bundle["fatigue_threshold"])

    print(f"Model: {bundle.get('model_name', 'unknown')}")
    print(f"Calibrated threshold: {calibrated_threshold:.2f}")

    print("\nLoading labelled live logs...")

    live_alert, live_fatigue = load_live_logs()

    print(f"Live Alert samples: {len(live_alert)}")
    print(f"Live Fatigue samples: {len(live_fatigue)}")

    combined = pd.concat(
        [live_alert, live_fatigue],
        ignore_index=True,
    )

    features = combined[FEATURE_COLUMNS].copy()
    probabilities = pipeline.predict_proba(features)[:, 1]

    combined["fatigue_probability_v2"] = probabilities
    combined["prediction_default"] = (
        probabilities >= 0.50
    ).astype(int)
    combined["prediction_calibrated"] = (
        probabilities >= calibrated_threshold
    ).astype(int)

    combined["prediction_default_label"] = combined[
        "prediction_default"
    ].map({0: "Alert", 1: "Fatigue"})

    combined["prediction_calibrated_label"] = combined[
        "prediction_calibrated"
    ].map({0: "Alert", 1: "Fatigue"})

    y_true = combined["actual_class"].to_numpy(dtype=int)
    default_predictions = combined[
        "prediction_default"
    ].to_numpy(dtype=int)
    calibrated_predictions = combined[
        "prediction_calibrated"
    ].to_numpy(dtype=int)

    default_metrics = calculate_metrics(
        y_true,
        default_predictions,
        probabilities,
        0.50,
    )

    calibrated_metrics = calculate_metrics(
        y_true,
        calibrated_predictions,
        probabilities,
        calibrated_threshold,
    )

    default_matrix = confusion_matrix(
        y_true,
        default_predictions,
        labels=[0, 1],
    )

    calibrated_matrix = confusion_matrix(
        y_true,
        calibrated_predictions,
        labels=[0, 1],
    )

    default_report = classification_report(
        y_true,
        default_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    calibrated_report = classification_report(
        y_true,
        calibrated_predictions,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    default_session_metrics = create_session_metrics(
        combined,
        prediction_column="prediction_default",
        threshold=0.50,
        threshold_type="default",
    )

    calibrated_session_metrics = create_session_metrics(
        combined,
        prediction_column="prediction_calibrated",
        threshold=calibrated_threshold,
        threshold_type="calibrated",
    )

    session_report = pd.concat(
        [default_session_metrics, calibrated_session_metrics],
        ignore_index=True,
    )

    feature_comparison = create_feature_comparison(
        live_alert,
        live_fatigue,
    )

    session_report.to_csv(
        SESSION_METRICS_PATH,
        index=False,
    )

    combined.to_csv(
        COMBINED_PREDICTIONS_PATH,
        index=False,
    )

    feature_comparison.to_csv(
        FEATURE_COMPARISON_PATH,
        index=False,
    )

    save_confusion_matrix_plot(
        default_matrix,
        DEFAULT_CONFUSION_MATRIX_PATH,
        "V2 Live Logs — Default Threshold 0.50",
    )

    save_confusion_matrix_plot(
        calibrated_matrix,
        CALIBRATED_CONFUSION_MATRIX_PATH,
        (
            "V2 Live Logs — Calibrated Threshold "
            f"{calibrated_threshold:.2f}"
        ),
    )

    save_probability_distribution_plot(
        combined,
        calibrated_threshold,
    )

    save_feature_comparison_plot(
        feature_comparison
    )

    summary = {
        "project": "DriverGuardianAI V2",
        "model_path": str(MODEL_PATH),
        "live_alert_path": str(LIVE_ALERT_PATH),
        "live_fatigue_path": str(LIVE_FATIGUE_PATH),
        "calibrated_threshold": calibrated_threshold,
        "live_alert_samples": int(len(live_alert)),
        "live_fatigue_samples": int(len(live_fatigue)),
        "default_metrics": default_metrics,
        "calibrated_metrics": calibrated_metrics,
        "session_metrics": session_report.to_dict(
            orient="records"
        ),
        "important_note": (
            "These logs were generated before V2 deployment. "
            "They test feature compatibility, but fresh V2 live "
            "recordings are still required."
        ),
    }

    with SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("Combined Live Evaluation — Default Threshold 0.50")
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
        "Combined Live Evaluation — Calibrated Threshold "
        f"{calibrated_threshold:.2f}"
    )
    print("=" * 72)
    print("\nConfusion Matrix:")
    print(calibrated_matrix)
    print("\nClassification Report:")
    print(calibrated_report)
    print("\nMetrics:")

    for key, value in calibrated_metrics.items():
        print(f"{key}: {value}")

    print("\nSession-level results:")
    print(session_report.to_string(index=False))

    print("\nFeature means:")
    print(feature_comparison.to_string(index=False))

    print("\n" + "=" * 72)
    print("V2 live-log evaluation completed successfully.")
    print("=" * 72)
    print("\nResults saved to:")
    print(RESULTS_DIRECTORY)


if __name__ == "__main__":
    main()