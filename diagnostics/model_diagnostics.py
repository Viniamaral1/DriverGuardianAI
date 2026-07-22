"""
Live model diagnostics for DriverGuardianAI.

This script evaluates three labelled real-time sessions:

- logs/alert.csv
- logs/mild.csv
- logs/moderate.csv

It generates:
- Live confusion matrix
- Accuracy, precision, recall and F1
- Per-session prediction distributions
- Confidence analysis
- False-warning analysis
- Critical-alert analysis
- Feature summaries
- Feature-distribution plots
- Prediction timelines
- A machine-readable diagnostics summary

Run from the project root:

    python diagnostics/model_diagnostics.py

Or from Jupyter:

    %run diagnostics/model_diagnostics.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

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


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

LOG_DIRECTORY = PROJECT_ROOT / "logs"

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "live_model_diagnostics"
)

PLOT_DIRECTORY = OUTPUT_DIRECTORY / "plots"


# ============================================================
# SESSION CONFIGURATION
# ============================================================

SESSION_FILES = {
    "Alert": LOG_DIRECTORY / "alert.csv",
    "Mild Fatigue": LOG_DIRECTORY / "mild.csv",
    "Moderate/Severe Fatigue": (
        LOG_DIRECTORY / "moderate.csv"
    )
}

CLASS_NAMES = [
    "Alert",
    "Mild Fatigue",
    "Moderate/Severe Fatigue"
]

CLASS_TO_ID = {
    "Alert": 0,
    "Mild Fatigue": 1,
    "Moderate/Severe Fatigue": 2
}

ID_TO_CLASS = {
    value: key
    for key, value in CLASS_TO_ID.items()
}

FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count"
]

REQUIRED_COLUMNS = [
    "timestamp",
    "prediction",
    "predicted_class",
    "confidence",
    "temporal_state",
    "alert_level",
    "trigger_alert",
    "moderate_ratio",
    "mild_or_higher_ratio",
    "average_confidence",
    "consecutive_moderate",
    "history_size",
    "condition",
    *FEATURE_COLUMNS
]


# ============================================================
# DATA LOADING
# ============================================================

def validate_session_file(
    filepath: Path,
    dataframe: pd.DataFrame
) -> None:
    """
    Validate that a live-session CSV contains all required columns.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{filepath} is missing required columns: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(
            f"{filepath} contains no rows."
        )


def load_live_sessions() -> pd.DataFrame:
    """
    Load and combine all three labelled live sessions.
    """

    session_frames = []

    for actual_label, filepath in SESSION_FILES.items():

        if not filepath.exists():
            raise FileNotFoundError(
                f"Required live-session file was not found: "
                f"{filepath}"
            )

        dataframe = pd.read_csv(
            filepath
        )

        validate_session_file(
            filepath,
            dataframe
        )

        dataframe = dataframe.copy()

        dataframe["actual_label"] = actual_label
        dataframe["actual_class"] = CLASS_TO_ID[
            actual_label
        ]

        dataframe["session_name"] = (
            actual_label
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
        )

        dataframe["timestamp"] = pd.to_datetime(
            dataframe["timestamp"],
            errors="coerce"
        )

        dataframe = dataframe.sort_values(
            "timestamp"
        ).reset_index(
            drop=True
        )

        if dataframe["timestamp"].isna().any():
            raise ValueError(
                f"{filepath} contains invalid timestamps."
            )

        session_start = dataframe[
            "timestamp"
        ].iloc[0]

        dataframe["elapsed_seconds"] = (
            dataframe["timestamp"] - session_start
        ).dt.total_seconds()

        for feature in FEATURE_COLUMNS:

            if feature == "low_light":
                dataframe[feature] = (
                    dataframe[feature]
                    .astype(str)
                    .str.lower()
                    .map(
                        {
                            "true": 1.0,
                            "false": 0.0,
                            "1": 1.0,
                            "0": 0.0,
                            "1.0": 1.0,
                            "0.0": 0.0
                        }
                    )
                )

            else:
                dataframe[feature] = pd.to_numeric(
                    dataframe[feature],
                    errors="coerce"
                )

        numeric_columns = [
            "predicted_class",
            "confidence",
            "moderate_ratio",
            "mild_or_higher_ratio",
            "average_confidence",
            "consecutive_moderate",
            "history_size",
            *FEATURE_COLUMNS
        ]

        for column in numeric_columns:
            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

        dataframe["trigger_alert"] = (
            dataframe["trigger_alert"]
            .astype(str)
            .str.lower()
            .map(
                {
                    "true": True,
                    "false": False,
                    "1": True,
                    "0": False
                }
            )
            .fillna(False)
        )

        session_frames.append(
            dataframe
        )

        print(
            f"Loaded {actual_label}: "
            f"{len(dataframe)} samples"
        )

    combined = pd.concat(
        session_frames,
        ignore_index=True
    )

    return combined


# ============================================================
# METRICS
# ============================================================

def calculate_overall_metrics(
    dataframe: pd.DataFrame
) -> Dict:
    """
    Calculate overall live-session classification metrics.
    """

    y_true = dataframe[
        "actual_class"
    ].astype(int)

    y_pred = dataframe[
        "predicted_class"
    ].astype(int)

    metrics = {
        "samples": int(
            len(dataframe)
        ),
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
        "precision_macro": float(
            precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0
            )
        ),
        "recall_macro": float(
            recall_score(
                y_true,
                y_pred,
                average="macro",
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

    return metrics


def create_classification_report(
    dataframe: pd.DataFrame
) -> Tuple[str, Dict]:
    """
    Generate text and dictionary classification reports.
    """

    y_true = dataframe[
        "actual_class"
    ].astype(int)

    y_pred = dataframe[
        "predicted_class"
    ].astype(int)

    report_text = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    report_dictionary = classification_report(
        y_true,
        y_pred,
        labels=[0, 1, 2],
        target_names=CLASS_NAMES,
        zero_division=0,
        output_dict=True
    )

    return (
        report_text,
        report_dictionary
    )


# ============================================================
# SESSION SUMMARIES
# ============================================================

def find_first_critical_alert_time(
    session: pd.DataFrame
):
    """
    Return seconds until the first critical alert trigger.
    """

    triggered = session[
        session["trigger_alert"] == True
    ]

    if triggered.empty:
        return None

    return float(
        triggered["elapsed_seconds"].iloc[0]
    )


def create_session_summary(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Summarise predictions and decisions for each live session.
    """

    rows = []

    for actual_label in CLASS_NAMES:

        session = dataframe[
            dataframe["actual_label"]
            == actual_label
        ].copy()

        prediction_counts = (
            session["prediction"]
            .value_counts()
        )

        correct_count = int(
            (
                session["predicted_class"]
                == session["actual_class"]
            ).sum()
        )

        total_samples = int(
            len(session)
        )

        duration_seconds = float(
            session["elapsed_seconds"].max()
        )

        rows.append(
            {
                "actual_session": actual_label,
                "samples": total_samples,
                "duration_seconds": duration_seconds,
                "correct_predictions": correct_count,
                "session_accuracy": (
                    correct_count / total_samples
                ),
                "predicted_alert": int(
                    prediction_counts.get(
                        "Alert",
                        0
                    )
                ),
                "predicted_mild": int(
                    prediction_counts.get(
                        "Mild Fatigue",
                        0
                    )
                ),
                "predicted_moderate": int(
                    prediction_counts.get(
                        "Moderate/Severe Fatigue",
                        0
                    )
                ),
                "average_confidence": float(
                    session["confidence"].mean()
                ),
                "minimum_confidence": float(
                    session["confidence"].min()
                ),
                "maximum_confidence": float(
                    session["confidence"].max()
                ),
                "warning_rows": int(
                    (
                        session["alert_level"]
                        == "warning"
                    ).sum()
                ),
                "critical_rows": int(
                    (
                        session["alert_level"]
                        == "critical"
                    ).sum()
                ),
                "critical_alert_triggers": int(
                    session[
                        "trigger_alert"
                    ].sum()
                ),
                "first_critical_alert_seconds": (
                    find_first_critical_alert_time(
                        session
                    )
                )
            }
        )

    return pd.DataFrame(
        rows
    )


# ============================================================
# FEATURE ANALYSIS
# ============================================================

def create_feature_summary(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Calculate feature statistics for each labelled session.
    """

    rows = []

    for actual_label in CLASS_NAMES:

        session = dataframe[
            dataframe["actual_label"]
            == actual_label
        ]

        for feature in FEATURE_COLUMNS:

            values = session[
                feature
            ].dropna()

            rows.append(
                {
                    "actual_session": actual_label,
                    "feature": feature,
                    "count": int(
                        len(values)
                    ),
                    "mean": float(
                        values.mean()
                    ),
                    "std": float(
                        values.std()
                    ),
                    "minimum": float(
                        values.min()
                    ),
                    "p05": float(
                        values.quantile(0.05)
                    ),
                    "median": float(
                        values.median()
                    ),
                    "p95": float(
                        values.quantile(0.95)
                    ),
                    "maximum": float(
                        values.max()
                    )
                }
            )

    return pd.DataFrame(
        rows
    )


# ============================================================
# PLOTS
# ============================================================

def save_confusion_matrix_plot(
    dataframe: pd.DataFrame
) -> None:
    """
    Save raw and normalised live confusion matrices.
    """

    y_true = dataframe[
        "actual_class"
    ].astype(int)

    y_pred = dataframe[
        "predicted_class"
    ].astype(int)

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[0, 1, 2]
    )

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=CLASS_NAMES,
        columns=CLASS_NAMES
    )

    matrix_dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "live_confusion_matrix.csv"
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        matrix
    )

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

    for row_index in range(
        matrix.shape[0]
    ):

        for column_index in range(
            matrix.shape[1]
        ):

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

    plt.xlabel(
        "Predicted live class"
    )

    plt.ylabel(
        "Actual recorded session"
    )

    plt.title(
        "DriverGuardianAI Live Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIRECTORY
        / "live_confusion_matrix.png",
        dpi=300
    )

    plt.close()

    row_totals = matrix.sum(
        axis=1,
        keepdims=True
    )

    normalised = np.divide(
        matrix,
        row_totals,
        out=np.zeros_like(
            matrix,
            dtype=float
        ),
        where=row_totals != 0
    )

    plt.figure(
        figsize=(8, 6)
    )

    plt.imshow(
        normalised,
        vmin=0.0,
        vmax=1.0
    )

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

    for row_index in range(
        normalised.shape[0]
    ):

        for column_index in range(
            normalised.shape[1]
        ):

            plt.text(
                column_index,
                row_index,
                (
                    f"{normalised[row_index, column_index]:.1%}"
                ),
                ha="center",
                va="center"
            )

    plt.xlabel(
        "Predicted live class"
    )

    plt.ylabel(
        "Actual recorded session"
    )

    plt.title(
        "Normalised Live Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIRECTORY
        / "live_confusion_matrix_normalised.png",
        dpi=300
    )

    plt.close()


def save_prediction_distribution_plot(
    session_summary: pd.DataFrame
) -> None:
    """
    Plot predicted-class distribution for every session.
    """

    plot_data = session_summary.set_index(
        "actual_session"
    )[
        [
            "predicted_alert",
            "predicted_mild",
            "predicted_moderate"
        ]
    ]

    plot_data.columns = CLASS_NAMES

    percentages = plot_data.div(
        plot_data.sum(
            axis=1
        ),
        axis=0
    )

    axis = percentages.plot(
        kind="bar",
        stacked=True,
        figsize=(10, 6)
    )

    axis.set_xlabel(
        "Actual recorded session"
    )

    axis.set_ylabel(
        "Prediction proportion"
    )

    axis.set_title(
        "Live Prediction Distribution by Session"
    )

    axis.legend(
        title="Predicted class",
        bbox_to_anchor=(
            1.02,
            1.0
        ),
        loc="upper left"
    )

    plt.xticks(
        rotation=20,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIRECTORY
        / "prediction_distribution_by_session.png",
        dpi=300
    )

    plt.close()


def save_confidence_plot(
    dataframe: pd.DataFrame
) -> None:
    """
    Plot confidence distributions for all sessions.
    """

    plt.figure(
        figsize=(9, 6)
    )

    for actual_label in CLASS_NAMES:

        session = dataframe[
            dataframe["actual_label"]
            == actual_label
        ]

        plt.hist(
            session["confidence"],
            bins=20,
            density=True,
            alpha=0.50,
            label=actual_label
        )

    plt.xlabel(
        "Model confidence"
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        "Live Confidence Distribution"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PLOT_DIRECTORY
        / "confidence_distribution_by_session.png",
        dpi=300
    )

    plt.close()


def save_prediction_timeline_plots(
    dataframe: pd.DataFrame
) -> None:
    """
    Plot prediction class and confidence over time.
    """

    for actual_label in CLASS_NAMES:

        session = dataframe[
            dataframe["actual_label"]
            == actual_label
        ].copy()

        session = session.sort_values(
            "elapsed_seconds"
        )

        filename_label = (
            actual_label
            .lower()
            .replace("/", "_")
            .replace(" ", "_")
        )

        plt.figure(
            figsize=(11, 5)
        )

        plt.step(
            session["elapsed_seconds"],
            session["predicted_class"],
            where="post",
            label="Predicted class"
        )

        plt.yticks(
            [0, 1, 2],
            CLASS_NAMES
        )

        plt.xlabel(
            "Elapsed seconds"
        )

        plt.ylabel(
            "Predicted state"
        )

        plt.title(
            f"Prediction Timeline — {actual_label} Session"
        )

        plt.tight_layout()

        plt.savefig(
            PLOT_DIRECTORY
            / f"{filename_label}_prediction_timeline.png",
            dpi=300
        )

        plt.close()

        plt.figure(
            figsize=(11, 5)
        )

        plt.plot(
            session["elapsed_seconds"],
            session["confidence"],
            label="Prediction confidence"
        )

        plt.plot(
            session["elapsed_seconds"],
            session["average_confidence"],
            label="Rolling average confidence"
        )

        triggered = session[
            session["trigger_alert"]
            == True
        ]

        if not triggered.empty:

            plt.scatter(
                triggered["elapsed_seconds"],
                triggered["confidence"],
                marker="x",
                s=80,
                label="Critical alert trigger"
            )

        plt.xlabel(
            "Elapsed seconds"
        )

        plt.ylabel(
            "Confidence"
        )

        plt.ylim(
            0.0,
            1.05
        )

        plt.title(
            f"Confidence Timeline — {actual_label} Session"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            PLOT_DIRECTORY
            / f"{filename_label}_confidence_timeline.png",
            dpi=300
        )

        plt.close()


def save_feature_distribution_plots(
    dataframe: pd.DataFrame
) -> None:
    """
    Compare each feature across the three live sessions.
    """

    for feature in FEATURE_COLUMNS:

        plt.figure(
            figsize=(9, 6)
        )

        data_to_plot = []

        labels = []

        for actual_label in CLASS_NAMES:

            values = dataframe[
                dataframe["actual_label"]
                == actual_label
            ][feature].dropna()

            data_to_plot.append(
                values.to_numpy()
            )

            labels.append(
                actual_label
            )

        plt.boxplot(
            data_to_plot,
            labels=labels,
            showfliers=False
        )

        plt.xlabel(
            "Actual recorded session"
        )

        plt.ylabel(
            feature
        )

        plt.title(
            f"Live Feature Comparison — {feature}"
        )

        plt.xticks(
            rotation=20,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            PLOT_DIRECTORY
            / f"{feature}_live_session_comparison.png",
            dpi=300
        )

        plt.close()


# ============================================================
# DIAGNOSTIC INTERPRETATION
# ============================================================

def create_diagnostic_findings(
    session_summary: pd.DataFrame,
    overall_metrics: Dict
) -> List[str]:
    """
    Produce concise findings from the live evaluation.
    """

    findings = []

    findings.append(
        "Overall live accuracy was "
        f"{overall_metrics['accuracy']:.1%}."
    )

    for _, row in session_summary.iterrows():

        actual_session = row[
            "actual_session"
        ]

        findings.append(
            f"{actual_session}: "
            f"{row['session_accuracy']:.1%} of predictions "
            "matched the recorded session."
        )

        if row[
            "critical_alert_triggers"
        ] > 0:

            findings.append(
                f"{actual_session}: "
                f"{int(row['critical_alert_triggers'])} "
                "critical alerts were triggered."
            )

    alert_row = session_summary[
        session_summary["actual_session"]
        == "Alert"
    ].iloc[0]

    alert_false_warning_rate = (
        (
            alert_row["predicted_mild"]
            + alert_row["predicted_moderate"]
        )
        / alert_row["samples"]
    )

    findings.append(
        "The Alert-session false-fatigue prediction rate was "
        f"{alert_false_warning_rate:.1%}."
    )

    moderate_row = session_summary[
        session_summary["actual_session"]
        == "Moderate/Severe Fatigue"
    ].iloc[0]

    moderate_detection_rate = (
        moderate_row["predicted_moderate"]
        / moderate_row["samples"]
    )

    findings.append(
        "The Moderate-session frame-level detection rate was "
        f"{moderate_detection_rate:.1%}."
    )

    if alert_false_warning_rate > 0.25:

        findings.append(
            "The model currently produces too many false fatigue "
            "warnings during the Alert session."
        )

    if moderate_detection_rate >= 0.80:

        findings.append(
            "The model is sensitive to Moderate/Severe fatigue, "
            "but sensitivity is accompanied by poor specificity."
        )

    return findings


# ============================================================
# SAVE OUTPUTS
# ============================================================

def save_json(
    data: Dict,
    filepath: Path
) -> None:
    """
    Save JSON with NumPy-safe primitive values.
    """

    with filepath.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=4
        )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run the complete live model diagnostics workflow.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    PLOT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    print("=" * 72)
    print("DriverGuardianAI Live Model Diagnostics")
    print("=" * 72)

    dataframe = load_live_sessions()

    print(
        f"\nCombined live samples: {len(dataframe)}"
    )

    overall_metrics = calculate_overall_metrics(
        dataframe
    )

    (
        report_text,
        report_dictionary
    ) = create_classification_report(
        dataframe
    )

    session_summary = create_session_summary(
        dataframe
    )

    feature_summary = create_feature_summary(
        dataframe
    )

    findings = create_diagnostic_findings(
        session_summary,
        overall_metrics
    )

    session_summary.to_csv(
        OUTPUT_DIRECTORY
        / "live_session_summary.csv",
        index=False
    )

    feature_summary.to_csv(
        OUTPUT_DIRECTORY
        / "live_feature_summary.csv",
        index=False
    )

    dataframe.to_csv(
        OUTPUT_DIRECTORY
        / "combined_live_validation.csv",
        index=False
    )

    with (
        OUTPUT_DIRECTORY
        / "live_classification_report.txt"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report_text
        )

    save_json(
        overall_metrics,
        OUTPUT_DIRECTORY
        / "live_metrics.json"
    )

    save_json(
        report_dictionary,
        OUTPUT_DIRECTORY
        / "live_classification_report.json"
    )

    save_json(
        {
            "findings": findings
        },
        OUTPUT_DIRECTORY
        / "diagnostic_findings.json"
    )

    save_confusion_matrix_plot(
        dataframe
    )

    save_prediction_distribution_plot(
        session_summary
    )

    save_confidence_plot(
        dataframe
    )

    save_prediction_timeline_plots(
        dataframe
    )

    save_feature_distribution_plots(
        dataframe
    )

    print("\nOverall live metrics:")

    for metric_name, metric_value in (
        overall_metrics.items()
    ):

        print(
            f"{metric_name}: {metric_value}"
        )

    print("\nLive classification report:")

    print(
        report_text
    )

    print("\nSession summary:")

    print(
        session_summary.to_string(
            index=False
        )
    )

    print("\nMain diagnostic findings:")

    for finding in findings:

        print(
            f"- {finding}"
        )

    print("\n" + "=" * 72)

    print(
        "Diagnostics completed successfully."
    )

    print(
        f"Results saved to: {OUTPUT_DIRECTORY}"
    )

    print("=" * 72)


if __name__ == "__main__":

    main()