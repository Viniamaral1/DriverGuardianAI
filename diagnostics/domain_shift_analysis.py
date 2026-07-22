"""
DriverGuardianAI production domain-shift analysis.

Experiment 15
-------------
Compare the original offline dataset against two labelled live sessions:

- logs/hgb_live_alert.csv
- logs/hgb_live_fatigue.csv

The analysis answers:

1. Do live feature distributions match the original training data?
2. Which features have shifted most?
3. Are live Alert and live Fatigue behaviours distinguishable?
4. Which feature mismatches are likely contributing to false alarms?
5. Is the current model suitable as a research prototype or a
   production-ready safety system?

Outputs
-------
results/experiment15_domain_shift/
    overall_summary.json
    engineering_report.txt
    feature_drift_summary.csv
    training_feature_summary.csv
    live_feature_summary.csv
    live_session_comparison.csv
    prediction_summary.csv
    plots/
        <feature>_training_vs_live_alert.png
        <feature>_training_vs_live_fatigue.png
        <feature>_live_alert_vs_fatigue.png
        drift_score_ranking.png
        live_prediction_summary.png

Run from the project root:

    python diagnostics/domain_shift_analysis.py

Or in Jupyter:

    %run diagnostics/domain_shift_analysis.py
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from scipy.spatial.distance import jensenshannon
from scipy.stats import ks_2samp, wasserstein_distance


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRAINING_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_exp3.csv"
)

LIVE_ALERT_PATH = (
    PROJECT_ROOT
    / "logs"
    / "hgb_live_alert.csv"
)

LIVE_FATIGUE_PATH = (
    PROJECT_ROOT
    / "logs"
    / "hgb_live_fatigue.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "experiment15_domain_shift"
)

PLOT_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "plots"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COLUMN = "fatigue_level"

NUMERIC_FEATURES = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count",
]

ALL_FEATURES = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count",
]

TRAINING_ALERT_LABEL = "Alert"

TRAINING_FATIGUE_LABELS = {
    "Mild Fatigue",
    "Moderate Fatigue",
    "Severe Fatigue",
}

HISTOGRAM_BINS = 40

# Drift interpretation thresholds for absolute standardised mean
# difference. These are practical engineering categories rather than
# universal medical or regulatory limits.
SMALL_DRIFT_THRESHOLD = 0.50
MODERATE_DRIFT_THRESHOLD = 1.00
LARGE_DRIFT_THRESHOLD = 2.00


# ============================================================
# DATA LOADING
# ============================================================

def require_file(
    filepath: Path,
) -> None:
    """
    Raise a clear error when an expected file is missing.
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"Required file was not found: {filepath}"
        )


def convert_boolean_series(
    series: pd.Series,
) -> pd.Series:
    """
    Convert common boolean representations into 0.0 and 1.0.
    """

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

    converted = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
        .map(mapping)
    )

    numeric_fallback = pd.to_numeric(
        series,
        errors="coerce",
    )

    return converted.fillna(
        numeric_fallback
    )


def prepare_numeric_features(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert all analysed features into consistent numeric forms.
    """

    dataframe = dataframe.copy()

    for feature in NUMERIC_FEATURES:

        if feature == "low_light":
            dataframe[feature] = (
                convert_boolean_series(
                    dataframe[feature]
                )
            )

        else:
            dataframe[feature] = pd.to_numeric(
                dataframe[feature],
                errors="coerce",
            )

    return dataframe


def validate_columns(
    dataframe: pd.DataFrame,
    required_columns: List[str],
    name: str,
) -> None:
    """
    Validate that a DataFrame contains required columns.
    """

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{name} is missing columns: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(
            f"{name} contains no rows."
        )


def load_training_data() -> pd.DataFrame:
    """
    Load and prepare the original training dataset.
    """

    require_file(
        TRAINING_DATASET_PATH
    )

    dataframe = pd.read_csv(
        TRAINING_DATASET_PATH
    )

    validate_columns(
        dataframe,
        [
            TARGET_COLUMN,
            *ALL_FEATURES,
        ],
        "Training dataset",
    )

    dataframe = prepare_numeric_features(
        dataframe
    )

    dataframe["binary_label"] = np.where(
        dataframe[TARGET_COLUMN]
        == TRAINING_ALERT_LABEL,
        "Alert",
        "Fatigue",
    )

    return dataframe


def load_live_session(
    filepath: Path,
    actual_label: str,
) -> pd.DataFrame:
    """
    Load one live session and attach its known actual label.
    """

    require_file(
        filepath
    )

    dataframe = pd.read_csv(
        filepath
    )

    validate_columns(
        dataframe,
        [
            "prediction",
            "predicted_class",
            "confidence",
            "temporal_state",
            "alert_level",
            "trigger_alert",
            *ALL_FEATURES,
        ],
        f"Live {actual_label} session",
    )

    dataframe = prepare_numeric_features(
        dataframe
    )

    dataframe["actual_live_label"] = (
        actual_label
    )

    dataframe["trigger_alert"] = (
        dataframe["trigger_alert"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map(
            {
                "true": True,
                "false": False,
                "1": True,
                "0": False,
            }
        )
        .fillna(False)
    )

    return dataframe


# ============================================================
# STATISTICAL HELPERS
# ============================================================

def finite_values(
    series: pd.Series,
) -> np.ndarray:
    """
    Return finite numeric values only.
    """

    values = pd.to_numeric(
        series,
        errors="coerce",
    ).to_numpy(
        dtype=float
    )

    return values[
        np.isfinite(values)
    ]


def standardised_mean_difference(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
) -> float:
    """
    Measure the mean shift in reference standard-deviation units.
    """

    if (
        len(reference_values) < 2
        or len(comparison_values) < 1
    ):
        return float("nan")

    reference_std = float(
        np.std(
            reference_values,
            ddof=1,
        )
    )

    mean_difference = float(
        np.mean(comparison_values)
        - np.mean(reference_values)
    )

    if reference_std <= 1e-12:
        if abs(mean_difference) <= 1e-12:
            return 0.0

        return float(
            np.sign(mean_difference)
            * np.inf
        )

    return (
        mean_difference
        / reference_std
    )


def normalised_wasserstein_distance(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
) -> float:
    """
    Calculate Wasserstein distance divided by reference spread.
    """

    if (
        len(reference_values) == 0
        or len(comparison_values) == 0
    ):
        return float("nan")

    distance = float(
        wasserstein_distance(
            reference_values,
            comparison_values,
        )
    )

    reference_std = float(
        np.std(
            reference_values,
            ddof=1,
        )
    )

    if reference_std <= 1e-12:
        return (
            0.0
            if distance <= 1e-12
            else float("inf")
        )

    return (
        distance
        / reference_std
    )


def histogram_jensen_shannon_distance(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    bins: int = HISTOGRAM_BINS,
) -> float:
    """
    Estimate Jensen-Shannon distance between two distributions.
    """

    if (
        len(reference_values) == 0
        or len(comparison_values) == 0
    ):
        return float("nan")

    minimum = min(
        float(np.min(reference_values)),
        float(np.min(comparison_values)),
    )

    maximum = max(
        float(np.max(reference_values)),
        float(np.max(comparison_values)),
    )

    if minimum == maximum:
        return 0.0

    edges = np.linspace(
        minimum,
        maximum,
        bins + 1,
    )

    reference_histogram, _ = np.histogram(
        reference_values,
        bins=edges,
    )

    comparison_histogram, _ = np.histogram(
        comparison_values,
        bins=edges,
    )

    reference_probability = (
        reference_histogram.astype(float)
        + 1e-12
    )

    comparison_probability = (
        comparison_histogram.astype(float)
        + 1e-12
    )

    reference_probability /= (
        reference_probability.sum()
    )

    comparison_probability /= (
        comparison_probability.sum()
    )

    return float(
        jensenshannon(
            reference_probability,
            comparison_probability,
            base=2,
        )
    )


def outside_reference_range_rate(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
) -> Tuple[float, float, float]:
    """
    Return the percentage outside the reference 5th–95th percentile.
    """

    if (
        len(reference_values) == 0
        or len(comparison_values) == 0
    ):
        return (
            float("nan"),
            float("nan"),
            float("nan"),
        )

    lower = float(
        np.quantile(
            reference_values,
            0.05,
        )
    )

    upper = float(
        np.quantile(
            reference_values,
            0.95,
        )
    )

    outside_rate = float(
        np.mean(
            (comparison_values < lower)
            |
            (comparison_values > upper)
        )
    )

    return (
        lower,
        upper,
        outside_rate,
    )


def drift_category(
    absolute_smd: float,
) -> str:
    """
    Convert absolute SMD into a readable drift category.
    """

    if not np.isfinite(
        absolute_smd
    ):
        return "undefined"

    if absolute_smd < SMALL_DRIFT_THRESHOLD:
        return "low"

    if absolute_smd < MODERATE_DRIFT_THRESHOLD:
        return "moderate"

    if absolute_smd < LARGE_DRIFT_THRESHOLD:
        return "high"

    return "very high"


# ============================================================
# FEATURE SUMMARIES
# ============================================================

def summarise_features(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Generate descriptive statistics for each numeric feature.
    """

    rows = []

    for feature in NUMERIC_FEATURES:
        values = finite_values(
            dataframe[feature]
        )

        if len(values) == 0:
            continue

        rows.append(
            {
                "dataset": dataset_name,
                "feature": feature,
                "samples": int(
                    len(values)
                ),
                "mean": float(
                    np.mean(values)
                ),
                "std": float(
                    np.std(
                        values,
                        ddof=1,
                    )
                )
                if len(values) > 1
                else 0.0,
                "minimum": float(
                    np.min(values)
                ),
                "p05": float(
                    np.quantile(
                        values,
                        0.05,
                    )
                ),
                "median": float(
                    np.median(values)
                ),
                "p95": float(
                    np.quantile(
                        values,
                        0.95,
                    )
                ),
                "maximum": float(
                    np.max(values)
                ),
            }
        )

    return pd.DataFrame(
        rows
    )


def analyse_feature_drift(
    reference_dataframe: pd.DataFrame,
    comparison_dataframe: pd.DataFrame,
    reference_name: str,
    comparison_name: str,
) -> pd.DataFrame:
    """
    Calculate multiple drift metrics for every numeric feature.
    """

    rows = []

    for feature in NUMERIC_FEATURES:

        reference_values = finite_values(
            reference_dataframe[feature]
        )

        comparison_values = finite_values(
            comparison_dataframe[feature]
        )

        if (
            len(reference_values) == 0
            or len(comparison_values) == 0
        ):
            continue

        smd = standardised_mean_difference(
            reference_values,
            comparison_values,
        )

        absolute_smd = float(
            abs(smd)
        )

        normalised_wasserstein = (
            normalised_wasserstein_distance(
                reference_values,
                comparison_values,
            )
        )

        js_distance = (
            histogram_jensen_shannon_distance(
                reference_values,
                comparison_values,
            )
        )

        (
            reference_p05,
            reference_p95,
            outside_rate,
        ) = outside_reference_range_rate(
            reference_values,
            comparison_values,
        )

        ks_result = ks_2samp(
            reference_values,
            comparison_values,
            alternative="two-sided",
            method="auto",
        )

        # Composite ranking score. It is an engineering aid, not a
        # calibrated probability or a regulatory metric.
        finite_smd = (
            min(
                absolute_smd,
                5.0,
            )
            if np.isfinite(
                absolute_smd
            )
            else 5.0
        )

        finite_wasserstein = (
            min(
                normalised_wasserstein,
                5.0,
            )
            if np.isfinite(
                normalised_wasserstein
            )
            else 5.0
        )

        composite_score = (
            0.40 * finite_smd
            + 0.25 * finite_wasserstein
            + 0.20 * js_distance
            + 0.15 * outside_rate
        )

        rows.append(
            {
                "reference": reference_name,
                "comparison": comparison_name,
                "feature": feature,
                "reference_samples": int(
                    len(reference_values)
                ),
                "comparison_samples": int(
                    len(comparison_values)
                ),
                "reference_mean": float(
                    np.mean(reference_values)
                ),
                "comparison_mean": float(
                    np.mean(comparison_values)
                ),
                "reference_std": float(
                    np.std(
                        reference_values,
                        ddof=1,
                    )
                )
                if len(reference_values) > 1
                else 0.0,
                "comparison_std": float(
                    np.std(
                        comparison_values,
                        ddof=1,
                    )
                )
                if len(comparison_values) > 1
                else 0.0,
                "standardised_mean_difference": (
                    float(smd)
                ),
                "absolute_smd": absolute_smd,
                "normalised_wasserstein": float(
                    normalised_wasserstein
                ),
                "jensen_shannon_distance": float(
                    js_distance
                ),
                "ks_statistic": float(
                    ks_result.statistic
                ),
                "ks_pvalue": float(
                    ks_result.pvalue
                ),
                "reference_p05": reference_p05,
                "reference_p95": reference_p95,
                "comparison_outside_p05_p95": (
                    outside_rate
                ),
                "composite_drift_score": float(
                    composite_score
                ),
                "drift_category": drift_category(
                    absolute_smd
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "composite_drift_score",
        ascending=False,
    )


# ============================================================
# PREDICTION AND LIVE COMPARISON
# ============================================================

def prediction_summary(
    dataframe: pd.DataFrame,
    actual_label: str,
) -> Dict:
    """
    Summarise predictions and temporal alerts for one live session.
    """

    prediction_counts = (
        dataframe["prediction"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    correct_prediction = (
        dataframe["prediction"]
        .astype(str)
        == actual_label
    )

    return {
        "actual_label": actual_label,
        "samples": int(
            len(dataframe)
        ),
        "correct_predictions": int(
            correct_prediction.sum()
        ),
        "frame_accuracy": float(
            correct_prediction.mean()
        ),
        "prediction_counts": {
            str(key): int(value)
            for key, value
            in prediction_counts.items()
        },
        "average_confidence": float(
            pd.to_numeric(
                dataframe["confidence"],
                errors="coerce",
            ).mean()
        ),
        "critical_rows": int(
            (
                dataframe["alert_level"]
                .astype(str)
                == "critical"
            ).sum()
        ),
        "alert_triggers": int(
            dataframe[
                "trigger_alert"
            ].sum()
        ),
    }


def create_live_session_comparison(
    live_alert: pd.DataFrame,
    live_fatigue: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare feature values between the two live sessions.
    """

    rows = []

    for feature in NUMERIC_FEATURES:

        alert_values = finite_values(
            live_alert[feature]
        )

        fatigue_values = finite_values(
            live_fatigue[feature]
        )

        if (
            len(alert_values) == 0
            or len(fatigue_values) == 0
        ):
            continue

        pooled_variance = (
            (
                (len(alert_values) - 1)
                * np.var(
                    alert_values,
                    ddof=1,
                )
            )
            +
            (
                (len(fatigue_values) - 1)
                * np.var(
                    fatigue_values,
                    ddof=1,
                )
            )
        ) / (
            len(alert_values)
            + len(fatigue_values)
            - 2
        )

        if pooled_variance <= 1e-12:
            cohens_d = 0.0

        else:
            cohens_d = (
                np.mean(fatigue_values)
                - np.mean(alert_values)
            ) / np.sqrt(
                pooled_variance
            )

        rows.append(
            {
                "feature": feature,
                "live_alert_mean": float(
                    np.mean(alert_values)
                ),
                "live_fatigue_mean": float(
                    np.mean(fatigue_values)
                ),
                "fatigue_minus_alert": float(
                    np.mean(fatigue_values)
                    - np.mean(alert_values)
                ),
                "cohens_d_fatigue_vs_alert": float(
                    cohens_d
                ),
                "absolute_effect_size": float(
                    abs(cohens_d)
                ),
                "wasserstein_distance": float(
                    wasserstein_distance(
                        alert_values,
                        fatigue_values,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "absolute_effect_size",
        ascending=False,
    )


# ============================================================
# PLOTS
# ============================================================

def save_distribution_plot(
    reference_values: np.ndarray,
    comparison_values: np.ndarray,
    reference_label: str,
    comparison_label: str,
    feature: str,
    filepath: Path,
) -> None:
    """
    Save an overlaid distribution plot.
    """

    plt.figure(
        figsize=(
            9,
            6,
        )
    )

    plt.hist(
        reference_values,
        bins=HISTOGRAM_BINS,
        density=True,
        alpha=0.50,
        label=reference_label,
    )

    plt.hist(
        comparison_values,
        bins=HISTOGRAM_BINS,
        density=True,
        alpha=0.50,
        label=comparison_label,
    )

    plt.xlabel(
        feature
    )

    plt.ylabel(
        "Density"
    )

    plt.title(
        f"{feature}: {reference_label} vs {comparison_label}"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
    )

    plt.close()


def save_feature_distribution_plots(
    training_alert: pd.DataFrame,
    training_fatigue: pd.DataFrame,
    live_alert: pd.DataFrame,
    live_fatigue: pd.DataFrame,
) -> None:
    """
    Create all feature-distribution comparisons.
    """

    for feature in NUMERIC_FEATURES:

        training_alert_values = finite_values(
            training_alert[feature]
        )

        training_fatigue_values = finite_values(
            training_fatigue[feature]
        )

        live_alert_values = finite_values(
            live_alert[feature]
        )

        live_fatigue_values = finite_values(
            live_fatigue[feature]
        )

        save_distribution_plot(
            training_alert_values,
            live_alert_values,
            "Training Alert",
            "Live Alert",
            feature,
            PLOT_DIRECTORY
            / (
                f"{feature}_"
                "training_vs_live_alert.png"
            ),
        )

        save_distribution_plot(
            training_fatigue_values,
            live_fatigue_values,
            "Training Fatigue",
            "Live Fatigue",
            feature,
            PLOT_DIRECTORY
            / (
                f"{feature}_"
                "training_vs_live_fatigue.png"
            ),
        )

        save_distribution_plot(
            live_alert_values,
            live_fatigue_values,
            "Live Alert",
            "Live Fatigue",
            feature,
            PLOT_DIRECTORY
            / (
                f"{feature}_"
                "live_alert_vs_fatigue.png"
            ),
        )


def save_drift_ranking_plot(
    drift_summary: pd.DataFrame,
) -> None:
    """
    Plot composite drift score by feature and comparison.
    """

    plot_data = (
        drift_summary
        .pivot(
            index="feature",
            columns="comparison",
            values="composite_drift_score",
        )
        .fillna(0.0)
    )

    plot_data["maximum_drift"] = (
        plot_data.max(
            axis=1
        )
    )

    plot_data = plot_data.sort_values(
        "maximum_drift",
        ascending=True,
    ).drop(
        columns=[
            "maximum_drift"
        ]
    )

    axis = plot_data.plot(
        kind="barh",
        figsize=(
            10,
            7,
        ),
    )

    axis.set_xlabel(
        "Composite drift score"
    )

    axis.set_ylabel(
        "Feature"
    )

    axis.set_title(
        "Training-to-Live Feature Drift"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIRECTORY
        / "drift_score_ranking.png",
        dpi=300,
    )

    plt.close()


def save_prediction_summary_plot(
    prediction_rows: pd.DataFrame,
) -> None:
    """
    Plot frame accuracy and critical-row rate.
    """

    plot_data = prediction_rows.copy()

    plot_data[
        "critical_row_rate"
    ] = (
        plot_data["critical_rows"]
        / plot_data["samples"]
    )

    indexed = plot_data.set_index(
        "actual_label"
    )[
        [
            "frame_accuracy",
            "critical_row_rate",
        ]
    ]

    axis = indexed.plot(
        kind="bar",
        figsize=(
            8,
            6,
        ),
    )

    axis.set_ylim(
        0.0,
        1.05,
    )

    axis.set_ylabel(
        "Proportion"
    )

    axis.set_xlabel(
        "Actual live session"
    )

    axis.set_title(
        "Live Prediction and Alert Behaviour"
    )

    plt.xticks(
        rotation=0,
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIRECTORY
        / "live_prediction_summary.png",
        dpi=300,
    )

    plt.close()


# ============================================================
# INTERPRETATION
# ============================================================

def create_readiness_assessment(
    alert_prediction: Dict,
    fatigue_prediction: Dict,
    alert_drift: pd.DataFrame,
    fatigue_drift: pd.DataFrame,
) -> Dict:
    """
    Produce a transparent prototype-readiness assessment.
    """

    alert_accuracy = float(
        alert_prediction[
            "frame_accuracy"
        ]
    )

    fatigue_accuracy = float(
        fatigue_prediction[
            "frame_accuracy"
        ]
    )

    alert_false_positive_rate = (
        1.0
        - alert_accuracy
    )

    maximum_alert_drift = float(
        alert_drift[
            "composite_drift_score"
        ].max()
    )

    maximum_fatigue_drift = float(
        fatigue_drift[
            "composite_drift_score"
        ].max()
    )

    # This score is only a project-management summary. It must not be
    # represented as a validated safety or regulatory score.
    readiness_score = (
        100.0
        * (
            0.35 * alert_accuracy
            + 0.35 * fatigue_accuracy
            + 0.15 * max(
                0.0,
                1.0 - min(
                    maximum_alert_drift / 3.0,
                    1.0,
                ),
            )
            + 0.15 * max(
                0.0,
                1.0 - min(
                    maximum_fatigue_drift / 3.0,
                    1.0,
                ),
            )
        )
    )

    if (
        alert_false_positive_rate > 0.25
        or alert_accuracy < 0.70
    ):
        status = (
            "research prototype — not suitable "
            "for safety-critical deployment"
        )

    elif (
        fatigue_accuracy < 0.70
        or readiness_score < 70.0
    ):
        status = (
            "prototype requiring further external validation"
        )

    else:
        status = (
            "promising prototype requiring broader participant "
            "and environment validation"
        )

    return {
        "engineering_readiness_score": float(
            readiness_score
        ),
        "status": status,
        "alert_session_accuracy": alert_accuracy,
        "fatigue_session_accuracy": fatigue_accuracy,
        "alert_false_positive_rate": float(
            alert_false_positive_rate
        ),
        "maximum_alert_composite_drift": (
            maximum_alert_drift
        ),
        "maximum_fatigue_composite_drift": (
            maximum_fatigue_drift
        ),
        "important_note": (
            "This is an internal engineering summary, not a medical, "
            "automotive-safety, legal, regulatory, or certification score."
        ),
    }


def create_engineering_findings(
    alert_drift: pd.DataFrame,
    fatigue_drift: pd.DataFrame,
    live_comparison: pd.DataFrame,
    alert_prediction: Dict,
    fatigue_prediction: Dict,
    readiness: Dict,
) -> List[str]:
    """
    Generate concise evidence-based engineering findings.
    """

    findings = []

    findings.append(
        "The live Alert session was correctly classified on "
        f"{alert_prediction['frame_accuracy']:.1%} of frames."
    )

    findings.append(
        "The live Fatigue session was correctly classified on "
        f"{fatigue_prediction['frame_accuracy']:.1%} of frames."
    )

    top_alert = alert_drift.iloc[
        0
    ]

    findings.append(
        "The largest Training Alert to Live Alert drift was observed "
        f"for {top_alert['feature']} "
        f"(composite score {top_alert['composite_drift_score']:.3f}, "
        f"absolute SMD {top_alert['absolute_smd']:.2f})."
    )

    top_fatigue = fatigue_drift.iloc[
        0
    ]

    findings.append(
        "The largest Training Fatigue to Live Fatigue drift was "
        f"observed for {top_fatigue['feature']} "
        f"(composite score {top_fatigue['composite_drift_score']:.3f}, "
        f"absolute SMD {top_fatigue['absolute_smd']:.2f})."
    )

    if not live_comparison.empty:

        top_live_separator = (
            live_comparison.iloc[
                0
            ]
        )

        findings.append(
            "The strongest difference between the two live sessions "
            f"was {top_live_separator['feature']} "
            f"(absolute Cohen's d "
            f"{top_live_separator['absolute_effect_size']:.2f})."
        )

    if (
        alert_prediction[
            "frame_accuracy"
        ]
        < 0.50
    ):
        findings.append(
            "The deployed classifier does not currently recognise "
            "the live Alert state reliably."
        )

    if (
        fatigue_prediction[
            "frame_accuracy"
        ]
        >= 0.80
    ):
        findings.append(
            "The classifier is highly sensitive to the simulated "
            "Fatigue session, but this sensitivity is accompanied by "
            "poor Alert specificity."
        )

    findings.append(
        "The current engineering status is: "
        f"{readiness['status']}."
    )

    return findings


def write_engineering_report(
    filepath: Path,
    findings: List[str],
    readiness: Dict,
    alert_drift: pd.DataFrame,
    fatigue_drift: pd.DataFrame,
    live_comparison: pd.DataFrame,
) -> None:
    """
    Save a readable engineering report.
    """

    lines = [
        "DriverGuardianAI",
        "Experiment 15 — Production Domain-Shift Analysis",
        "=" * 72,
        "",
        "Executive summary",
        "-" * 72,
    ]

    for finding in findings:
        lines.append(
            f"- {finding}"
        )

    lines.extend(
        [
            "",
            "Readiness assessment",
            "-" * 72,
            (
                "Internal engineering score: "
                f"{readiness['engineering_readiness_score']:.1f}/100"
            ),
            (
                "Status: "
                f"{readiness['status']}"
            ),
            (
                "Alert false-positive rate: "
                f"{readiness['alert_false_positive_rate']:.1%}"
            ),
            "",
            (
                "Important: this score is an internal engineering "
                "summary only. It is not a medical, legal, automotive "
                "safety, regulatory, or certification score."
            ),
            "",
            "Highest drift: Training Alert vs Live Alert",
            "-" * 72,
            alert_drift[
                [
                    "feature",
                    "reference_mean",
                    "comparison_mean",
                    "absolute_smd",
                    "jensen_shannon_distance",
                    "comparison_outside_p05_p95",
                    "composite_drift_score",
                    "drift_category",
                ]
            ]
            .head(7)
            .to_string(
                index=False
            ),
            "",
            "Highest drift: Training Fatigue vs Live Fatigue",
            "-" * 72,
            fatigue_drift[
                [
                    "feature",
                    "reference_mean",
                    "comparison_mean",
                    "absolute_smd",
                    "jensen_shannon_distance",
                    "comparison_outside_p05_p95",
                    "composite_drift_score",
                    "drift_category",
                ]
            ]
            .head(7)
            .to_string(
                index=False
            ),
            "",
            "Differences between the two live sessions",
            "-" * 72,
            live_comparison[
                [
                    "feature",
                    "live_alert_mean",
                    "live_fatigue_mean",
                    "cohens_d_fatigue_vs_alert",
                    "absolute_effect_size",
                ]
            ]
            .to_string(
                index=False
            ),
            "",
            "Recommended interpretation",
            "-" * 72,
            (
                "The real-time software pipeline is operational, but "
                "external live validation reveals a domain shift between "
                "the original collected recordings and current webcam "
                "inference. The system should be presented as a research "
                "prototype. Future improvement should focus on alignment "
                "of feature extraction, session-aware labels, broader "
                "external validation, and participant-independent data "
                "rather than simply replacing the classifier."
            ),
            "",
        ]
    )

    filepath.write_text(
        "\n".join(
            lines
        ),
        encoding="utf-8",
    )


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run the full domain-shift analysis.
    """

    print("=" * 72)
    print("DriverGuardianAI")
    print("Experiment 15: Production Domain-Shift Analysis")
    print("=" * 72)

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    PLOT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading original training dataset..."
    )

    training = load_training_data()

    print(
        f"Training samples: {len(training)}"
    )

    print(
        "\nLoading labelled live sessions..."
    )

    live_alert = load_live_session(
        LIVE_ALERT_PATH,
        "Alert",
    )

    live_fatigue = load_live_session(
        LIVE_FATIGUE_PATH,
        "Fatigue",
    )

    print(
        f"Live Alert samples: {len(live_alert)}"
    )

    print(
        f"Live Fatigue samples: {len(live_fatigue)}"
    )

    training_alert = training[
        training["binary_label"]
        == "Alert"
    ].copy()

    training_fatigue = training[
        training["binary_label"]
        == "Fatigue"
    ].copy()

    training_summary = pd.concat(
        [
            summarise_features(
                training_alert,
                "Training Alert",
            ),
            summarise_features(
                training_fatigue,
                "Training Fatigue",
            ),
        ],
        ignore_index=True,
    )

    live_summary = pd.concat(
        [
            summarise_features(
                live_alert,
                "Live Alert",
            ),
            summarise_features(
                live_fatigue,
                "Live Fatigue",
            ),
        ],
        ignore_index=True,
    )

    print(
        "\nCalculating feature drift..."
    )

    alert_drift = analyse_feature_drift(
        training_alert,
        live_alert,
        "Training Alert",
        "Live Alert",
    )

    fatigue_drift = analyse_feature_drift(
        training_fatigue,
        live_fatigue,
        "Training Fatigue",
        "Live Fatigue",
    )

    drift_summary = pd.concat(
        [
            alert_drift,
            fatigue_drift,
        ],
        ignore_index=True,
    )

    live_comparison = (
        create_live_session_comparison(
            live_alert,
            live_fatigue,
        )
    )

    alert_prediction = prediction_summary(
        live_alert,
        "Alert",
    )

    fatigue_prediction = prediction_summary(
        live_fatigue,
        "Fatigue",
    )

    prediction_rows = pd.DataFrame(
        [
            {
                key: value
                for key, value
                in alert_prediction.items()
                if key != "prediction_counts"
            },
            {
                key: value
                for key, value
                in fatigue_prediction.items()
                if key != "prediction_counts"
            },
        ]
    )

    readiness = create_readiness_assessment(
        alert_prediction,
        fatigue_prediction,
        alert_drift,
        fatigue_drift,
    )

    findings = create_engineering_findings(
        alert_drift,
        fatigue_drift,
        live_comparison,
        alert_prediction,
        fatigue_prediction,
        readiness,
    )

    training_summary.to_csv(
        OUTPUT_DIRECTORY
        / "training_feature_summary.csv",
        index=False,
    )

    live_summary.to_csv(
        OUTPUT_DIRECTORY
        / "live_feature_summary.csv",
        index=False,
    )

    drift_summary.to_csv(
        OUTPUT_DIRECTORY
        / "feature_drift_summary.csv",
        index=False,
    )

    live_comparison.to_csv(
        OUTPUT_DIRECTORY
        / "live_session_comparison.csv",
        index=False,
    )

    prediction_rows.to_csv(
        OUTPUT_DIRECTORY
        / "prediction_summary.csv",
        index=False,
    )

    save_feature_distribution_plots(
        training_alert,
        training_fatigue,
        live_alert,
        live_fatigue,
    )

    save_drift_ranking_plot(
        drift_summary
    )

    save_prediction_summary_plot(
        prediction_rows
    )

    overall_summary = {
        "experiment": (
            "experiment15_domain_shift"
        ),
        "training_samples": int(
            len(training)
        ),
        "live_alert_samples": int(
            len(live_alert)
        ),
        "live_fatigue_samples": int(
            len(live_fatigue)
        ),
        "prediction_summary": {
            "alert_session": (
                alert_prediction
            ),
            "fatigue_session": (
                fatigue_prediction
            ),
        },
        "readiness_assessment": readiness,
        "findings": findings,
        "top_alert_drift_features": (
            alert_drift
            .head(5)
            .to_dict(
                orient="records"
            )
        ),
        "top_fatigue_drift_features": (
            fatigue_drift
            .head(5)
            .to_dict(
                orient="records"
            )
        ),
        "top_live_separating_features": (
            live_comparison
            .head(5)
            .to_dict(
                orient="records"
            )
        ),
    }

    with (
        OUTPUT_DIRECTORY
        / "overall_summary.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            overall_summary,
            file,
            indent=4,
        )

    write_engineering_report(
        OUTPUT_DIRECTORY
        / "engineering_report.txt",
        findings,
        readiness,
        alert_drift,
        fatigue_drift,
        live_comparison,
    )

    print(
        "\nPrediction summary:"
    )

    print(
        prediction_rows.to_string(
            index=False
        )
    )

    print(
        "\nHighest drift — Training Alert vs Live Alert:"
    )

    print(
        alert_drift[
            [
                "feature",
                "reference_mean",
                "comparison_mean",
                "absolute_smd",
                "jensen_shannon_distance",
                "comparison_outside_p05_p95",
                "composite_drift_score",
                "drift_category",
            ]
        ]
        .head(7)
        .to_string(
            index=False
        )
    )

    print(
        "\nHighest drift — Training Fatigue vs Live Fatigue:"
    )

    print(
        fatigue_drift[
            [
                "feature",
                "reference_mean",
                "comparison_mean",
                "absolute_smd",
                "jensen_shannon_distance",
                "comparison_outside_p05_p95",
                "composite_drift_score",
                "drift_category",
            ]
        ]
        .head(7)
        .to_string(
            index=False
        )
    )

    print(
        "\nStrongest differences between live sessions:"
    )

    print(
        live_comparison[
            [
                "feature",
                "live_alert_mean",
                "live_fatigue_mean",
                "cohens_d_fatigue_vs_alert",
                "absolute_effect_size",
            ]
        ]
        .to_string(
            index=False
        )
    )

    print(
        "\nMain engineering findings:"
    )

    for finding in findings:
        print(
            f"- {finding}"
        )

    print(
        "\nInternal engineering readiness assessment:"
    )

    print(
        "Score: "
        f"{readiness['engineering_readiness_score']:.1f}/100"
    )

    print(
        "Status: "
        f"{readiness['status']}"
    )

    print(
        "\nImportant: the readiness score is an internal engineering "
        "summary only, not a safety or regulatory certification."
    )

    print(
        "\n" + "=" * 72
    )

    print(
        "Experiment 15 completed successfully."
    )

    print(
        f"Results saved to: {OUTPUT_DIRECTORY}"
    )

    print(
        "=" * 72
    )


if __name__ == "__main__":
    main()