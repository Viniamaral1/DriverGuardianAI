"""
Dataset audit for DriverGuardianAI.

This script analyses the existing dataset before any further
retraining decisions are made.

It checks:
- class balance;
- missing values;
- duplicate rows;
- feature types and ranges;
- feature statistics by class;
- class separation;
- feature overlap;
- correlations;
- possible outliers;
- whether binary classification may be easier.

Run from the project root:

    python diagnostics/dataset_audit.py

Or from Jupyter:

    %run diagnostics/dataset_audit.py
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "dataset_exp3.csv"
)

OUTPUT_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "dataset_audit"
)

PLOT_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "plots"
)


# ============================================================
# DATASET CONFIGURATION
# ============================================================

TARGET_COLUMN = "fatigue_level"

DROP_COLUMNS = [
    "timestamp",
    "source_file",
    "state",
    "fatigue_score"
]

EXPECTED_FEATURES = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count"
]

NUMERIC_FEATURES = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count"
]

CLASS_ORDER = [
    "Alert",
    "Mild Fatigue",
    "Moderate Fatigue"
]


# ============================================================
# LOAD AND CLEAN
# ============================================================

def load_dataset():
    """
    Load and clean the original DriverGuardianAI dataset.
    """

    if not DATASET_PATH.exists():
        raise FileNotFoundError(
            f"Dataset was not found: {DATASET_PATH}"
        )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Original dataset shape: {dataframe.shape}"
    )

    dataframe = dataframe.copy()

    dataframe[TARGET_COLUMN] = (
        dataframe[TARGET_COLUMN]
        .replace(
            {
                "Severe Fatigue": "Moderate Fatigue"
            }
        )
    )

    dataframe.drop(
        columns=DROP_COLUMNS,
        errors="ignore",
        inplace=True
    )

    missing_features = [
        feature
        for feature in EXPECTED_FEATURES
        if feature not in dataframe.columns
    ]

    if missing_features:
        raise ValueError(
            "Dataset is missing expected features: "
            f"{missing_features}"
        )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"Dataset does not contain {TARGET_COLUMN}."
        )

    return dataframe


# ============================================================
# CONVERT TYPES
# ============================================================

def prepare_feature_types(
    dataframe
):
    """
    Convert booleans and numeric columns safely.
    """

    dataframe = dataframe.copy()

    for feature in NUMERIC_FEATURES:

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

    return dataframe


# ============================================================
# GENERAL AUDIT
# ============================================================

def create_general_summary(
    dataframe
):
    """
    Create general dataset quality statistics.
    """

    duplicate_rows = int(
        dataframe.duplicated().sum()
    )

    missing_values = (
        dataframe
        .isna()
        .sum()
        .to_dict()
    )

    class_counts = (
        dataframe[TARGET_COLUMN]
        .value_counts()
        .to_dict()
    )

    class_percentages = (
        dataframe[TARGET_COLUMN]
        .value_counts(
            normalize=True
        )
        .to_dict()
    )

    condition_counts = (
        dataframe["condition"]
        .astype(str)
        .value_counts()
        .to_dict()
    )

    summary = {
        "samples": int(
            len(dataframe)
        ),
        "columns": int(
            len(dataframe.columns)
        ),
        "duplicate_rows": duplicate_rows,
        "missing_values": {
            str(key): int(value)
            for key, value
            in missing_values.items()
        },
        "class_counts": {
            str(key): int(value)
            for key, value
            in class_counts.items()
        },
        "class_percentages": {
            str(key): float(value)
            for key, value
            in class_percentages.items()
        },
        "condition_counts": {
            str(key): int(value)
            for key, value
            in condition_counts.items()
        }
    }

    return summary


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def create_class_distribution(
    dataframe
):
    """
    Save class counts and percentages.
    """

    counts = (
        dataframe[TARGET_COLUMN]
        .value_counts()
        .reindex(
            CLASS_ORDER,
            fill_value=0
        )
    )

    output = pd.DataFrame(
        {
            "class": counts.index,
            "samples": counts.values
        }
    )

    output["percentage"] = (
        output["samples"]
        / output["samples"].sum()
    )

    output.to_csv(
        OUTPUT_DIRECTORY
        / "class_distribution.csv",
        index=False
    )

    plt.figure(
        figsize=(8, 5)
    )

    plt.bar(
        output["class"],
        output["samples"]
    )

    plt.xlabel(
        "Fatigue class"
    )

    plt.ylabel(
        "Samples"
    )

    plt.title(
        "DriverGuardianAI Class Distribution"
    )

    plt.xticks(
        rotation=15,
        ha="right"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIRECTORY
        / "class_distribution.png",
        dpi=300
    )

    plt.close()

    return output


# ============================================================
# FEATURE STATISTICS
# ============================================================

def create_feature_summary(
    dataframe
):
    """
    Calculate feature statistics for each class.
    """

    rows = []

    for class_name in CLASS_ORDER:

        class_data = dataframe[
            dataframe[TARGET_COLUMN]
            == class_name
        ]

        for feature in NUMERIC_FEATURES:

            values = (
                class_data[feature]
                .dropna()
                .astype(float)
            )

            if values.empty:
                continue

            rows.append(
                {
                    "class": class_name,
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
                        values.quantile(
                            0.05
                        )
                    ),
                    "p25": float(
                        values.quantile(
                            0.25
                        )
                    ),
                    "median": float(
                        values.median()
                    ),
                    "p75": float(
                        values.quantile(
                            0.75
                        )
                    ),
                    "p95": float(
                        values.quantile(
                            0.95
                        )
                    ),
                    "maximum": float(
                        values.max()
                    )
                }
            )

    summary = pd.DataFrame(
        rows
    )

    summary.to_csv(
        OUTPUT_DIRECTORY
        / "feature_summary_by_class.csv",
        index=False
    )

    return summary


# ============================================================
# EFFECT SIZE
# ============================================================

def calculate_cohens_d(
    values_a,
    values_b
):
    """
    Calculate Cohen's d between two feature distributions.
    """

    values_a = np.asarray(
        values_a,
        dtype=float
    )

    values_b = np.asarray(
        values_b,
        dtype=float
    )

    values_a = values_a[
        np.isfinite(values_a)
    ]

    values_b = values_b[
        np.isfinite(values_b)
    ]

    if (
        len(values_a) < 2
        or len(values_b) < 2
    ):
        return np.nan

    variance_a = np.var(
        values_a,
        ddof=1
    )

    variance_b = np.var(
        values_b,
        ddof=1
    )

    pooled_variance = (
        (
            (len(values_a) - 1)
            * variance_a
        )
        +
        (
            (len(values_b) - 1)
            * variance_b
        )
    ) / (
        len(values_a)
        + len(values_b)
        - 2
    )

    if pooled_variance <= 1e-12:
        return 0.0

    pooled_standard_deviation = np.sqrt(
        pooled_variance
    )

    return float(
        (
            np.mean(values_a)
            - np.mean(values_b)
        )
        / pooled_standard_deviation
    )


def create_effect_size_report(
    dataframe
):
    """
    Measure how strongly each feature separates class pairs.
    """

    class_pairs = [
        (
            "Alert",
            "Mild Fatigue"
        ),
        (
            "Alert",
            "Moderate Fatigue"
        ),
        (
            "Mild Fatigue",
            "Moderate Fatigue"
        )
    ]

    rows = []

    for feature in NUMERIC_FEATURES:

        for class_a, class_b in class_pairs:

            values_a = dataframe[
                dataframe[TARGET_COLUMN]
                == class_a
            ][feature].dropna()

            values_b = dataframe[
                dataframe[TARGET_COLUMN]
                == class_b
            ][feature].dropna()

            effect_size = calculate_cohens_d(
                values_a,
                values_b
            )

            absolute_effect_size = (
                abs(effect_size)
                if np.isfinite(effect_size)
                else np.nan
            )

            if not np.isfinite(
                absolute_effect_size
            ):
                interpretation = "unknown"

            elif absolute_effect_size < 0.20:
                interpretation = "negligible"

            elif absolute_effect_size < 0.50:
                interpretation = "small"

            elif absolute_effect_size < 0.80:
                interpretation = "medium"

            else:
                interpretation = "large"

            rows.append(
                {
                    "feature": feature,
                    "class_a": class_a,
                    "class_b": class_b,
                    "cohens_d": effect_size,
                    "absolute_effect_size": (
                        absolute_effect_size
                    ),
                    "interpretation": interpretation
                }
            )

    report = pd.DataFrame(
        rows
    )

    report = report.sort_values(
        "absolute_effect_size",
        ascending=False
    )

    report.to_csv(
        OUTPUT_DIRECTORY
        / "feature_effect_sizes.csv",
        index=False
    )

    return report


# ============================================================
# DISTRIBUTION OVERLAP
# ============================================================

def calculate_histogram_overlap(
    values_a,
    values_b,
    bins=50
):
    """
    Estimate overlap between two distributions.

    Returns:
        0 = no overlap
        1 = complete overlap
    """

    values_a = np.asarray(
        values_a,
        dtype=float
    )

    values_b = np.asarray(
        values_b,
        dtype=float
    )

    values_a = values_a[
        np.isfinite(values_a)
    ]

    values_b = values_b[
        np.isfinite(values_b)
    ]

    if (
        len(values_a) == 0
        or len(values_b) == 0
    ):
        return np.nan

    minimum = min(
        values_a.min(),
        values_b.min()
    )

    maximum = max(
        values_a.max(),
        values_b.max()
    )

    if minimum == maximum:
        return 1.0

    bin_edges = np.linspace(
        minimum,
        maximum,
        bins + 1
    )

    histogram_a, _ = np.histogram(
        values_a,
        bins=bin_edges,
        density=True
    )

    histogram_b, _ = np.histogram(
        values_b,
        bins=bin_edges,
        density=True
    )

    bin_width = (
        bin_edges[1]
        - bin_edges[0]
    )

    overlap = np.sum(
        np.minimum(
            histogram_a,
            histogram_b
        )
    ) * bin_width

    return float(
        np.clip(
            overlap,
            0.0,
            1.0
        )
    )


def create_overlap_report(
    dataframe
):
    """
    Measure distribution overlap between classes.
    """

    class_pairs = [
        (
            "Alert",
            "Mild Fatigue"
        ),
        (
            "Alert",
            "Moderate Fatigue"
        ),
        (
            "Mild Fatigue",
            "Moderate Fatigue"
        )
    ]

    rows = []

    for feature in NUMERIC_FEATURES:

        for class_a, class_b in class_pairs:

            values_a = dataframe[
                dataframe[TARGET_COLUMN]
                == class_a
            ][feature].dropna()

            values_b = dataframe[
                dataframe[TARGET_COLUMN]
                == class_b
            ][feature].dropna()

            overlap = (
                calculate_histogram_overlap(
                    values_a,
                    values_b
                )
            )

            rows.append(
                {
                    "feature": feature,
                    "class_a": class_a,
                    "class_b": class_b,
                    "distribution_overlap": (
                        overlap
                    )
                }
            )

    report = pd.DataFrame(
        rows
    )

    report = report.sort_values(
        "distribution_overlap",
        ascending=True
    )

    report.to_csv(
        OUTPUT_DIRECTORY
        / "feature_distribution_overlap.csv",
        index=False
    )

    return report


# ============================================================
# CORRELATION
# ============================================================

def create_correlation_report(
    dataframe
):
    """
    Calculate feature correlation matrix.
    """

    correlation = (
        dataframe[
            NUMERIC_FEATURES
        ]
        .corr()
    )

    correlation.to_csv(
        OUTPUT_DIRECTORY
        / "feature_correlations.csv"
    )

    plt.figure(
        figsize=(9, 7)
    )

    image = plt.imshow(
        correlation,
        vmin=-1.0,
        vmax=1.0
    )

    plt.colorbar(
        image
    )

    plt.xticks(
        range(
            len(NUMERIC_FEATURES)
        ),
        NUMERIC_FEATURES,
        rotation=45,
        ha="right"
    )

    plt.yticks(
        range(
            len(NUMERIC_FEATURES)
        ),
        NUMERIC_FEATURES
    )

    for row_index in range(
        len(NUMERIC_FEATURES)
    ):

        for column_index in range(
            len(NUMERIC_FEATURES)
        ):

            plt.text(
                column_index,
                row_index,
                (
                    f"{correlation.iloc[row_index, column_index]:.2f}"
                ),
                ha="center",
                va="center",
                fontsize=8
            )

    plt.title(
        "DriverGuardianAI Feature Correlations"
    )

    plt.tight_layout()

    plt.savefig(
        PLOT_DIRECTORY
        / "feature_correlations.png",
        dpi=300
    )

    plt.close()

    return correlation


# ============================================================
# OUTLIER REPORT
# ============================================================

def create_outlier_report(
    dataframe
):
    """
    Estimate outliers using the IQR method.
    """

    rows = []

    for class_name in CLASS_ORDER:

        class_data = dataframe[
            dataframe[TARGET_COLUMN]
            == class_name
        ]

        for feature in NUMERIC_FEATURES:

            values = (
                class_data[feature]
                .dropna()
                .astype(float)
            )

            if values.empty:
                continue

            first_quartile = values.quantile(
                0.25
            )

            third_quartile = values.quantile(
                0.75
            )

            interquartile_range = (
                third_quartile
                - first_quartile
            )

            lower_bound = (
                first_quartile
                - 1.5 * interquartile_range
            )

            upper_bound = (
                third_quartile
                + 1.5 * interquartile_range
            )

            outlier_mask = (
                (values < lower_bound)
                |
                (values > upper_bound)
            )

            rows.append(
                {
                    "class": class_name,
                    "feature": feature,
                    "samples": int(
                        len(values)
                    ),
                    "outliers": int(
                        outlier_mask.sum()
                    ),
                    "outlier_percentage": float(
                        outlier_mask.mean()
                    ),
                    "lower_bound": float(
                        lower_bound
                    ),
                    "upper_bound": float(
                        upper_bound
                    )
                }
            )

    report = pd.DataFrame(
        rows
    )

    report.to_csv(
        OUTPUT_DIRECTORY
        / "outlier_report.csv",
        index=False
    )

    return report


# ============================================================
# FEATURE PLOTS
# ============================================================

def save_feature_distribution_plots(
    dataframe
):
    """
    Plot every numeric feature by class.
    """

    for feature in NUMERIC_FEATURES:

        plt.figure(
            figsize=(9, 6)
        )

        for class_name in CLASS_ORDER:

            values = dataframe[
                dataframe[TARGET_COLUMN]
                == class_name
            ][feature].dropna()

            plt.hist(
                values,
                bins=40,
                density=True,
                alpha=0.45,
                label=class_name
            )

        plt.xlabel(
            feature
        )

        plt.ylabel(
            "Density"
        )

        plt.title(
            f"{feature} Distribution by Class"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            PLOT_DIRECTORY
            / f"{feature}_distribution.png",
            dpi=300
        )

        plt.close()

        boxplot_values = []

        for class_name in CLASS_ORDER:

            values = dataframe[
                dataframe[TARGET_COLUMN]
                == class_name
            ][feature].dropna()

            boxplot_values.append(
                values.to_numpy()
            )

        plt.figure(
            figsize=(9, 6)
        )

        plt.boxplot(
            boxplot_values,
            tick_labels=CLASS_ORDER,
            showfliers=False
        )

        plt.xlabel(
            "Fatigue class"
        )

        plt.ylabel(
            feature
        )

        plt.title(
            f"{feature} by Fatigue Class"
        )

        plt.xticks(
            rotation=15,
            ha="right"
        )

        plt.tight_layout()

        plt.savefig(
            PLOT_DIRECTORY
            / f"{feature}_boxplot.png",
            dpi=300
        )

        plt.close()


# ============================================================
# BINARY CLASS COMPARISON
# ============================================================

def create_binary_summary(
    dataframe
):
    """
    Compare possible binary label strategies.

    This does not train a binary model.
    It only describes the resulting class balance.
    """

    binary_fatigue = dataframe[
        TARGET_COLUMN
    ].replace(
        {
            "Alert": "Alert",
            "Mild Fatigue": "Fatigue",
            "Moderate Fatigue": "Fatigue"
        }
    )

    binary_counts = (
        binary_fatigue
        .value_counts()
    )

    alert_vs_mild = dataframe[
        dataframe[TARGET_COLUMN]
        .isin(
            [
                "Alert",
                "Mild Fatigue"
            ]
        )
    ][TARGET_COLUMN]

    alert_mild_counts = (
        alert_vs_mild
        .value_counts()
    )

    summary = {
        "alert_vs_all_fatigue": {
            str(key): int(value)
            for key, value
            in binary_counts.items()
        },
        "alert_vs_mild_only": {
            str(key): int(value)
            for key, value
            in alert_mild_counts.items()
        }
    }

    with (
        OUTPUT_DIRECTORY
        / "binary_class_options.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            summary,
            file,
            indent=4
        )

    return summary


# ============================================================
# FINDINGS
# ============================================================

def create_findings(
    class_distribution,
    effect_sizes,
    overlap_report,
    general_summary
):
    """
    Create concise diagnostic findings.
    """

    findings = []

    moderate_row = class_distribution[
        class_distribution["class"]
        == "Moderate Fatigue"
    ]

    if not moderate_row.empty:

        moderate_percentage = float(
            moderate_row[
                "percentage"
            ].iloc[0]
        )

        findings.append(
            "Moderate Fatigue represents "
            f"{moderate_percentage:.1%} of the dataset."
        )

    largest_effects = (
        effect_sizes
        .dropna(
            subset=[
                "absolute_effect_size"
            ]
        )
        .head(10)
    )

    if not largest_effects.empty:

        top_feature = largest_effects.iloc[
            0
        ]

        findings.append(
            "The strongest observed class-separation feature was "
            f"{top_feature['feature']} for "
            f"{top_feature['class_a']} versus "
            f"{top_feature['class_b']} "
            f"(absolute Cohen's d "
            f"{top_feature['absolute_effect_size']:.2f})."
        )

    alert_mild_overlap = overlap_report[
        (
            overlap_report["class_a"]
            == "Alert"
        )
        &
        (
            overlap_report["class_b"]
            == "Mild Fatigue"
        )
    ]

    if not alert_mild_overlap.empty:

        average_overlap = float(
            alert_mild_overlap[
                "distribution_overlap"
            ].mean()
        )

        findings.append(
            "Average feature-distribution overlap between "
            "Alert and Mild Fatigue was "
            f"{average_overlap:.1%}."
        )

        if average_overlap > 0.70:

            findings.append(
                "Alert and Mild Fatigue show substantial feature "
                "overlap, so a three-class model may be difficult "
                "without temporal or richer image features."
            )

    findings.append(
        "Duplicate rows detected: "
        f"{general_summary['duplicate_rows']}."
    )

    return findings


# ============================================================
# MAIN
# ============================================================

def main():
    """
    Run the complete dataset audit.
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
    print("DriverGuardianAI Dataset Audit")
    print("=" * 72)

    dataframe = load_dataset()

    dataframe = prepare_feature_types(
        dataframe
    )

    print(
        f"Cleaned dataset shape: {dataframe.shape}"
    )

    general_summary = create_general_summary(
        dataframe
    )

    class_distribution = create_class_distribution(
        dataframe
    )

    feature_summary = create_feature_summary(
        dataframe
    )

    effect_sizes = create_effect_size_report(
        dataframe
    )

    overlap_report = create_overlap_report(
        dataframe
    )

    correlation = create_correlation_report(
        dataframe
    )

    outlier_report = create_outlier_report(
        dataframe
    )

    save_feature_distribution_plots(
        dataframe
    )

    binary_summary = create_binary_summary(
        dataframe
    )

    findings = create_findings(
        class_distribution,
        effect_sizes,
        overlap_report,
        general_summary
    )

    with (
        OUTPUT_DIRECTORY
        / "general_summary.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            general_summary,
            file,
            indent=4
        )

    with (
        OUTPUT_DIRECTORY
        / "audit_findings.json"
    ).open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            {
                "findings": findings
            },
            file,
            indent=4
        )

    print("\nClass distribution:")

    print(
        class_distribution.to_string(
            index=False
        )
    )

    print("\nLargest feature effects:")

    print(
        effect_sizes[
            [
                "feature",
                "class_a",
                "class_b",
                "cohens_d",
                "absolute_effect_size",
                "interpretation"
            ]
        ]
        .head(15)
        .to_string(
            index=False
        )
    )

    print("\nLowest distribution overlap:")

    print(
        overlap_report
        .head(15)
        .to_string(
            index=False
        )
    )

    print("\nMain findings:")

    for finding in findings:

        print(
            f"- {finding}"
        )

    print("\nBinary class options:")

    print(
        json.dumps(
            binary_summary,
            indent=4
        )
    )

    print("\n" + "=" * 72)

    print(
        "Dataset audit completed successfully."
    )

    print(
        f"Results saved to: {OUTPUT_DIRECTORY}"
    )

    print("=" * 72)


if __name__ == "__main__":
    main()