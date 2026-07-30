"""
Feature-alignment analysis for DriverGuardianAI.

This script compares live features recorded by the LoggingAgent
against the training features in dataset_exp3.csv.

It identifies feature-distribution mismatch between:

- the original dataset-generation pipeline;
- the current real-time VisionAgent.

Outputs
-------
results/feature_alignment/
    feature_alignment_summary.csv
    condition_distribution.csv
    feature_alignment_plots/
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


DATASET_PATH = Path(
    "data/dataset_exp3.csv"
)

LOG_DIRECTORY = Path(
    "logs"
)

OUTPUT_DIRECTORY = Path(
    "results/feature_alignment"
)

PLOT_DIRECTORY = (
    OUTPUT_DIRECTORY
    / "feature_alignment_plots"
)

NUMERIC_FEATURES = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count"
]

CATEGORICAL_FEATURES = [
    "condition"
]


def find_latest_log(
    log_directory: Path
) -> Path:
    """
    Find the most recently modified session log.
    """

    log_files = list(
        log_directory.glob(
            "driver_guardian_session_*.csv"
        )
    )

    if not log_files:

        raise FileNotFoundError(
            "No DriverGuardianAI session logs were found "
            f"inside: {log_directory}"
        )

    return max(
        log_files,
        key=lambda path: path.stat().st_mtime
    )


def load_dataframes():
    """
    Load the training dataset and latest live session.
    """

    if not DATASET_PATH.exists():

        raise FileNotFoundError(
            f"Dataset was not found: {DATASET_PATH}"
        )

    latest_log = find_latest_log(
        LOG_DIRECTORY
    )

    training_data = pd.read_csv(
        DATASET_PATH
    )

    live_data = pd.read_csv(
        latest_log
    )

    print(
        f"Training dataset: {DATASET_PATH}"
    )

    print(
        f"Training samples: {len(training_data)}"
    )

    print(
        f"\nLatest live log: {latest_log}"
    )

    print(
        f"Live samples: {len(live_data)}"
    )

    return (
        training_data,
        live_data,
        latest_log
    )


def ensure_required_columns(
    training_data: pd.DataFrame,
    live_data: pd.DataFrame
):
    """
    Validate that both sources contain the expected features.
    """

    required_columns = (
        NUMERIC_FEATURES
        + CATEGORICAL_FEATURES
    )

    missing_training = [
        column
        for column in required_columns
        if column not in training_data.columns
    ]

    missing_live = [
        column
        for column in required_columns
        if column not in live_data.columns
    ]

    if missing_training:

        raise ValueError(
            "Training dataset is missing columns: "
            f"{missing_training}"
        )

    if missing_live:

        raise ValueError(
            "Live log is missing columns: "
            f"{missing_live}"
        )


def convert_numeric_columns(
    dataframe: pd.DataFrame
) -> pd.DataFrame:
    """
    Convert comparison features to numeric values.
    """

    dataframe = dataframe.copy()

    for column in NUMERIC_FEATURES:

        if column == "low_light":

            dataframe[column] = (
                dataframe[column]
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

            dataframe[column] = pd.to_numeric(
                dataframe[column],
                errors="coerce"
            )

    return dataframe


def calculate_feature_summary(
    training_data: pd.DataFrame,
    live_data: pd.DataFrame
) -> pd.DataFrame:
    """
    Compare feature statistics and estimate distribution drift.
    """

    rows = []

    for feature in NUMERIC_FEATURES:

        training_values = (
            training_data[feature]
            .dropna()
            .astype(float)
        )

        live_values = (
            live_data[feature]
            .dropna()
            .astype(float)
        )

        if training_values.empty:

            continue

        training_mean = float(
            training_values.mean()
        )

        training_std = float(
            training_values.std()
        )

        training_min = float(
            training_values.min()
        )

        training_max = float(
            training_values.max()
        )

        training_p05 = float(
            training_values.quantile(0.05)
        )

        training_median = float(
            training_values.median()
        )

        training_p95 = float(
            training_values.quantile(0.95)
        )

        if live_values.empty:

            live_mean = np.nan
            live_std = np.nan
            live_min = np.nan
            live_max = np.nan
            live_median = np.nan
            outside_training_range = np.nan
            outside_central_range = np.nan
            standardized_mean_difference = np.nan

        else:

            live_mean = float(
                live_values.mean()
            )

            live_std = float(
                live_values.std()
            )

            live_min = float(
                live_values.min()
            )

            live_max = float(
                live_values.max()
            )

            live_median = float(
                live_values.median()
            )

            outside_training_range = float(
                (
                    (
                        live_values
                        < training_min
                    )
                    |
                    (
                        live_values
                        > training_max
                    )
                ).mean()
            )

            outside_central_range = float(
                (
                    (
                        live_values
                        < training_p05
                    )
                    |
                    (
                        live_values
                        > training_p95
                    )
                ).mean()
            )

            if (
                np.isfinite(training_std)
                and training_std > 1e-12
            ):

                standardized_mean_difference = (
                    live_mean - training_mean
                ) / training_std

            else:

                standardized_mean_difference = np.nan

        rows.append(
            {
                "feature": feature,
                "training_count": len(
                    training_values
                ),
                "live_count": len(
                    live_values
                ),
                "training_mean": training_mean,
                "training_std": training_std,
                "training_min": training_min,
                "training_p05": training_p05,
                "training_median": training_median,
                "training_p95": training_p95,
                "training_max": training_max,
                "live_mean": live_mean,
                "live_std": live_std,
                "live_min": live_min,
                "live_median": live_median,
                "live_max": live_max,
                "live_outside_full_range": (
                    outside_training_range
                ),
                "live_outside_5_95_range": (
                    outside_central_range
                ),
                "standardized_mean_difference": (
                    standardized_mean_difference
                )
            }
        )

    return pd.DataFrame(
        rows
    )


def save_condition_distribution(
    training_data: pd.DataFrame,
    live_data: pd.DataFrame
):
    """
    Save training and live condition frequencies.
    """

    training_conditions = (
        training_data["condition"]
        .astype(str)
        .value_counts()
        .rename("training_count")
    )

    live_conditions = (
        live_data["condition"]
        .astype(str)
        .value_counts()
        .rename("live_count")
    )

    condition_comparison = pd.concat(
        [
            training_conditions,
            live_conditions
        ],
        axis=1
    ).fillna(0)

    condition_comparison[
        "training_percentage"
    ] = (
        condition_comparison["training_count"]
        / condition_comparison[
            "training_count"
        ].sum()
    )

    if (
        condition_comparison[
            "live_count"
        ].sum()
        > 0
    ):

        condition_comparison[
            "live_percentage"
        ] = (
            condition_comparison["live_count"]
            / condition_comparison[
                "live_count"
            ].sum()
        )

    else:

        condition_comparison[
            "live_percentage"
        ] = 0.0

    condition_comparison.to_csv(
        OUTPUT_DIRECTORY
        / "condition_distribution.csv"
    )

    print(
        "\nCondition comparison:"
    )

    print(
        condition_comparison
    )


def plot_feature_distributions(
    training_data: pd.DataFrame,
    live_data: pd.DataFrame
):
    """
    Save one histogram per feature.
    """

    PLOT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    for feature in NUMERIC_FEATURES:

        training_values = (
            training_data[feature]
            .dropna()
            .astype(float)
        )

        live_values = (
            live_data[feature]
            .dropna()
            .astype(float)
        )

        if training_values.empty:

            continue

        plt.figure(
            figsize=(9, 5)
        )

        plt.hist(
            training_values,
            bins=40,
            density=True,
            alpha=0.60,
            label="Training dataset"
        )

        if not live_values.empty:

            plt.hist(
                live_values,
                bins=25,
                density=True,
                alpha=0.60,
                label="Live Vision Agent"
            )

        plt.xlabel(
            feature
        )

        plt.ylabel(
            "Density"
        )

        plt.title(
            f"Training vs Live: {feature}"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            PLOT_DIRECTORY
            / f"{feature}_alignment.png",
            dpi=300
        )

        plt.close()


def print_drift_warnings(
    summary: pd.DataFrame
):
    """
    Print features showing possible distribution mismatch.
    """

    print(
        "\nPotential feature-alignment warnings:"
    )

    warnings_found = False

    for _, row in summary.iterrows():

        feature = row["feature"]

        central_range_outside = row[
            "live_outside_5_95_range"
        ]

        standardized_difference = row[
            "standardized_mean_difference"
        ]

        warning_reasons = []

        if (
            pd.notna(central_range_outside)
            and central_range_outside > 0.25
        ):

            warning_reasons.append(
                f"{central_range_outside:.1%} of live "
                "values fall outside the training "
                "5th–95th percentile range"
            )

        if (
            pd.notna(standardized_difference)
            and abs(
                standardized_difference
            ) > 1.0
        ):

            warning_reasons.append(
                "live mean differs from training mean by "
                f"{standardized_difference:.2f} standard "
                "deviations"
            )

        if warning_reasons:

            warnings_found = True

            print(
                f"\n- {feature}:"
            )

            for reason in warning_reasons:

                print(
                    f"    {reason}"
                )

    if not warnings_found:

        print(
            "No major warnings detected in this session."
        )


def main():
    """
    Run complete feature-alignment analysis.
    """

    OUTPUT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    (
        training_data,
        live_data,
        latest_log
    ) = load_dataframes()

    ensure_required_columns(
        training_data,
        live_data
    )

    training_data = convert_numeric_columns(
        training_data
    )

    live_data = convert_numeric_columns(
        live_data
    )

    summary = calculate_feature_summary(
        training_data,
        live_data
    )

    summary_path = (
        OUTPUT_DIRECTORY
        / "feature_alignment_summary.csv"
    )

    summary.to_csv(
        summary_path,
        index=False
    )

    save_condition_distribution(
        training_data,
        live_data
    )

    plot_feature_distributions(
        training_data,
        live_data
    )

    print(
        "\nFeature-alignment summary:"
    )

    display_columns = [
        "feature",
        "training_mean",
        "live_mean",
        "training_p05",
        "training_p95",
        "live_outside_5_95_range",
        "standardized_mean_difference"
    ]

    print(
        summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print_drift_warnings(
        summary
    )

    print(
        "\nAnalysis complete."
    )

    print(
        f"Analysed live log: {latest_log}"
    )

    print(
        f"Summary saved to: {summary_path}"
    )

    print(
        f"Plots saved to: {PLOT_DIRECTORY}"
    )


if __name__ == "__main__":

    main()