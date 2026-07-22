"""
Diagnose feature dependence in the DriverGuardianAI V2 model.

Purpose
-------
The V2 model performs extremely well on the participant-aware test set,
but it predicts Fatigue almost continuously during live testing.

This script measures whether two unstable features are dominating those
live predictions:

- hands_detected
- blink_count

For every live row, it compares the original prediction with several
counterfactual feature versions:

1. Original features
2. Hands always 0
3. Hands always 1
4. Blink count always 0
5. Blink count fixed to the V2 training median
6. Hands 0 and blink 0
7. Hands 1 and blink 0

It does not retrain or modify the model.

Inputs
------
models/v2/driver_guardian_hgb_v2.joblib

By default, the script automatically loads the two newest CSV files from:

logs/v2/

Optional manual paths can be set in MANUAL_LOG_PATHS below.

Outputs
-------
results/v2/feature_dependence/
    feature_dependence_summary.json
    scenario_summary.csv
    session_scenario_summary.csv
    row_level_counterfactuals.csv
    probability_shift_by_scenario.png
    original_probability_by_session.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v2"
    / "driver_guardian_hgb_v2.joblib"
)

LOG_DIRECTORY = (
    PROJECT_ROOT
    / "logs"
    / "v2"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "feature_dependence"
)

SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "feature_dependence_summary.json"
)

SCENARIO_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "scenario_summary.csv"
)

SESSION_SCENARIO_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "session_scenario_summary.csv"
)

ROW_LEVEL_PATH = (
    RESULTS_DIRECTORY
    / "row_level_counterfactuals.csv"
)

PROBABILITY_SHIFT_PLOT_PATH = (
    RESULTS_DIRECTORY
    / "probability_shift_by_scenario.png"
)

ORIGINAL_PROBABILITY_PLOT_PATH = (
    RESULTS_DIRECTORY
    / "original_probability_by_session.png"
)


# ============================================================
# OPTIONAL MANUAL LOG PATHS
# ============================================================

# Leave this empty to automatically use the two newest V2 log files.
#
# Example:
#
# MANUAL_LOG_PATHS = [
#     PROJECT_ROOT / "logs" / "v2"
#     / "driver_guardian_v2_session_20260717_154546.csv",
#
#     PROJECT_ROOT / "logs" / "v2"
#     / "driver_guardian_v2_session_20260717_154904.csv",
# ]

MANUAL_LOG_PATHS: List[Path] = []


# ============================================================
# SETTINGS
# ============================================================

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

TRAINING_BLINK_MEDIAN = 3.0

SCENARIOS = {
    "original": {},
    "hands_always_0": {
        "hands_detected": 0.0,
    },
    "hands_always_1": {
        "hands_detected": 1.0,
    },
    "blink_always_0": {
        "blink_count": 0.0,
    },
    "blink_training_median": {
        "blink_count": TRAINING_BLINK_MEDIAN,
    },
    "hands_0_blink_0": {
        "hands_detected": 0.0,
        "blink_count": 0.0,
    },
    "hands_1_blink_0": {
        "hands_detected": 1.0,
        "blink_count": 0.0,
    },
}


# ============================================================
# VALIDATION AND LOADING
# ============================================================

def require_file(filepath: Path) -> None:
    """
    Raise a clear error when a required file is missing.
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"Required file was not found: {filepath}"
        )


def discover_log_paths() -> List[Path]:
    """
    Return manually selected logs or the two newest V2 logs.
    """

    if MANUAL_LOG_PATHS:
        paths = [
            Path(path)
            for path in MANUAL_LOG_PATHS
        ]

        for path in paths:
            require_file(path)

        return paths

    if not LOG_DIRECTORY.exists():
        raise FileNotFoundError(
            f"V2 log directory was not found: {LOG_DIRECTORY}"
        )

    candidates = sorted(
        LOG_DIRECTORY.glob(
            "driver_guardian_v2_session_*.csv"
        ),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if len(candidates) < 2:
        raise FileNotFoundError(
            "At least two V2 session logs are required. "
            f"Only {len(candidates)} were found in {LOG_DIRECTORY}."
        )

    return list(
        reversed(
            candidates[:2]
        )
    )


def convert_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert common Boolean representations to float 0.0 or 1.0.
    """

    text = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

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

    converted = converted.fillna(
        pd.to_numeric(
            series,
            errors="coerce",
        )
    )

    invalid = (
        converted.isna()
        & series.notna()
    )

    if invalid.any():
        examples = (
            series.loc[invalid]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{column_name} contains unsupported values: "
            f"{examples[:10]}"
        )

    return converted.astype("float64")


def prepare_log(
    dataframe: pd.DataFrame,
    filepath: Path,
    session_index: int,
) -> pd.DataFrame:
    """
    Validate and prepare one V2 live log.
    """

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{filepath.name} is missing features: {missing}"
        )

    if dataframe.empty:
        raise ValueError(
            f"{filepath.name} contains no rows."
        )

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
        ).astype("float64")

    dataframe["condition"] = (
        dataframe["condition"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    dataframe["session_name"] = (
        f"session_{session_index}_{filepath.stem}"
    )

    dataframe["source_log"] = str(filepath)

    dataframe["source_row"] = np.arange(
        len(dataframe),
        dtype=int,
    )

    return dataframe


def load_logs() -> Tuple[pd.DataFrame, List[Path]]:
    """
    Load and combine the selected V2 logs.
    """

    paths = discover_log_paths()

    prepared_logs = []

    for index, filepath in enumerate(
        paths,
        start=1,
    ):
        dataframe = pd.read_csv(filepath)

        prepared_logs.append(
            prepare_log(
                dataframe,
                filepath,
                index,
            )
        )

    combined = pd.concat(
        prepared_logs,
        ignore_index=True,
    )

    return combined, paths


def load_model_bundle() -> Dict[str, Any]:
    """
    Load and validate the V2 model bundle.
    """

    require_file(MODEL_PATH)

    bundle = joblib.load(MODEL_PATH)

    required = {
        "pipeline",
        "fatigue_threshold",
        "feature_columns",
    }

    missing = required.difference(
        bundle.keys()
    )

    if missing:
        raise KeyError(
            "Model bundle is missing keys: "
            f"{sorted(missing)}"
        )

    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature order does not match this diagnostic.\n"
            f"Model: {bundle['feature_columns']}\n"
            f"Expected: {FEATURE_COLUMNS}"
        )

    return bundle


# ============================================================
# COUNTERFACTUAL PREDICTIONS
# ============================================================

def apply_scenario(
    original_features: pd.DataFrame,
    changes: Dict[str, float],
) -> pd.DataFrame:
    """
    Copy features and apply one counterfactual scenario.
    """

    scenario_features = original_features.copy()

    for feature, value in changes.items():
        scenario_features[feature] = float(value)

    return scenario_features


def calculate_scenario_predictions(
    combined_logs: pd.DataFrame,
    pipeline,
    fatigue_threshold: float,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """
    Calculate predictions for all counterfactual scenarios.
    """

    original_features = combined_logs[
        FEATURE_COLUMNS
    ].copy()

    row_level = combined_logs[
        [
            "session_name",
            "source_log",
            "source_row",
            *FEATURE_COLUMNS,
        ]
    ].copy()

    scenario_rows = []
    session_rows = []

    original_probabilities: Optional[
        np.ndarray
    ] = None

    for scenario_name, changes in SCENARIOS.items():

        scenario_features = apply_scenario(
            original_features,
            changes,
        )

        probabilities = pipeline.predict_proba(
            scenario_features
        )[:, 1]

        predictions = (
            probabilities
            >= fatigue_threshold
        ).astype(int)

        row_level[
            f"{scenario_name}_fatigue_probability"
        ] = probabilities

        row_level[
            f"{scenario_name}_prediction"
        ] = predictions

        if scenario_name == "original":
            original_probabilities = probabilities

        scenario_rows.append(
            {
                "scenario": scenario_name,
                "samples": int(len(probabilities)),
                "mean_fatigue_probability": float(
                    probabilities.mean()
                ),
                "median_fatigue_probability": float(
                    np.median(probabilities)
                ),
                "minimum_fatigue_probability": float(
                    probabilities.min()
                ),
                "maximum_fatigue_probability": float(
                    probabilities.max()
                ),
                "predicted_fatigue_rows": int(
                    predictions.sum()
                ),
                "predicted_alert_rows": int(
                    (predictions == 0).sum()
                ),
                "predicted_fatigue_rate": float(
                    predictions.mean()
                ),
            }
        )

        temporary = pd.DataFrame(
            {
                "session_name": combined_logs[
                    "session_name"
                ].values,
                "probability": probabilities,
                "prediction": predictions,
            }
        )

        for session_name, group in temporary.groupby(
            "session_name",
            sort=True,
        ):
            session_rows.append(
                {
                    "scenario": scenario_name,
                    "session_name": session_name,
                    "samples": int(len(group)),
                    "mean_fatigue_probability": float(
                        group["probability"].mean()
                    ),
                    "median_fatigue_probability": float(
                        group["probability"].median()
                    ),
                    "minimum_fatigue_probability": float(
                        group["probability"].min()
                    ),
                    "maximum_fatigue_probability": float(
                        group["probability"].max()
                    ),
                    "predicted_fatigue_rows": int(
                        group["prediction"].sum()
                    ),
                    "predicted_alert_rows": int(
                        (group["prediction"] == 0).sum()
                    ),
                    "predicted_fatigue_rate": float(
                        group["prediction"].mean()
                    ),
                }
            )

    if original_probabilities is None:
        raise RuntimeError(
            "Original scenario was not evaluated."
        )

    scenario_summary = pd.DataFrame(
        scenario_rows
    )

    session_summary = pd.DataFrame(
        session_rows
    )

    for scenario_name in SCENARIOS:

        if scenario_name == "original":
            continue

        scenario_probabilities = row_level[
            f"{scenario_name}_fatigue_probability"
        ].to_numpy(dtype=float)

        probability_shift = (
            scenario_probabilities
            - original_probabilities
        )

        row_level[
            f"{scenario_name}_probability_shift"
        ] = probability_shift

        mask = (
            scenario_summary["scenario"]
            == scenario_name
        )

        scenario_summary.loc[
            mask,
            "mean_probability_shift_vs_original",
        ] = float(
            probability_shift.mean()
        )

        scenario_summary.loc[
            mask,
            "mean_absolute_probability_shift",
        ] = float(
            np.abs(probability_shift).mean()
        )

        scenario_summary.loc[
            mask,
            "rows_changed_by_at_least_10_percent",
        ] = int(
            (
                np.abs(probability_shift)
                >= 0.10
            ).sum()
        )

        original_predictions = row_level[
            "original_prediction"
        ].to_numpy(dtype=int)

        scenario_predictions = row_level[
            f"{scenario_name}_prediction"
        ].to_numpy(dtype=int)

        scenario_summary.loc[
            mask,
            "rows_whose_class_changed",
        ] = int(
            (
                original_predictions
                != scenario_predictions
            ).sum()
        )

    original_mask = (
        scenario_summary["scenario"]
        == "original"
    )

    scenario_summary.loc[
        original_mask,
        "mean_probability_shift_vs_original",
    ] = 0.0

    scenario_summary.loc[
        original_mask,
        "mean_absolute_probability_shift",
    ] = 0.0

    scenario_summary.loc[
        original_mask,
        "rows_changed_by_at_least_10_percent",
    ] = 0

    scenario_summary.loc[
        original_mask,
        "rows_whose_class_changed",
    ] = 0

    return (
        row_level,
        scenario_summary,
        session_summary,
    )


# ============================================================
# INTERPRETATION
# ============================================================

def build_findings(
    scenario_summary: pd.DataFrame,
) -> List[str]:
    """
    Convert scenario effects into engineering findings.
    """

    indexed = scenario_summary.set_index(
        "scenario"
    )

    findings = []

    original_rate = float(
        indexed.loc[
            "original",
            "predicted_fatigue_rate",
        ]
    )

    findings.append(
        "The original live logs were predicted as Fatigue on "
        f"{original_rate:.1%} of rows."
    )

    hands_zero_shift = float(
        indexed.loc[
            "hands_always_0",
            "mean_absolute_probability_shift",
        ]
    )

    hands_one_shift = float(
        indexed.loc[
            "hands_always_1",
            "mean_absolute_probability_shift",
        ]
    )

    blink_zero_shift = float(
        indexed.loc[
            "blink_always_0",
            "mean_absolute_probability_shift",
        ]
    )

    combined_zero_shift = float(
        indexed.loc[
            "hands_0_blink_0",
            "mean_absolute_probability_shift",
        ]
    )

    findings.append(
        "Forcing hands_detected to 0 changed Fatigue probability "
        f"by {hands_zero_shift:.1%} on average."
    )

    findings.append(
        "Forcing hands_detected to 1 changed Fatigue probability "
        f"by {hands_one_shift:.1%} on average."
    )

    findings.append(
        "Resetting blink_count to 0 changed Fatigue probability "
        f"by {blink_zero_shift:.1%} on average."
    )

    findings.append(
        "Forcing both hands_detected=0 and blink_count=0 changed "
        f"Fatigue probability by {combined_zero_shift:.1%} on average."
    )

    largest_non_original = (
        scenario_summary[
            scenario_summary[
                "scenario"
            ]
            != "original"
        ]
        .sort_values(
            "mean_absolute_probability_shift",
            ascending=False,
        )
        .iloc[0]
    )

    findings.append(
        "The strongest tested counterfactual was "
        f"{largest_non_original['scenario']}, with an average absolute "
        "probability change of "
        f"{largest_non_original['mean_absolute_probability_shift']:.1%}."
    )

    if (
        max(
            hands_zero_shift,
            hands_one_shift,
        )
        >= 0.20
    ):
        findings.append(
            "hands_detected has a large effect on live predictions and "
            "should be tested in an ablation model."
        )
    else:
        findings.append(
            "hands_detected alone does not explain most live predictions."
        )

    if blink_zero_shift >= 0.20:
        findings.append(
            "Cumulative blink_count has a large effect and may encode "
            "recording progress rather than transferable fatigue."
        )
    else:
        findings.append(
            "blink_count alone does not explain most live predictions."
        )

    return findings


# ============================================================
# PLOTS
# ============================================================

def save_probability_shift_plot(
    scenario_summary: pd.DataFrame,
) -> None:
    """
    Save average probability changes for all scenarios.
    """

    plot_data = (
        scenario_summary[
            scenario_summary[
                "scenario"
            ]
            != "original"
        ]
        .sort_values(
            "mean_absolute_probability_shift",
            ascending=False,
        )
    )

    plt.figure(
        figsize=(11, 6)
    )

    plt.bar(
        plot_data["scenario"],
        plot_data[
            "mean_absolute_probability_shift"
        ],
    )

    plt.ylabel(
        "Mean absolute Fatigue-probability change"
    )

    plt.xlabel(
        "Counterfactual scenario"
    )

    plt.title(
        "DriverGuardianAI V2 Feature Dependence"
    )

    plt.xticks(
        rotation=35,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        PROBABILITY_SHIFT_PLOT_PATH,
        dpi=300,
    )

    plt.close()


def save_original_probability_plot(
    row_level: pd.DataFrame,
) -> None:
    """
    Plot original live Fatigue probabilities by session.
    """

    plt.figure(
        figsize=(11, 6)
    )

    for session_name, group in row_level.groupby(
        "session_name",
        sort=True,
    ):
        plt.plot(
            group["source_row"],
            group[
                "original_fatigue_probability"
            ],
            label=session_name,
        )

    plt.axhline(
        0.69,
        linestyle="--",
        label="Saved threshold 0.69",
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Inference row"
    )

    plt.ylabel(
        "Fatigue probability"
    )

    plt.title(
        "Original V2 Live Probabilities"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ORIGINAL_PROBABILITY_PLOT_PATH,
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run the complete counterfactual diagnostic.
    """

    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Feature Dependence Diagnostic")
    print("=" * 72)

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading V2 model bundle..."
    )

    bundle = load_model_bundle()

    pipeline = bundle["pipeline"]

    fatigue_threshold = float(
        bundle["fatigue_threshold"]
    )

    print(
        f"Fatigue threshold: {fatigue_threshold:.2f}"
    )

    print(
        "\nLoading V2 live logs..."
    )

    combined_logs, log_paths = load_logs()

    for path in log_paths:
        print(
            f"- {path}"
        )

    print(
        f"Combined rows: {len(combined_logs)}"
    )

    print(
        "\nRunning counterfactual predictions..."
    )

    (
        row_level,
        scenario_summary,
        session_summary,
    ) = calculate_scenario_predictions(
        combined_logs,
        pipeline,
        fatigue_threshold,
    )

    findings = build_findings(
        scenario_summary
    )

    row_level.to_csv(
        ROW_LEVEL_PATH,
        index=False,
    )

    scenario_summary.to_csv(
        SCENARIO_SUMMARY_PATH,
        index=False,
    )

    session_summary.to_csv(
        SESSION_SCENARIO_SUMMARY_PATH,
        index=False,
    )

    save_probability_shift_plot(
        scenario_summary
    )

    save_original_probability_plot(
        row_level
    )

    summary = {
        "project": "DriverGuardianAI V2",
        "model_path": str(MODEL_PATH),
        "fatigue_threshold": (
            fatigue_threshold
        ),
        "log_paths": [
            str(path)
            for path in log_paths
        ],
        "rows": int(
            len(combined_logs)
        ),
        "training_blink_median": (
            TRAINING_BLINK_MEDIAN
        ),
        "scenarios": (
            scenario_summary.to_dict(
                orient="records"
            )
        ),
        "findings": findings,
        "important_note": (
            "This is a counterfactual feature-dependence test. "
            "It changes feature values without retraining the model, "
            "so it diagnoses model sensitivity rather than estimating "
            "the accuracy of a future ablation model."
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

    print("\nScenario summary:")

    display_columns = [
        "scenario",
        "samples",
        "mean_fatigue_probability",
        "predicted_fatigue_rate",
        "mean_probability_shift_vs_original",
        "mean_absolute_probability_shift",
        "rows_whose_class_changed",
    ]

    print(
        scenario_summary[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nSession-level scenario summary:"
    )

    print(
        session_summary.to_string(
            index=False
        )
    )

    print(
        "\nMain findings:"
    )

    for finding in findings:
        print(
            f"- {finding}"
        )

    print("\n" + "=" * 72)
    print("Feature dependence diagnostic completed.")
    print("=" * 72)

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIRECTORY
    )


if __name__ == "__main__":
    main()