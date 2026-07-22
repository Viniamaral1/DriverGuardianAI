"""
DriverGuardianAI V2
Experiment 23: Core Live Prediction Explainability

Purpose
-------
Explain the core-behaviour Histogram Gradient Boosting model on two
fresh labelled live sessions.

Model features
--------------
- ear
- yawn_score
- head_tilt

The script calculates:
- global SHAP importance;
- per-session SHAP importance;
- row-level feature contributions;
- strongest false-fatigue explanations from the Alert session;
- strongest correct-fatigue explanations from the Fatigue session;
- probability-versus-feature plots.

Required files
--------------
models/v2/ablation/driver_guardian_core_behaviour.joblib

logs/v2/core_behaviour/
    driver_guardian_v2_session_20260718_174331.csv  -> Alert
    driver_guardian_v2_session_20260718_174536.csv  -> Fatigue

Important
---------
Confirm the labels in LIVE_LOGS before running.

Outputs
-------
results/v2/core_live_explainability/
    global_shap_importance.csv
    session_shap_importance.csv
    row_level_explanations.csv
    selected_row_explanations.csv
    experiment_summary.json
    global_shap_importance.png
    session_shap_importance.png
    probability_vs_ear.png
    probability_vs_yawn_score.png
    probability_vs_head_tilt.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "v2"
    / "ablation"
    / "driver_guardian_core_behaviour.joblib"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "core_live_explainability"
)

GLOBAL_IMPORTANCE_PATH = (
    RESULTS_DIRECTORY
    / "global_shap_importance.csv"
)

SESSION_IMPORTANCE_PATH = (
    RESULTS_DIRECTORY
    / "session_shap_importance.csv"
)

ROW_LEVEL_PATH = (
    RESULTS_DIRECTORY
    / "row_level_explanations.csv"
)

SELECTED_ROWS_PATH = (
    RESULTS_DIRECTORY
    / "selected_row_explanations.csv"
)

SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "experiment_summary.json"
)

GLOBAL_IMPORTANCE_PLOT_PATH = (
    RESULTS_DIRECTORY
    / "global_shap_importance.png"
)

SESSION_IMPORTANCE_PLOT_PATH = (
    RESULTS_DIRECTORY
    / "session_shap_importance.png"
)


# ============================================================
# LIVE SESSION LABELS
# ============================================================

LIVE_LOGS = [
    {
        "path": (
            PROJECT_ROOT
            / "logs"
            / "v2"
            / "core_behaviour"
            / "driver_guardian_v2_session_20260718_174331.csv"
        ),
        "actual_label": "Alert",
        "actual_class": 0,
        "session_name": "live_alert",
    },
    {
        "path": (
            PROJECT_ROOT
            / "logs"
            / "v2"
            / "core_behaviour"
            / "driver_guardian_v2_session_20260718_174536.csv"
        ),
        "actual_label": "Fatigue",
        "actual_class": 1,
        "session_name": "live_fatigue",
    },
]


# ============================================================
# SETTINGS
# ============================================================

FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
]

MAX_FALSE_FATIGUE_ROWS = 15
MAX_CORRECT_FATIGUE_ROWS = 15

RANDOM_STATE = 42


# ============================================================
# HELPERS
# ============================================================

def require_file(
    filepath: Path,
) -> None:
    """
    Raise a clear error when a required file is missing.
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"Required file was not found: {filepath}"
        )


def prepare_live_dataframe(
    dataframe: pd.DataFrame,
    specification: Dict[str, Any],
) -> pd.DataFrame:
    """
    Validate and prepare one live-session dataframe.
    """

    missing = [
        feature
        for feature in FEATURE_COLUMNS
        if feature not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{specification['path']} is missing features: {missing}"
        )

    dataframe = dataframe.copy()

    for feature in FEATURE_COLUMNS:
        dataframe[feature] = pd.to_numeric(
            dataframe[feature],
            errors="coerce",
        ).astype("float64")

    dataframe = dataframe.dropna(
        subset=FEATURE_COLUMNS
    ).reset_index(
        drop=True
    )

    dataframe["session_name"] = str(
        specification["session_name"]
    )

    dataframe["actual_label"] = str(
        specification["actual_label"]
    )

    dataframe["actual_class"] = int(
        specification["actual_class"]
    )

    dataframe["source_log"] = str(
        specification["path"]
    )

    dataframe["source_row"] = np.arange(
        len(dataframe),
        dtype=int,
    )

    return dataframe


def load_live_logs() -> pd.DataFrame:
    """
    Load and combine the two fresh core-behaviour sessions.
    """

    frames = []

    for specification in LIVE_LOGS:
        filepath = Path(
            specification["path"]
        )

        require_file(
            filepath
        )

        dataframe = pd.read_csv(
            filepath
        )

        frames.append(
            prepare_live_dataframe(
                dataframe,
                specification,
            )
        )

    return pd.concat(
        frames,
        ignore_index=True,
    )


def load_model_bundle() -> Dict[str, Any]:
    """
    Load and validate the saved core model.
    """

    require_file(
        MODEL_PATH
    )

    bundle = joblib.load(
        MODEL_PATH
    )

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

    feature_columns = list(
        bundle["feature_columns"]
    )

    if feature_columns != FEATURE_COLUMNS:
        raise ValueError(
            "Unexpected model feature order.\n"
            f"Expected: {FEATURE_COLUMNS}\n"
            f"Found: {feature_columns}"
        )

    return bundle


def extract_transformed_model(
    pipeline,
    feature_dataframe: pd.DataFrame,
) -> Tuple[
    Any,
    np.ndarray,
    List[str],
]:
    """
    Apply fitted preprocessing and return the underlying classifier.
    """

    preprocessor = pipeline.named_steps[
        "preprocessor"
    ]

    classifier = pipeline.named_steps[
        "classifier"
    ]

    transformed = preprocessor.transform(
        feature_dataframe
    )

    transformed = np.asarray(
        transformed,
        dtype=np.float64,
    )

    try:
        transformed_names = list(
            preprocessor.get_feature_names_out()
        )
    except Exception:
        transformed_names = list(
            FEATURE_COLUMNS
        )

    if transformed.shape[1] != len(
        transformed_names
    ):
        transformed_names = [
            f"feature_{index}"
            for index in range(
                transformed.shape[1]
            )
        ]

    return (
        classifier,
        transformed,
        transformed_names,
    )


def calculate_shap_values(
    classifier,
    transformed_features: np.ndarray,
) -> Tuple[
    np.ndarray,
    str,
    Any,
]:
    """
    Calculate SHAP values using TreeExplainer.

    HistGradientBoostingClassifier is supported by current SHAP
    TreeExplainer releases. A clear error is raised if the local SHAP
    version does not support it.
    """

    try:
        explainer = shap.TreeExplainer(
            classifier
        )

        explanation = explainer(
            transformed_features
        )

        values = np.asarray(
            explanation.values
        )

        # Binary classifiers may return:
        # rows x features
        # or rows x features x classes.
        if values.ndim == 3:
            values = values[:, :, 1]

        if values.ndim != 2:
            raise ValueError(
                "Unexpected SHAP value shape: "
                f"{values.shape}"
            )

        return (
            values.astype(
                "float64"
            ),
            "tree",
            explanation,
        )

    except Exception as error:
        raise RuntimeError(
            "SHAP TreeExplainer could not explain the saved "
            "Histogram Gradient Boosting classifier. Update SHAP "
            "in the current environment or send this error message "
            "for a permutation-explainer version.\n"
            f"Original error: {error}"
        ) from error


# ============================================================
# REPORTS
# ============================================================

def create_global_importance(
    shap_values: np.ndarray,
    transformed_names: List[str],
) -> pd.DataFrame:
    """
    Calculate global mean absolute SHAP importance.
    """

    mean_absolute = np.mean(
        np.abs(
            shap_values
        ),
        axis=0,
    )

    total = float(
        mean_absolute.sum()
    )

    if total <= 1e-15:
        relative = np.zeros_like(
            mean_absolute
        )
    else:
        relative = (
            mean_absolute
            / total
        )

    return pd.DataFrame(
        {
            "feature": transformed_names,
            "mean_absolute_shap": (
                mean_absolute
            ),
            "relative_importance": (
                relative
            ),
        }
    ).sort_values(
        "mean_absolute_shap",
        ascending=False,
    ).reset_index(
        drop=True
    )


def create_session_importance(
    live: pd.DataFrame,
    shap_values: np.ndarray,
    transformed_names: List[str],
) -> pd.DataFrame:
    """
    Calculate mean absolute SHAP importance by live session.
    """

    rows = []

    for session_name in live[
        "session_name"
    ].unique():

        mask = (
            live[
                "session_name"
            ].to_numpy()
            == session_name
        )

        session_values = shap_values[
            mask
        ]

        mean_absolute = np.mean(
            np.abs(
                session_values
            ),
            axis=0,
        )

        total = float(
            mean_absolute.sum()
        )

        if total <= 1e-15:
            relative = np.zeros_like(
                mean_absolute
            )
        else:
            relative = (
                mean_absolute
                / total
            )

        for (
            feature,
            absolute_value,
            relative_value,
        ) in zip(
            transformed_names,
            mean_absolute,
            relative,
        ):
            rows.append(
                {
                    "session_name": session_name,
                    "feature": feature,
                    "mean_absolute_shap": float(
                        absolute_value
                    ),
                    "relative_importance": float(
                        relative_value
                    ),
                }
            )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "session_name",
            "mean_absolute_shap",
        ],
        ascending=[
            True,
            False,
        ],
    )


def create_row_level_report(
    live: pd.DataFrame,
    probabilities: np.ndarray,
    threshold: float,
    shap_values: np.ndarray,
    transformed_names: List[str],
) -> pd.DataFrame:
    """
    Create row-level predictions and SHAP contributions.
    """

    output = live[
        [
            "session_name",
            "actual_label",
            "actual_class",
            "source_log",
            "source_row",
            *FEATURE_COLUMNS,
        ]
    ].copy()

    output[
        "fatigue_probability"
    ] = probabilities

    output[
        "decision_threshold"
    ] = float(
        threshold
    )

    output[
        "predicted_class"
    ] = (
        probabilities
        >= threshold
    ).astype(
        int
    )

    output[
        "predicted_label"
    ] = np.where(
        output[
            "predicted_class"
        ]
        == 1,
        "Fatigue",
        "Alert",
    )

    output[
        "correct_prediction"
    ] = (
        output[
            "predicted_class"
        ]
        == output[
            "actual_class"
        ]
    )

    for index, feature in enumerate(
        transformed_names
    ):
        output[
            f"shap_{feature}"
        ] = shap_values[
            :,
            index,
        ]

    output[
        "dominant_feature"
    ] = [
        transformed_names[
            int(
                np.argmax(
                    np.abs(
                        row
                    )
                )
            )
        ]
        for row in shap_values
    ]

    output[
        "dominant_absolute_shap"
    ] = np.max(
        np.abs(
            shap_values
        ),
        axis=1,
    )

    return output


def select_explanation_rows(
    row_level: pd.DataFrame,
) -> pd.DataFrame:
    """
    Select important failure and success examples.
    """

    false_fatigue = row_level[
        (
            row_level[
                "actual_label"
            ]
            == "Alert"
        )
        &
        (
            row_level[
                "predicted_label"
            ]
            == "Fatigue"
        )
    ].sort_values(
        "fatigue_probability",
        ascending=False,
    ).head(
        MAX_FALSE_FATIGUE_ROWS
    ).copy()

    false_fatigue[
        "selection_type"
    ] = "strongest_false_fatigue"

    correct_fatigue = row_level[
        (
            row_level[
                "actual_label"
            ]
            == "Fatigue"
        )
        &
        (
            row_level[
                "predicted_label"
            ]
            == "Fatigue"
        )
    ].sort_values(
        "fatigue_probability",
        ascending=False,
    ).head(
        MAX_CORRECT_FATIGUE_ROWS
    ).copy()

    correct_fatigue[
        "selection_type"
    ] = "strongest_correct_fatigue"

    return pd.concat(
        [
            false_fatigue,
            correct_fatigue,
        ],
        ignore_index=True,
    )


# ============================================================
# PLOTS
# ============================================================

def save_global_importance_plot(
    global_importance: pd.DataFrame,
) -> None:
    """
    Save global SHAP importance.
    """

    plot_data = global_importance.sort_values(
        "mean_absolute_shap",
        ascending=True,
    )

    plt.figure(
        figsize=(9, 5)
    )

    plt.barh(
        plot_data[
            "feature"
        ],
        plot_data[
            "mean_absolute_shap"
        ],
    )

    plt.xlabel(
        "Mean absolute SHAP value"
    )

    plt.ylabel(
        "Feature"
    )

    plt.title(
        "Core Model Global Live SHAP Importance"
    )

    plt.tight_layout()

    plt.savefig(
        GLOBAL_IMPORTANCE_PLOT_PATH,
        dpi=300,
    )

    plt.close()


def save_session_importance_plot(
    session_importance: pd.DataFrame,
) -> None:
    """
    Compare SHAP importance between Alert and Fatigue sessions.
    """

    pivot = session_importance.pivot(
        index="feature",
        columns="session_name",
        values="mean_absolute_shap",
    ).fillna(
        0.0
    )

    positions = np.arange(
        len(
            pivot.index
        )
    )

    sessions = list(
        pivot.columns
    )

    width = (
        0.8
        / max(
            len(sessions),
            1,
        )
    )

    plt.figure(
        figsize=(10, 6)
    )

    for session_index, session_name in enumerate(
        sessions
    ):
        offset = (
            session_index
            - (
                len(sessions)
                - 1
            )
            / 2.0
        ) * width

        plt.bar(
            positions
            + offset,
            pivot[
                session_name
            ].to_numpy(),
            width=width,
            label=session_name,
        )

    plt.xticks(
        positions,
        pivot.index,
    )

    plt.xlabel(
        "Feature"
    )

    plt.ylabel(
        "Mean absolute SHAP value"
    )

    plt.title(
        "Core Model SHAP Importance by Live Session"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        SESSION_IMPORTANCE_PLOT_PATH,
        dpi=300,
    )

    plt.close()


def save_probability_feature_plots(
    row_level: pd.DataFrame,
) -> None:
    """
    Save Fatigue probability against each core feature.
    """

    for feature in FEATURE_COLUMNS:

        plt.figure(
            figsize=(9, 6)
        )

        for session_name, group in row_level.groupby(
            "session_name",
            sort=True,
        ):
            plt.scatter(
                group[
                    feature
                ],
                group[
                    "fatigue_probability"
                ],
                alpha=0.65,
                label=session_name,
            )

        plt.axhline(
            float(
                row_level[
                    "decision_threshold"
                ].iloc[
                    0
                ]
            ),
            linestyle="--",
            label="Decision threshold",
        )

        plt.ylim(
            -0.02,
            1.02,
        )

        plt.xlabel(
            feature
        )

        plt.ylabel(
            "Fatigue probability"
        )

        plt.title(
            f"Fatigue Probability vs {feature}"
        )

        plt.legend()

        plt.tight_layout()

        plt.savefig(
            RESULTS_DIRECTORY
            / (
                "probability_vs_"
                f"{feature}.png"
            ),
            dpi=300,
        )

        plt.close()


# ============================================================
# FINDINGS
# ============================================================

def build_findings(
    global_importance: pd.DataFrame,
    session_importance: pd.DataFrame,
    row_level: pd.DataFrame,
) -> List[str]:
    """
    Create concise engineering conclusions.
    """

    findings = []

    strongest = global_importance.iloc[
        0
    ]

    findings.append(
        "The most influential feature across the two fresh live "
        f"sessions was {strongest['feature']}, accounting for "
        f"{strongest['relative_importance']:.1%} of total mean "
        "absolute SHAP importance."
    )

    for session_name in row_level[
        "session_name"
    ].unique():

        session_rows = row_level[
            row_level[
                "session_name"
            ]
            == session_name
        ]

        accuracy = float(
            session_rows[
                "correct_prediction"
            ].mean()
        )

        fatigue_rate = float(
            (
                session_rows[
                    "predicted_label"
                ]
                == "Fatigue"
            ).mean()
        )

        findings.append(
            f"{session_name}: frame accuracy was {accuracy:.1%}, "
            f"and {fatigue_rate:.1%} of frames were predicted as "
            "Fatigue."
        )

        importance_rows = session_importance[
            session_importance[
                "session_name"
            ]
            == session_name
        ].sort_values(
            "mean_absolute_shap",
            ascending=False,
        )

        if not importance_rows.empty:
            top_row = importance_rows.iloc[
                0
            ]

            findings.append(
                f"{session_name}: the largest SHAP dependence was "
                f"{top_row['feature']} "
                f"({top_row['relative_importance']:.1%} of session "
                "importance)."
            )

    alert_false_fatigue = row_level[
        (
            row_level[
                "actual_label"
            ]
            == "Alert"
        )
        &
        (
            row_level[
                "predicted_label"
            ]
            == "Fatigue"
        )
    ]

    if not alert_false_fatigue.empty:
        dominant_counts = (
            alert_false_fatigue[
                "dominant_feature"
            ]
            .value_counts(
                normalize=True
            )
        )

        dominant_feature = str(
            dominant_counts.index[
                0
            ]
        )

        dominant_rate = float(
            dominant_counts.iloc[
                0
            ]
        )

        findings.append(
            "Among false-Fatigue rows in the Alert session, "
            f"{dominant_feature} was the largest absolute SHAP "
            f"contributor in {dominant_rate:.1%} of cases."
        )

    return findings


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run core live explainability analysis.
    """

    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Experiment 23: Core Live Explainability")
    print("=" * 72)

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading core model..."
    )

    bundle = load_model_bundle()

    pipeline = bundle[
        "pipeline"
    ]

    fatigue_threshold = float(
        bundle[
            "fatigue_threshold"
        ]
    )

    print(
        f"Fatigue threshold: {fatigue_threshold:.2f}"
    )

    print(
        f"Features: {bundle['feature_columns']}"
    )

    print(
        "\nLoading fresh live sessions..."
    )

    live = load_live_logs()

    print(
        live[
            [
                "session_name",
                "actual_label",
            ]
        ]
        .value_counts()
        .to_string()
    )

    feature_dataframe = live[
        FEATURE_COLUMNS
    ].copy()

    probabilities = pipeline.predict_proba(
        feature_dataframe
    )[:, 1]

    (
        classifier,
        transformed_features,
        transformed_names,
    ) = extract_transformed_model(
        pipeline,
        feature_dataframe,
    )

    print(
        "\nTransformed feature names:"
    )

    for feature in transformed_names:
        print(
            f"- {feature}"
        )

    print(
        "\nCalculating SHAP explanations..."
    )

    (
        shap_values,
        explainer_type,
        _,
    ) = calculate_shap_values(
        classifier,
        transformed_features,
    )

    global_importance = create_global_importance(
        shap_values,
        transformed_names,
    )

    session_importance = create_session_importance(
        live,
        shap_values,
        transformed_names,
    )

    row_level = create_row_level_report(
        live,
        probabilities,
        fatigue_threshold,
        shap_values,
        transformed_names,
    )

    selected_rows = select_explanation_rows(
        row_level
    )

    global_importance.to_csv(
        GLOBAL_IMPORTANCE_PATH,
        index=False,
    )

    session_importance.to_csv(
        SESSION_IMPORTANCE_PATH,
        index=False,
    )

    row_level.to_csv(
        ROW_LEVEL_PATH,
        index=False,
    )

    selected_rows.to_csv(
        SELECTED_ROWS_PATH,
        index=False,
    )

    save_global_importance_plot(
        global_importance
    )

    save_session_importance_plot(
        session_importance
    )

    save_probability_feature_plots(
        row_level
    )

    findings = build_findings(
        global_importance,
        session_importance,
        row_level,
    )

    summary = {
        "project": "DriverGuardianAI V2",
        "experiment": "core_live_explainability",
        "model_path": str(
            MODEL_PATH
        ),
        "fatigue_threshold": (
            fatigue_threshold
        ),
        "explainer_type": (
            explainer_type
        ),
        "live_log_configuration": [
            {
                "path": str(
                    item[
                        "path"
                    ]
                ),
                "actual_label": (
                    item[
                        "actual_label"
                    ]
                ),
                "session_name": (
                    item[
                        "session_name"
                    ]
                ),
            }
            for item in LIVE_LOGS
        ],
        "global_importance": (
            global_importance.to_dict(
                orient="records"
            )
        ),
        "findings": findings,
        "important_note": (
            "SHAP values explain the fitted model's local decisions. "
            "They do not establish that a feature is causally related "
            "to real driver fatigue."
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

    print(
        "\nGlobal SHAP importance:"
    )

    print(
        global_importance.to_string(
            index=False
        )
    )

    print(
        "\nSession SHAP importance:"
    )

    print(
        session_importance.to_string(
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

    print(
        "\n" + "=" * 72
    )

    print(
        "Experiment 23 completed successfully."
    )

    print(
        "=" * 72
    )

    print(
        "\nResults saved to:"
    )

    print(
        RESULTS_DIRECTORY
    )


if __name__ == "__main__":
    main()