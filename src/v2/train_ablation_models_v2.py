"""
DriverGuardianAI V2
Feature Ablation Model Experiment

Trains and compares four Histogram Gradient Boosting variants:

1. full
2. no_hands
3. no_hands_no_blink
4. core_behaviour

Data protocol
-------------
- Fit model and preprocessing on train.csv only.
- Select Fatigue threshold on calibration.csv only.
- Evaluate once on untouched test.csv.
- Evaluate deployment transfer on two labelled live logs.

Binary target
-------------
Alert = 0
Mild Fatigue or Moderate Fatigue = 1

Required files
--------------
data/splits/v2/train.csv
data/splits/v2/calibration.csv
data/splits/v2/test.csv
logs/v2/driver_guardian_v2_session_20260717_154546.csv
logs/v2/driver_guardian_v2_session_20260717_154904.csv

IMPORTANT
---------
The live-session labels below must match what you actually performed.
By default this script assumes:

- 15:45:46 session = Alert
- 15:49:04 session = Fatigue

Swap the two labels in LIVE_LOGS if that order is wrong.

Outputs
-------
models/v2/ablation/
    driver_guardian_<variant>.joblib

results/v2/ablation_models/
    model_comparison.csv
    test_metrics.csv
    live_metrics.csv
    threshold_search_<variant>.csv
    predictions_test_<variant>.csv
    predictions_live_<variant>.csv
    experiment_summary.json
    test_balanced_accuracy.png
    live_balanced_accuracy.png
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils.class_weight import compute_sample_weight


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "v2"
    / "train.csv"
)

CALIBRATION_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "v2"
    / "calibration.csv"
)

TEST_PATH = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "v2"
    / "test.csv"
)

MODEL_DIRECTORY = (
    PROJECT_ROOT
    / "models"
    / "v2"
    / "ablation"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "ablation_models"
)

MODEL_COMPARISON_PATH = (
    RESULTS_DIRECTORY
    / "model_comparison.csv"
)

TEST_METRICS_PATH = (
    RESULTS_DIRECTORY
    / "test_metrics.csv"
)

LIVE_METRICS_PATH = (
    RESULTS_DIRECTORY
    / "live_metrics.csv"
)

SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "experiment_summary.json"
)

TEST_PLOT_PATH = (
    RESULTS_DIRECTORY
    / "test_balanced_accuracy.png"
)

LIVE_PLOT_PATH = (
    RESULTS_DIRECTORY
    / "live_balanced_accuracy.png"
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
            / "driver_guardian_v2_session_20260717_154546.csv"
        ),
        "actual_label": "Alert",
        "session_name": "live_alert",
    },
    {
        "path": (
            PROJECT_ROOT
            / "logs"
            / "v2"
            / "driver_guardian_v2_session_20260717_154904.csv"
        ),
        "actual_label": "Fatigue",
        "session_name": "live_fatigue",
    },
]


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

RANDOM_STATE = 42

TARGET_COLUMN = "fatigue_level"

TARGET_MAPPING = {
    "Alert": 0,
    "Mild Fatigue": 1,
    "Moderate Fatigue": 1,
}

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

MODEL_VARIANTS = {
    "full": [
        "ear",
        "yawn_score",
        "head_tilt",
        "hands_detected",
        "condition",
        "low_light",
        "face_confidence",
        "blink_count",
    ],
    "no_hands": [
        "ear",
        "yawn_score",
        "head_tilt",
        "condition",
        "low_light",
        "face_confidence",
        "blink_count",
    ],
    "no_hands_no_blink": [
        "ear",
        "yawn_score",
        "head_tilt",
        "condition",
        "low_light",
        "face_confidence",
    ],
    "core_behaviour": [
        "ear",
        "yawn_score",
        "head_tilt",
    ],
}

CATEGORICAL_CANDIDATES = {
    "condition",
}

BINARY_CANDIDATES = {
    "hands_detected",
    "low_light",
}

THRESHOLD_MIN = 0.20
THRESHOLD_MAX = 0.95
THRESHOLD_STEP = 0.01

MINIMUM_DESIRED_SENSITIVITY = 0.60
FALSE_POSITIVE_PENALTY = 0.10
SENSITIVITY_SHORTFALL_PENALTY = 0.30


# ============================================================
# UTILITIES
# ============================================================

def require_file(filepath: Path) -> None:
    """
    Raise a clear error when a required file is absent.
    """

    if not filepath.exists():
        raise FileNotFoundError(
            f"Required file was not found: {filepath}"
        )


def convert_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert Boolean-like values into float 0.0 or 1.0.
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

    invalid_mask = (
        converted.isna()
        & series.notna()
    )

    if invalid_mask.any():
        examples = (
            series.loc[invalid_mask]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"{column_name} contains unsupported values: "
            f"{examples[:10]}"
        )

    return converted.astype("float64")


def prepare_features(
    dataframe: pd.DataFrame,
    dataset_name: str,
) -> pd.DataFrame:
    """
    Type-convert all possible features without scaling.
    """

    missing = [
        feature
        for feature in ALL_FEATURES
        if feature not in dataframe.columns
    ]

    if missing:
        raise ValueError(
            f"{dataset_name} is missing features: {missing}"
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

    return dataframe


def prepare_split(
    filepath: Path,
    split_name: str,
) -> pd.DataFrame:
    """
    Load one labelled participant split.
    """

    require_file(filepath)

    dataframe = prepare_features(
        pd.read_csv(filepath),
        split_name,
    )

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"{split_name} is missing {TARGET_COLUMN}."
        )

    unknown_labels = set(
        dataframe[TARGET_COLUMN]
        .dropna()
        .astype(str)
        .unique()
    ).difference(
        TARGET_MAPPING.keys()
    )

    if unknown_labels:
        raise ValueError(
            f"{split_name} contains unsupported labels: "
            f"{sorted(unknown_labels)}"
        )

    dataframe["binary_target"] = (
        dataframe[TARGET_COLUMN]
        .astype(str)
        .map(TARGET_MAPPING)
        .astype(int)
    )

    return dataframe


def load_live_logs() -> pd.DataFrame:
    """
    Load both manually labelled live sessions.
    """

    frames = []

    for specification in LIVE_LOGS:
        filepath = Path(
            specification["path"]
        )

        require_file(filepath)

        dataframe = prepare_features(
            pd.read_csv(filepath),
            filepath.name,
        )

        actual_label = str(
            specification["actual_label"]
        )

        if actual_label not in {
            "Alert",
            "Fatigue",
        }:
            raise ValueError(
                "Live actual_label must be Alert or Fatigue."
            )

        dataframe["actual_label"] = actual_label
        dataframe["actual_class"] = (
            0
            if actual_label == "Alert"
            else 1
        )

        dataframe["session_name"] = str(
            specification["session_name"]
        )

        dataframe["source_log"] = str(
            filepath
        )

        frames.append(dataframe)

    return pd.concat(
        frames,
        ignore_index=True,
    )


# ============================================================
# MODEL BUILDING
# ============================================================

def build_pipeline(
    feature_columns: List[str],
) -> Pipeline:
    """
    Create preprocessing and Histogram Gradient Boosting.
    """

    numeric_features = [
        feature
        for feature in feature_columns
        if feature not in CATEGORICAL_CANDIDATES
    ]

    categorical_features = [
        feature
        for feature in feature_columns
        if feature in CATEGORICAL_CANDIDATES
    ]

    transformers = []

    if numeric_features:
        numeric_pipeline = Pipeline(
            steps=[
                (
                    "imputer",
                    SimpleImputer(
                        strategy="median"
                    ),
                ),
            ]
        )

        transformers.append(
            (
                "numeric",
                numeric_pipeline,
                numeric_features,
            )
        )

    if categorical_features:
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

        transformers.append(
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            )
        )

    preprocessor = ColumnTransformer(
        transformers=transformers,
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
            (
                "preprocessor",
                preprocessor,
            ),
            (
                "classifier",
                classifier,
            ),
        ]
    )


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Calculate binary metrics at one threshold.
    """

    predictions = (
        probabilities
        >= threshold
    ).astype(int)

    matrix = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1],
    )

    true_negative = int(matrix[0, 0])
    false_positive = int(matrix[0, 1])
    false_negative = int(matrix[1, 0])
    true_positive = int(matrix[1, 1])

    sensitivity = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        ) > 0
        else 0.0
    )

    specificity = (
        true_negative
        / (
            true_negative
            + false_positive
        )
        if (
            true_negative
            + false_positive
        ) > 0
        else 0.0
    )

    false_positive_rate = (
        1.0
        - specificity
    )

    metrics = {
        "threshold": float(threshold),
        "samples": int(len(y_true)),
        "accuracy": float(
            accuracy_score(
                y_true,
                predictions,
            )
        ),
        "balanced_accuracy": float(
            balanced_accuracy_score(
                y_true,
                predictions,
            )
        ),
        "precision_fatigue": float(
            precision_score(
                y_true,
                predictions,
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
        "false_positive_rate": float(
            false_positive_rate
        ),
        "f1_fatigue": float(
            f1_score(
                y_true,
                predictions,
                pos_label=1,
                zero_division=0,
            )
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive,
    }

    if len(
        np.unique(y_true)
    ) == 2:
        metrics["roc_auc"] = float(
            roc_auc_score(
                y_true,
                probabilities,
            )
        )

        metrics["average_precision"] = float(
            average_precision_score(
                y_true,
                probabilities,
            )
        )
    else:
        metrics["roc_auc"] = None
        metrics["average_precision"] = None

    return metrics


def search_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
) -> Tuple[
    float,
    pd.DataFrame,
]:
    """
    Select threshold using calibration data only.
    """

    rows = []

    best_threshold = 0.50
    best_score = -np.inf

    thresholds = np.arange(
        THRESHOLD_MIN,
        THRESHOLD_MAX
        + THRESHOLD_STEP,
        THRESHOLD_STEP,
    )

    for threshold in thresholds:
        metrics = calculate_metrics(
            y_true,
            probabilities,
            float(threshold),
        )

        sensitivity_shortfall = max(
            0.0,
            MINIMUM_DESIRED_SENSITIVITY
            - metrics[
                "recall_sensitivity"
            ],
        )

        selection_score = (
            metrics[
                "balanced_accuracy"
            ]
            - FALSE_POSITIVE_PENALTY
            * metrics[
                "false_positive_rate"
            ]
            - SENSITIVITY_SHORTFALL_PENALTY
            * sensitivity_shortfall
        )

        metrics[
            "selection_score"
        ] = float(
            selection_score
        )

        rows.append(metrics)

        if selection_score > best_score:
            best_score = selection_score
            best_threshold = float(
                threshold
            )

    results = pd.DataFrame(rows)

    results = results.sort_values(
        "selection_score",
        ascending=False,
    ).reset_index(drop=True)

    return best_threshold, results


# ============================================================
# TRAIN ONE VARIANT
# ============================================================

def train_variant(
    variant_name: str,
    feature_columns: List[str],
    train: pd.DataFrame,
    calibration: pd.DataFrame,
    test: pd.DataFrame,
    live: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Train, calibrate, test, and save one feature variant.
    """

    print("\n" + "-" * 72)
    print(f"Training variant: {variant_name}")
    print("-" * 72)

    print(
        f"Features ({len(feature_columns)}): "
        f"{feature_columns}"
    )

    X_train = train[
        feature_columns
    ].copy()

    y_train = train[
        "binary_target"
    ].to_numpy(dtype=int)

    X_calibration = calibration[
        feature_columns
    ].copy()

    y_calibration = calibration[
        "binary_target"
    ].to_numpy(dtype=int)

    X_test = test[
        feature_columns
    ].copy()

    y_test = test[
        "binary_target"
    ].to_numpy(dtype=int)

    X_live = live[
        feature_columns
    ].copy()

    y_live = live[
        "actual_class"
    ].to_numpy(dtype=int)

    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train,
    )

    pipeline = build_pipeline(
        feature_columns
    )

    pipeline.fit(
        X_train,
        y_train,
        classifier__sample_weight=(
            sample_weights
        ),
    )

    calibration_probabilities = (
        pipeline.predict_proba(
            X_calibration
        )[:, 1]
    )

    (
        selected_threshold,
        threshold_results,
    ) = search_threshold(
        y_calibration,
        calibration_probabilities,
    )

    threshold_results.to_csv(
        RESULTS_DIRECTORY
        / f"threshold_search_{variant_name}.csv",
        index=False,
    )

    calibration_metrics = calculate_metrics(
        y_calibration,
        calibration_probabilities,
        selected_threshold,
    )

    test_probabilities = pipeline.predict_proba(
        X_test
    )[:, 1]

    test_metrics = calculate_metrics(
        y_test,
        test_probabilities,
        selected_threshold,
    )

    live_probabilities = pipeline.predict_proba(
        X_live
    )[:, 1]

    live_metrics = calculate_metrics(
        y_live,
        live_probabilities,
        selected_threshold,
    )

    test_predictions = test[
        [
            "participant_id",
            "session_id",
            TARGET_COLUMN,
        ]
    ].copy()

    test_predictions[
        "actual_class"
    ] = y_test

    test_predictions[
        "fatigue_probability"
    ] = test_probabilities

    test_predictions[
        "predicted_class"
    ] = (
        test_probabilities
        >= selected_threshold
    ).astype(int)

    test_predictions.to_csv(
        RESULTS_DIRECTORY
        / f"predictions_test_{variant_name}.csv",
        index=False,
    )

    live_predictions = live[
        [
            "session_name",
            "actual_label",
            "source_log",
        ]
    ].copy()

    live_predictions[
        "actual_class"
    ] = y_live

    live_predictions[
        "fatigue_probability"
    ] = live_probabilities

    live_predictions[
        "predicted_class"
    ] = (
        live_probabilities
        >= selected_threshold
    ).astype(int)

    live_predictions.to_csv(
        RESULTS_DIRECTORY
        / f"predictions_live_{variant_name}.csv",
        index=False,
    )

    model_bundle = {
        "project_version": "v2",
        "experiment": (
            "feature_ablation"
        ),
        "variant_name": variant_name,
        "model_name": (
            "hist_gradient_boosting"
        ),
        "pipeline": pipeline,
        "fatigue_threshold": float(
            selected_threshold
        ),
        "feature_columns": (
            feature_columns
        ),
        "class_names": [
            "Alert",
            "Fatigue",
        ],
        "target_mapping": (
            TARGET_MAPPING
        ),
        "calibration_metrics": (
            calibration_metrics
        ),
        "test_metrics": (
            test_metrics
        ),
        "live_metrics": (
            live_metrics
        ),
    }

    model_path = (
        MODEL_DIRECTORY
        / (
            "driver_guardian_"
            f"{variant_name}.joblib"
        )
    )

    joblib.dump(
        model_bundle,
        model_path,
    )

    print(
        "Selected threshold: "
        f"{selected_threshold:.2f}"
    )

    print(
        "Test balanced accuracy: "
        f"{test_metrics['balanced_accuracy']:.3f}"
    )

    print(
        "Test sensitivity: "
        f"{test_metrics['recall_sensitivity']:.3f}"
    )

    print(
        "Test specificity: "
        f"{test_metrics['specificity']:.3f}"
    )

    print(
        "Live balanced accuracy: "
        f"{live_metrics['balanced_accuracy']:.3f}"
    )

    print(
        "Live sensitivity: "
        f"{live_metrics['recall_sensitivity']:.3f}"
    )

    print(
        "Live specificity: "
        f"{live_metrics['specificity']:.3f}"
    )

    return {
        "variant": variant_name,
        "features": feature_columns,
        "model_path": str(model_path),
        "selected_threshold": float(
            selected_threshold
        ),
        "calibration_metrics": (
            calibration_metrics
        ),
        "test_metrics": (
            test_metrics
        ),
        "live_metrics": (
            live_metrics
        ),
    }


# ============================================================
# PLOTS
# ============================================================

def save_metric_plot(
    comparison: pd.DataFrame,
    metric_column: str,
    filepath: Path,
    title: str,
) -> None:
    """
    Save one model-comparison bar chart.
    """

    plot_data = comparison.sort_values(
        metric_column,
        ascending=False,
    )

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        plot_data["variant"],
        plot_data[metric_column],
    )

    plt.ylim(
        0.0,
        1.05,
    )

    plt.xlabel(
        "Model variant"
    )

    plt.ylabel(
        metric_column.replace(
            "_",
            " ",
        ).title()
    )

    plt.title(title)

    plt.xticks(
        rotation=25,
        ha="right",
    )

    plt.tight_layout()

    plt.savefig(
        filepath,
        dpi=300,
    )

    plt.close()


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Run all ablation model experiments.
    """

    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Feature Ablation Models")
    print("=" * 72)

    MODEL_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading participant-aware splits..."
    )

    train = prepare_split(
        TRAIN_PATH,
        "train",
    )

    calibration = prepare_split(
        CALIBRATION_PATH,
        "calibration",
    )

    test = prepare_split(
        TEST_PATH,
        "test",
    )

    print(
        f"Train rows: {len(train)}"
    )

    print(
        f"Calibration rows: {len(calibration)}"
    )

    print(
        f"Test rows: {len(test)}"
    )

    print(
        "\nLoading labelled live sessions..."
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

    results = []

    for variant_name, feature_columns in (
        MODEL_VARIANTS.items()
    ):
        result = train_variant(
            variant_name,
            feature_columns,
            train,
            calibration,
            test,
            live,
        )

        results.append(result)

    comparison_rows = []
    test_rows = []
    live_rows = []

    for result in results:
        variant = result["variant"]

        calibration_metrics = (
            result[
                "calibration_metrics"
            ]
        )

        test_metrics = result[
            "test_metrics"
        ]

        live_metrics = result[
            "live_metrics"
        ]

        comparison_rows.append(
            {
                "variant": variant,
                "feature_count": len(
                    result["features"]
                ),
                "selected_threshold": (
                    result[
                        "selected_threshold"
                    ]
                ),
                "calibration_balanced_accuracy": (
                    calibration_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "test_balanced_accuracy": (
                    test_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "test_sensitivity": (
                    test_metrics[
                        "recall_sensitivity"
                    ]
                ),
                "test_specificity": (
                    test_metrics[
                        "specificity"
                    ]
                ),
                "test_roc_auc": (
                    test_metrics[
                        "roc_auc"
                    ]
                ),
                "live_balanced_accuracy": (
                    live_metrics[
                        "balanced_accuracy"
                    ]
                ),
                "live_sensitivity": (
                    live_metrics[
                        "recall_sensitivity"
                    ]
                ),
                "live_specificity": (
                    live_metrics[
                        "specificity"
                    ]
                ),
                "live_false_positive_rate": (
                    live_metrics[
                        "false_positive_rate"
                    ]
                ),
                "model_path": (
                    result[
                        "model_path"
                    ]
                ),
            }
        )

        test_rows.append(
            {
                "variant": variant,
                **test_metrics,
            }
        )

        live_rows.append(
            {
                "variant": variant,
                **live_metrics,
            }
        )

    comparison = pd.DataFrame(
        comparison_rows
    )

    comparison = comparison.sort_values(
        [
            "test_balanced_accuracy",
            "live_balanced_accuracy",
        ],
        ascending=[
            False,
            False,
        ],
    ).reset_index(
        drop=True
    )

    test_metrics_table = pd.DataFrame(
        test_rows
    )

    live_metrics_table = pd.DataFrame(
        live_rows
    )

    comparison.to_csv(
        MODEL_COMPARISON_PATH,
        index=False,
    )

    test_metrics_table.to_csv(
        TEST_METRICS_PATH,
        index=False,
    )

    live_metrics_table.to_csv(
        LIVE_METRICS_PATH,
        index=False,
    )

    save_metric_plot(
        comparison,
        "test_balanced_accuracy",
        TEST_PLOT_PATH,
        (
            "V2 Ablation Models — "
            "Untouched Test Balanced Accuracy"
        ),
    )

    save_metric_plot(
        comparison,
        "live_balanced_accuracy",
        LIVE_PLOT_PATH,
        (
            "V2 Ablation Models — "
            "Live Balanced Accuracy"
        ),
    )

    summary = {
        "project": "DriverGuardianAI V2",
        "experiment": (
            "feature_ablation_models"
        ),
        "live_log_configuration": [
            {
                "path": str(
                    item["path"]
                ),
                "actual_label": (
                    item["actual_label"]
                ),
                "session_name": (
                    item["session_name"]
                ),
            }
            for item in LIVE_LOGS
        ],
        "results": results,
        "comparison": comparison.to_dict(
            orient="records"
        ),
        "important_notes": [
            (
                "Thresholds were selected using calibration.csv only."
            ),
            (
                "Karys and Shayna remained untouched test participants."
            ),
            (
                "Live sessions are deployment-transfer diagnostics, "
                "not an independent participant test."
            ),
            (
                "Verify the two live labels before interpreting live "
                "accuracy."
            ),
        ],
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

    display_columns = [
        "variant",
        "feature_count",
        "selected_threshold",
        "test_balanced_accuracy",
        "test_sensitivity",
        "test_specificity",
        "live_balanced_accuracy",
        "live_sensitivity",
        "live_specificity",
        "live_false_positive_rate",
    ]

    print("\n" + "=" * 72)
    print("Ablation Model Comparison")
    print("=" * 72)

    print(
        comparison[
            display_columns
        ].to_string(
            index=False
        )
    )

    print(
        "\nImportant: confirm that the first live file was Alert "
        "and the second was Fatigue."
    )

    print(
        "\nModels saved to:"
    )

    print(MODEL_DIRECTORY)

    print(
        "\nResults saved to:"
    )

    print(RESULTS_DIRECTORY)


if __name__ == "__main__":
    main()