"""
DriverGuardianAI V2 - Full Feature Reliance Diagnostic.

Measures how strongly each model feature influences predictions on the
latest two V2 live logs using training-reference replacement and
permutation tests.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = PROJECT_ROOT / "models" / "v2" / "driver_guardian_hgb_v2.joblib"
TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "v2" / "train.csv"
LOG_DIRECTORY = PROJECT_ROOT / "logs" / "v2"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "all_feature_reliance"
SUMMARY_PATH = RESULTS_DIRECTORY / "feature_reliance_summary.json"
REFERENCE_VALUES_PATH = RESULTS_DIRECTORY / "feature_reference_values.csv"
RANKING_PATH = RESULTS_DIRECTORY / "feature_reliance_ranking.csv"
SESSION_RANKING_PATH = RESULTS_DIRECTORY / "session_feature_reliance.csv"
ROW_LEVEL_PATH = RESULTS_DIRECTORY / "row_level_feature_reliance.csv"
RANKING_PLOT_PATH = RESULTS_DIRECTORY / "feature_reliance_ranking.png"
PREDICTION_RATE_PLOT_PATH = RESULTS_DIRECTORY / "prediction_rate_by_feature.png"

MANUAL_LOG_PATHS: List[Path] = []
RANDOM_STATE = 42
PROBABILITY_CHANGE_THRESHOLD = 0.10

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
    "face_confidence",
    "blink_count",
]

BINARY_FEATURES = [
    "hands_detected",
    "low_light",
]

CATEGORICAL_FEATURES = [
    "condition",
]


def require_file(filepath: Path) -> None:
    if not filepath.exists():
        raise FileNotFoundError(f"Required file was not found: {filepath}")


def discover_log_paths() -> List[Path]:
    if MANUAL_LOG_PATHS:
        paths = [Path(path) for path in MANUAL_LOG_PATHS]
        for path in paths:
            require_file(path)
        return paths

    if not LOG_DIRECTORY.exists():
        raise FileNotFoundError(f"V2 log directory was not found: {LOG_DIRECTORY}")

    candidates = sorted(
        LOG_DIRECTORY.glob("driver_guardian_v2_session_*.csv"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )

    if len(candidates) < 2:
        raise FileNotFoundError(
            f"At least two V2 logs are required; found {len(candidates)}."
        )

    return list(reversed(candidates[:2]))


def convert_boolean_series(series: pd.Series, column_name: str) -> pd.Series:
    text = series.astype(str).str.strip().str.lower()
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
    converted = converted.fillna(pd.to_numeric(series, errors="coerce"))
    invalid = converted.isna() & series.notna()
    if invalid.any():
        examples = series.loc[invalid].astype(str).unique().tolist()
        raise ValueError(
            f"{column_name} contains unsupported values: {examples[:10]}"
        )
    return converted.astype("float64")


def prepare_dataframe(dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    missing = [feature for feature in FEATURE_COLUMNS if feature not in dataframe]
    if missing:
        raise ValueError(f"{dataset_name} is missing features: {missing}")
    if dataframe.empty:
        raise ValueError(f"{dataset_name} contains no rows.")

    dataframe = dataframe.copy()
    dataframe["hands_detected"] = convert_boolean_series(
        dataframe["hands_detected"], "hands_detected"
    )
    dataframe["low_light"] = convert_boolean_series(
        dataframe["low_light"], "low_light"
    )

    for feature in NUMERIC_FEATURES:
        dataframe[feature] = pd.to_numeric(
            dataframe[feature], errors="coerce"
        ).astype("float64")

    dataframe["condition"] = (
        dataframe["condition"].astype(str).str.strip().str.lower()
    )
    return dataframe


def load_training_data() -> pd.DataFrame:
    require_file(TRAIN_PATH)
    return prepare_dataframe(pd.read_csv(TRAIN_PATH), "training split")


def load_live_logs() -> Tuple[pd.DataFrame, List[Path]]:
    paths = discover_log_paths()
    frames = []

    for index, filepath in enumerate(paths, start=1):
        dataframe = prepare_dataframe(pd.read_csv(filepath), filepath.name)
        dataframe["session_name"] = f"session_{index}_{filepath.stem}"
        dataframe["source_log"] = str(filepath)
        dataframe["source_row"] = np.arange(len(dataframe), dtype=int)
        frames.append(dataframe)

    return pd.concat(frames, ignore_index=True), paths


def load_model_bundle() -> Dict[str, Any]:
    require_file(MODEL_PATH)
    bundle = joblib.load(MODEL_PATH)
    required = {"pipeline", "fatigue_threshold", "feature_columns"}
    missing = required.difference(bundle.keys())
    if missing:
        raise KeyError(f"Model bundle is missing keys: {sorted(missing)}")
    if list(bundle["feature_columns"]) != FEATURE_COLUMNS:
        raise ValueError(
            "Model feature order does not match this script.\n"
            f"Model: {bundle['feature_columns']}\nExpected: {FEATURE_COLUMNS}"
        )
    return bundle


def calculate_reference_values(training: pd.DataFrame) -> Dict[str, Any]:
    references: Dict[str, Any] = {}

    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(training[feature], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No valid training values for {feature}.")
        references[feature] = float(values.median())

    for feature in BINARY_FEATURES:
        values = pd.to_numeric(training[feature], errors="coerce").dropna()
        if values.empty:
            raise ValueError(f"No valid training values for {feature}.")
        references[feature] = float(values.mode().iloc[0])

    for feature in CATEGORICAL_FEATURES:
        values = training[feature].dropna().astype(str)
        if values.empty:
            raise ValueError(f"No valid training values for {feature}.")
        references[feature] = str(values.mode().iloc[0])

    return references


def create_reference_report(
    training: pd.DataFrame,
    references: Dict[str, Any],
) -> pd.DataFrame:
    rows = []

    for feature in FEATURE_COLUMNS:
        if feature in CATEGORICAL_FEATURES:
            rows.append(
                {
                    "feature": feature,
                    "reference_type": "mode",
                    "reference_value": references[feature],
                    "training_mean": None,
                    "training_std": None,
                    "training_minimum": None,
                    "training_maximum": None,
                    "training_values": " | ".join(
                        sorted(training[feature].dropna().astype(str).unique())
                    ),
                }
            )
        else:
            values = pd.to_numeric(training[feature], errors="coerce").dropna()
            rows.append(
                {
                    "feature": feature,
                    "reference_type": (
                        "mode" if feature in BINARY_FEATURES else "median"
                    ),
                    "reference_value": references[feature],
                    "training_mean": float(values.mean()),
                    "training_std": float(values.std(ddof=1)),
                    "training_minimum": float(values.min()),
                    "training_maximum": float(values.max()),
                    "training_values": None,
                }
            )

    return pd.DataFrame(rows)


def predict_probabilities(pipeline, dataframe: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(dataframe[FEATURE_COLUMNS])[:, 1]


def calculate_effect_metrics(
    original_probabilities: np.ndarray,
    modified_probabilities: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    original_predictions = (original_probabilities >= threshold).astype(int)
    modified_predictions = (modified_probabilities >= threshold).astype(int)
    shifts = modified_probabilities - original_probabilities

    return {
        "mean_original_probability": float(original_probabilities.mean()),
        "mean_modified_probability": float(modified_probabilities.mean()),
        "mean_signed_probability_shift": float(shifts.mean()),
        "mean_absolute_probability_shift": float(np.abs(shifts).mean()),
        "median_absolute_probability_shift": float(np.median(np.abs(shifts))),
        "maximum_absolute_probability_shift": float(np.abs(shifts).max()),
        "rows_changed_by_at_least_10_percent": int(
            (np.abs(shifts) >= PROBABILITY_CHANGE_THRESHOLD).sum()
        ),
        "rows_whose_class_changed": int(
            (original_predictions != modified_predictions).sum()
        ),
        "original_fatigue_rate": float(original_predictions.mean()),
        "modified_fatigue_rate": float(modified_predictions.mean()),
    }


def evaluate_all_features(
    live_logs: pd.DataFrame,
    pipeline,
    fatigue_threshold: float,
    references: Dict[str, Any],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(RANDOM_STATE)
    original_features = live_logs[FEATURE_COLUMNS].copy()
    original_probabilities = predict_probabilities(pipeline, original_features)

    row_level = live_logs[
        ["session_name", "source_log", "source_row", *FEATURE_COLUMNS]
    ].copy()
    row_level["original_fatigue_probability"] = original_probabilities
    row_level["original_prediction"] = (
        original_probabilities >= fatigue_threshold
    ).astype(int)

    ranking_rows = []
    session_rows = []

    for feature in FEATURE_COLUMNS:
        replacement_features = original_features.copy()
        replacement_features[feature] = references[feature]
        replacement_probabilities = predict_probabilities(
            pipeline, replacement_features
        )
        replacement_metrics = calculate_effect_metrics(
            original_probabilities,
            replacement_probabilities,
            fatigue_threshold,
        )
        ranking_rows.append(
            {
                "feature": feature,
                "method": "replace_with_training_reference",
                "reference_value": references[feature],
                "samples": int(len(live_logs)),
                **replacement_metrics,
            }
        )
        row_level[f"{feature}_reference_probability"] = replacement_probabilities
        row_level[f"{feature}_reference_shift"] = (
            replacement_probabilities - original_probabilities
        )

        permutation_features = original_features.copy()
        permutation_values = permutation_features[feature].to_numpy(copy=True)
        rng.shuffle(permutation_values)
        permutation_features[feature] = permutation_values
        permutation_probabilities = predict_probabilities(
            pipeline, permutation_features
        )
        permutation_metrics = calculate_effect_metrics(
            original_probabilities,
            permutation_probabilities,
            fatigue_threshold,
        )
        ranking_rows.append(
            {
                "feature": feature,
                "method": "permutation",
                "reference_value": None,
                "samples": int(len(live_logs)),
                **permutation_metrics,
            }
        )
        row_level[f"{feature}_permutation_probability"] = permutation_probabilities
        row_level[f"{feature}_permutation_shift"] = (
            permutation_probabilities - original_probabilities
        )

        temp = pd.DataFrame(
            {
                "session_name": live_logs["session_name"].values,
                "original_probability": original_probabilities,
                "replacement_probability": replacement_probabilities,
                "permutation_probability": permutation_probabilities,
            }
        )

        for session_name, group in temp.groupby("session_name", sort=True):
            session_rows.append(
                {
                    "session_name": session_name,
                    "feature": feature,
                    "method": "replace_with_training_reference",
                    "reference_value": references[feature],
                    "samples": int(len(group)),
                    **calculate_effect_metrics(
                        group["original_probability"].to_numpy(float),
                        group["replacement_probability"].to_numpy(float),
                        fatigue_threshold,
                    ),
                }
            )
            session_rows.append(
                {
                    "session_name": session_name,
                    "feature": feature,
                    "method": "permutation",
                    "reference_value": None,
                    "samples": int(len(group)),
                    **calculate_effect_metrics(
                        group["original_probability"].to_numpy(float),
                        group["permutation_probability"].to_numpy(float),
                        fatigue_threshold,
                    ),
                }
            )

    ranking = pd.DataFrame(ranking_rows).sort_values(
        ["mean_absolute_probability_shift", "rows_whose_class_changed"],
        ascending=[False, False],
    ).reset_index(drop=True)

    return ranking, pd.DataFrame(session_rows), row_level


def build_findings(ranking: pd.DataFrame) -> List[str]:
    findings = []
    replacement = ranking[
        ranking["method"] == "replace_with_training_reference"
    ].sort_values("mean_absolute_probability_shift", ascending=False)
    permutation = ranking[
        ranking["method"] == "permutation"
    ].sort_values("mean_absolute_probability_shift", ascending=False)

    strongest_replacement = replacement.iloc[0]
    strongest_permutation = permutation.iloc[0]

    findings.append(
        "The strongest training-reference replacement effect was "
        f"{strongest_replacement['feature']}, with an average absolute "
        "Fatigue-probability change of "
        f"{strongest_replacement['mean_absolute_probability_shift']:.1%}."
    )
    findings.append(
        "The strongest permutation effect was "
        f"{strongest_permutation['feature']}, with an average absolute "
        "Fatigue-probability change of "
        f"{strongest_permutation['mean_absolute_probability_shift']:.1%}."
    )

    for _, row in replacement.head(4).iterrows():
        reference_type = (
            "mode"
            if row["feature"] in BINARY_FEATURES + CATEGORICAL_FEATURES
            else "median"
        )
        findings.append(
            f"Replacing {row['feature']} with its training {reference_type} "
            f"changed {int(row['rows_whose_class_changed'])} classifications "
            "and shifted probability by "
            f"{row['mean_absolute_probability_shift']:.1%} on average."
        )

    hands_row = replacement[replacement["feature"] == "hands_detected"].iloc[0]
    if (
        hands_row["mean_absolute_probability_shift"] >= 0.20
        or hands_row["rows_whose_class_changed"]
        >= 0.20 * hands_row["samples"]
    ):
        findings.append(
            "hands_detected remains a strong shortcut candidate. A retrained "
            "no-hands model should be evaluated next."
        )

    return findings


def save_reliance_ranking_plot(ranking: pd.DataFrame) -> None:
    plot_data = ranking[
        ranking["method"] == "replace_with_training_reference"
    ].sort_values("mean_absolute_probability_shift", ascending=False)

    plt.figure(figsize=(11, 6))
    plt.bar(
        plot_data["feature"],
        plot_data["mean_absolute_probability_shift"],
    )
    plt.xlabel("Feature")
    plt.ylabel("Mean absolute Fatigue-probability change")
    plt.title("DriverGuardianAI V2 Full Feature Reliance")
    plt.xticks(rotation=35, ha="right")
    plt.tight_layout()
    plt.savefig(RANKING_PLOT_PATH, dpi=300)
    plt.close()


def save_prediction_rate_plot(ranking: pd.DataFrame) -> None:
    plot_data = ranking[
        ranking["method"] == "replace_with_training_reference"
    ].sort_values("modified_fatigue_rate", ascending=False)

    positions = np.arange(len(plot_data))
    width = 0.38
    plt.figure(figsize=(12, 6))
    plt.bar(
        positions - width / 2,
        plot_data["original_fatigue_rate"],
        width=width,
        label="Original",
    )
    plt.bar(
        positions + width / 2,
        plot_data["modified_fatigue_rate"],
        width=width,
        label="Feature replaced",
    )
    plt.xticks(positions, plot_data["feature"], rotation=35, ha="right")
    plt.ylim(0.0, 1.05)
    plt.ylabel("Predicted Fatigue rate")
    plt.xlabel("Replaced feature")
    plt.title("Prediction Rate After Training-Reference Replacement")
    plt.legend()
    plt.tight_layout()
    plt.savefig(PREDICTION_RATE_PLOT_PATH, dpi=300)
    plt.close()


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Full Feature Reliance Diagnostic")
    print("=" * 72)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    print("\nLoading model bundle...")
    bundle = load_model_bundle()
    pipeline = bundle["pipeline"]
    fatigue_threshold = float(bundle["fatigue_threshold"])
    print(f"Fatigue threshold: {fatigue_threshold:.2f}")

    print("\nLoading training split...")
    training = load_training_data()
    print(f"Training rows: {len(training)}")

    references = calculate_reference_values(training)
    reference_report = create_reference_report(training, references)
    reference_report.to_csv(REFERENCE_VALUES_PATH, index=False)

    print("\nTraining reference values:")
    print(
        reference_report[
            ["feature", "reference_type", "reference_value"]
        ].to_string(index=False)
    )

    print("\nLoading V2 live logs...")
    live_logs, log_paths = load_live_logs()
    for filepath in log_paths:
        print(f"- {filepath}")
    print(f"Combined live rows: {len(live_logs)}")

    print("\nEvaluating all feature replacements and permutations...")
    ranking, session_ranking, row_level = evaluate_all_features(
        live_logs,
        pipeline,
        fatigue_threshold,
        references,
    )

    findings = build_findings(ranking)

    ranking.to_csv(RANKING_PATH, index=False)
    session_ranking.to_csv(SESSION_RANKING_PATH, index=False)
    row_level.to_csv(ROW_LEVEL_PATH, index=False)
    save_reliance_ranking_plot(ranking)
    save_prediction_rate_plot(ranking)

    summary = {
        "project": "DriverGuardianAI V2",
        "model_path": str(MODEL_PATH),
        "training_path": str(TRAIN_PATH),
        "live_log_paths": [str(path) for path in log_paths],
        "fatigue_threshold": fatigue_threshold,
        "training_rows": int(len(training)),
        "live_rows": int(len(live_logs)),
        "reference_values": references,
        "ranking": ranking.to_dict(orient="records"),
        "findings": findings,
        "important_note": (
            "Feature replacement and permutation diagnose reliance of the "
            "existing model. They do not estimate the final performance of "
            "a model retrained without a feature."
        ),
    }

    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    display_columns = [
        "feature",
        "method",
        "reference_value",
        "mean_absolute_probability_shift",
        "rows_changed_by_at_least_10_percent",
        "rows_whose_class_changed",
        "modified_fatigue_rate",
    ]

    print("\nFeature reliance ranking:")
    print(ranking[display_columns].to_string(index=False))

    print("\nMain findings:")
    for finding in findings:
        print(f"- {finding}")

    print("\n" + "=" * 72)
    print("Full feature reliance diagnostic completed.")
    print("=" * 72)
    print("\nResults saved to:")
    print(RESULTS_DIRECTORY)


if __name__ == "__main__":
    main()