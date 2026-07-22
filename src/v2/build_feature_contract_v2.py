"""
Build the DriverGuardianAI V2 raw-feature contract.

The V2 model was trained on raw collection features. This script records
that contract, profiles the raw train/calibration/test splits, and compares
legacy V1 live logs when they are available.

Run:
    %run src/v2/build_feature_contract_v2.py
"""

import json
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

TRAIN_PATH = PROJECT_ROOT / "data" / "splits" / "v2" / "train.csv"
CALIBRATION_PATH = PROJECT_ROOT / "data" / "splits" / "v2" / "calibration.csv"
TEST_PATH = PROJECT_ROOT / "data" / "splits" / "v2" / "test.csv"
LIVE_ALERT_PATH = PROJECT_ROOT / "logs" / "hgb_live_alert.csv"
LIVE_FATIGUE_PATH = PROJECT_ROOT / "logs" / "hgb_live_fatigue.csv"

MODEL_DIRECTORY = PROJECT_ROOT / "models" / "v2"
CONTRACT_PATH = MODEL_DIRECTORY / "feature_contract_v2.json"

RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "feature_contract"
REPORT_PATH = RESULTS_DIRECTORY / "feature_contract_report.txt"
TRAINING_STATS_PATH = RESULTS_DIRECTORY / "training_feature_statistics.csv"
SPLIT_STATS_PATH = RESULTS_DIRECTORY / "split_feature_statistics.csv"
LIVE_STATS_PATH = RESULTS_DIRECTORY / "live_feature_statistics.csv"
DRIFT_PATH = RESULTS_DIRECTORY / "live_vs_training_drift.csv"
CONDITION_DISTRIBUTION_PATH = RESULTS_DIRECTORY / "condition_distribution.csv"

FEATURE_ORDER = [
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
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count",
]

CONTRACT_DEFINITIONS = {
    "ear": {
        "dtype": "float",
        "unit": "ratio",
        "runtime_definition": (
            "Mean raw eye-aspect ratio across both eyes. Return the raw EAR. "
            "Do not divide by 0.30, clip to 0-1, or scale manually."
        ),
    },
    "yawn_score": {
        "dtype": "float",
        "unit": "ratio",
        "runtime_definition": (
            "Raw mouth-opening ratio used by the collector. Return the raw "
            "ratio. Do not divide by 0.60 or clip to 0-1."
        ),
    },
    "head_tilt": {
        "dtype": "float",
        "unit": "pixels",
        "runtime_definition": (
            "Absolute vertical displacement between the two eye reference "
            "points in pixel coordinates. Do not convert to degrees."
        ),
    },
    "hands_detected": {
        "dtype": "float",
        "unit": "binary",
        "runtime_definition": (
            "1.0 when a detected wrist satisfies the original lower-frame "
            "rule (wrist y > 0.65 of frame height); otherwise 0.0."
        ),
    },
    "condition": {
        "dtype": "string",
        "unit": "category",
        "allowed_values": ["none", "glasses", "hat", "dark"],
        "runtime_definition": (
            "One of none, glasses, hat, or dark. Dark can be inferred from "
            "brightness; glasses and hat require a user-selected condition "
            "unless separate detectors are added."
        ),
    },
    "low_light": {
        "dtype": "bool",
        "unit": "binary",
        "runtime_definition": (
            "True when mean grayscale brightness is below 50.0; otherwise False."
        ),
    },
    "face_confidence": {
        "dtype": "float",
        "unit": "probability",
        "runtime_definition": (
            "Use the original collection representation. With legacy Face "
            "Mesh, use 1.0 when landmarks are returned."
        ),
    },
    "blink_count": {
        "dtype": "float",
        "unit": "count",
        "runtime_definition": (
            "Cumulative blink count from the start of the recording session. "
            "Do not divide by 30 and do not use a rolling normalised rate."
        ),
    },
}


def require_file(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Required file was not found: {path}")


def to_binary_float(series: pd.Series) -> pd.Series:
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
    values = text.map(mapping)
    values = values.fillna(pd.to_numeric(series, errors="coerce"))
    return values.astype("float64")


def prepare_dataframe(df: pd.DataFrame, name: str) -> pd.DataFrame:
    missing = [feature for feature in FEATURE_ORDER if feature not in df.columns]
    if missing:
        raise ValueError(f"{name} is missing features: {missing}")

    df = df.copy()
    df["hands_detected"] = to_binary_float(df["hands_detected"])
    df["low_light"] = to_binary_float(df["low_light"])

    for feature in [
        "ear",
        "yawn_score",
        "head_tilt",
        "face_confidence",
        "blink_count",
    ]:
        df[feature] = pd.to_numeric(df[feature], errors="coerce").astype("float64")

    df["condition"] = df["condition"].astype(str).str.strip().str.lower()
    return df


def load_splits() -> Dict[str, pd.DataFrame]:
    paths = {
        "train": TRAIN_PATH,
        "calibration": CALIBRATION_PATH,
        "test": TEST_PATH,
    }
    result = {}
    for name, path in paths.items():
        require_file(path)
        result[name] = prepare_dataframe(pd.read_csv(path), name)
    return result


def load_live_logs() -> Dict[str, pd.DataFrame]:
    result = {}
    candidates = {
        "live_alert_v1": LIVE_ALERT_PATH,
        "live_fatigue_v1": LIVE_FATIGUE_PATH,
    }
    for name, path in candidates.items():
        if path.exists():
            result[name] = prepare_dataframe(pd.read_csv(path), name)
    return result


def summarise(df: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce").astype("float64")
        values = values[np.isfinite(values)]
        if values.empty:
            continue
        rows.append(
            {
                "dataset": dataset_name,
                "feature": feature,
                "samples": int(len(values)),
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
                "minimum": float(values.min()),
                "p01": float(values.quantile(0.01)),
                "p05": float(values.quantile(0.05)),
                "median": float(values.median()),
                "p95": float(values.quantile(0.95)),
                "p99": float(values.quantile(0.99)),
                "maximum": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def condition_distribution(datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for dataset_name, df in datasets.items():
        counts = df["condition"].value_counts(dropna=False)
        total = len(df)
        for condition, count in counts.items():
            rows.append(
                {
                    "dataset": dataset_name,
                    "condition": str(condition),
                    "samples": int(count),
                    "percentage": float(count / total),
                }
            )
    return pd.DataFrame(rows)


def standardised_mean_difference(reference: pd.Series, comparison: pd.Series) -> float:
    reference = pd.to_numeric(reference, errors="coerce").astype("float64").dropna()
    comparison = pd.to_numeric(comparison, errors="coerce").astype("float64").dropna()
    if len(reference) < 2 or comparison.empty:
        return float("nan")

    reference_std = float(reference.std(ddof=1))
    difference = float(comparison.mean() - reference.mean())

    if reference_std <= 1e-12:
        if abs(difference) <= 1e-12:
            return 0.0
        return float(np.sign(difference) * np.inf)

    return difference / reference_std


def build_drift_report(training: pd.DataFrame, live_logs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for live_name, live_df in live_logs.items():
        for feature in NUMERIC_FEATURES:
            train_values = pd.to_numeric(training[feature], errors="coerce").astype("float64").dropna()
            live_values = pd.to_numeric(live_df[feature], errors="coerce").astype("float64").dropna()
            if train_values.empty or live_values.empty:
                continue

            p05 = float(train_values.quantile(0.05))
            p95 = float(train_values.quantile(0.95))
            outside_rate = float(((live_values < p05) | (live_values > p95)).mean())
            smd = standardised_mean_difference(train_values, live_values)

            rows.append(
                {
                    "live_dataset": live_name,
                    "feature": feature,
                    "training_mean": float(train_values.mean()),
                    "live_mean": float(live_values.mean()),
                    "training_p05": p05,
                    "training_p95": p95,
                    "live_outside_training_p05_p95": outside_rate,
                    "standardised_mean_difference": float(smd),
                    "absolute_smd": float(abs(smd)),
                }
            )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values(
        ["live_dataset", "absolute_smd"],
        ascending=[True, False],
    )


def build_contract(reference: pd.DataFrame, stats: pd.DataFrame) -> Dict:
    feature_contract = {}

    for feature in FEATURE_ORDER:
        definition = dict(CONTRACT_DEFINITIONS[feature])
        if feature in NUMERIC_FEATURES:
            row = stats.loc[stats["feature"] == feature]
            if row.empty:
                raise RuntimeError(f"Statistics missing for {feature}")
            record = row.iloc[0].to_dict()
            definition["reference_distribution"] = {
                key: (int(value) if key == "samples" else float(value))
                for key, value in record.items()
                if key not in {"dataset", "feature"}
            }
        feature_contract[feature] = definition

    return {
        "project": "DriverGuardianAI",
        "project_version": "v2",
        "contract_version": "1.0",
        "feature_order": FEATURE_ORDER,
        "feature_contract": feature_contract,
        "condition_values_observed": sorted(
            reference["condition"].dropna().astype(str).unique().tolist()
        ),
        "runtime_rules": {
            "apply_scaling_before_model": False,
            "apply_clipping_before_model": False,
            "apply_manual_normalisation_before_model": False,
            "model_bundle_contains_preprocessor": True,
            "reset_blink_count_at_session_start": True,
            "low_light_brightness_threshold": 50.0,
            "hand_wrist_lower_frame_ratio": 0.65,
        },
        "important_notes": [
            "The V2 model expects raw feature values.",
            "Do not reuse EAR/0.30, yawn/0.60, head-tilt/45, or blink/30.",
            "Reference statistics are diagnostic ranges, not clipping limits.",
        ],
    }


def write_report(contract: Dict, split_stats: pd.DataFrame, drift: pd.DataFrame) -> None:
    lines = [
        "DriverGuardianAI V2",
        "Raw Feature Contract",
        "=" * 72,
        "",
        "The V2 model expects raw features. No manual scaling is applied.",
        "",
    ]

    for feature in FEATURE_ORDER:
        definition = contract["feature_contract"][feature]
        lines.extend(
            [
                feature,
                "-" * len(feature),
                f"Unit: {definition['unit']}",
                f"Definition: {definition['runtime_definition']}",
                "",
            ]
        )

    lines.extend(
        [
            "Split statistics",
            "-" * 72,
            split_stats.to_string(index=False),
            "",
        ]
    )

    if not drift.empty:
        lines.extend(
            [
                "Legacy V1 live-log mismatch",
                "-" * 72,
                drift.to_string(index=False),
                "",
                "The old logs contain V1-normalised values and are not valid V2 raw logs.",
            ]
        )

    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Raw Feature Contract")
    print("=" * 72)

    MODEL_DIRECTORY.mkdir(parents=True, exist_ok=True)
    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)

    splits = load_splits()
    reference = pd.concat(list(splits.values()), ignore_index=True)
    print(f"\nReference samples: {len(reference)}")

    reference_stats = summarise(reference, "all_v2_raw_data")
    reference_stats.to_csv(TRAINING_STATS_PATH, index=False)

    split_stats = pd.concat(
        [summarise(df, name) for name, df in splits.items()],
        ignore_index=True,
    )
    split_stats.to_csv(SPLIT_STATS_PATH, index=False)

    live_logs = load_live_logs()
    if live_logs:
        live_stats = pd.concat(
            [summarise(df, name) for name, df in live_logs.items()],
            ignore_index=True,
        )
    else:
        live_stats = pd.DataFrame()
    live_stats.to_csv(LIVE_STATS_PATH, index=False)

    drift = build_drift_report(reference, live_logs)
    drift.to_csv(DRIFT_PATH, index=False)

    condition_distribution({**splits, **live_logs}).to_csv(
        CONDITION_DISTRIBUTION_PATH,
        index=False,
    )

    contract = build_contract(reference, reference_stats)
    CONTRACT_PATH.write_text(json.dumps(contract, indent=4), encoding="utf-8")
    write_report(contract, split_stats, drift)

    print("\nRaw reference feature statistics:")
    print(reference_stats.to_string(index=False))

    if not drift.empty:
        print("\nLegacy live-log mismatch:")
        print(drift.to_string(index=False))

    print("\n" + "=" * 72)
    print("Feature contract created successfully.")
    print("=" * 72)
    print(f"\nContract saved to:\n{CONTRACT_PATH}")
    print(f"\nReports saved to:\n{RESULTS_DIRECTORY}")
    print("\nNext step: build vision_agent_v2.py to emit these raw values.")


if __name__ == "__main__":
    main()