"""
Clean the DriverGuardianAI V2 raw master dataset.

Input
-----
data/processed/driver_guardian_raw_merged.csv

Output
------
data/processed/driver_guardian_v2_clean.csv

Reports
-------
results/v2/dataset_cleaning/
    cleaning_summary.json
    removed_rows.csv
    suspicious_rows.csv
    feature_summary_before.csv
    feature_summary_after.csv
    participant_summary.csv
    session_summary.csv
    class_distribution_before.csv
    class_distribution_after.csv
    duplicate_summary.json

This script does not scale, normalise, encode, split, balance, or train.
"""

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "driver_guardian_raw_merged.csv"
OUTPUT_DATASET_PATH = PROJECT_ROOT / "data" / "processed" / "driver_guardian_v2_clean.csv"
RESULTS_DIRECTORY = PROJECT_ROOT / "results" / "v2" / "dataset_cleaning"

TARGET_COLUMN = "fatigue_level"

REQUIRED_COLUMNS = [
    "timestamp", "participant_id", "session_id", "source_file", "source_path",
    "source_row", "ear", "yawn_score", "head_tilt", "hands_detected",
    "state", "condition", "low_light", "fatigue_score", "face_confidence",
    "fatigue_level", "blink_count",
]

MODEL_FEATURE_COLUMNS = [
    "ear", "yawn_score", "head_tilt", "hands_detected", "condition",
    "low_light", "face_confidence", "blink_count",
]

NUMERIC_COLUMNS = [
    "ear", "yawn_score", "head_tilt", "fatigue_score",
    "face_confidence", "blink_count",
]

BOOLEAN_COLUMNS = ["hands_detected", "low_light"]
TEXT_COLUMNS = [
    "participant_id", "session_id", "source_file", "source_path",
    "state", "condition", "fatigue_level",
]

ALLOWED_TARGETS = {"Alert", "Mild Fatigue", "Moderate Fatigue", "Severe Fatigue"}
ALLOWED_STATES = {"normal", "drowsy"}
ALLOWED_CONDITIONS = {"none", "glasses", "hat", "dark"}

HARD_VALID_RANGES = {
    "ear": (0.0, 1.0),
    "yawn_score": (0.0, 2.0),
    "head_tilt": (0.0, 500.0),
    "fatigue_score": (-10.0, 20.0),
    "face_confidence": (0.0, 1.0),
    "blink_count": (0.0, 10000.0),
}

SOFT_EXPECTED_RANGES = {
    "ear": (0.10, 0.50),
    "yawn_score": (0.0, 1.0),
    "head_tilt": (0.0, 50.0),
    "face_confidence": (0.50, 1.0),
    "blink_count": (0.0, 500.0),
}


def load_dataset() -> pd.DataFrame:
    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(f"Merged raw dataset was not found: {INPUT_DATASET_PATH}")
    dataframe = pd.read_csv(INPUT_DATASET_PATH)
    missing = [column for column in REQUIRED_COLUMNS if column not in dataframe.columns]
    if missing:
        raise ValueError(f"Merged raw dataset is missing columns: {missing}")
    if dataframe.empty:
        raise ValueError("Merged raw dataset contains no rows.")
    return dataframe


def prepare_types(dataframe: pd.DataFrame) -> pd.DataFrame:
    dataframe = dataframe.copy()
    dataframe["timestamp"] = pd.to_datetime(dataframe["timestamp"], errors="coerce")
    dataframe["source_row"] = pd.to_numeric(dataframe["source_row"], errors="coerce")
    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(dataframe[column], errors="coerce")
    boolean_map = {
        "true": True, "false": False, "1": True, "0": False,
        "1.0": True, "0.0": False,
    }
    for column in BOOLEAN_COLUMNS:
        dataframe[column] = (
            dataframe[column].astype(str).str.strip().str.lower().map(boolean_map)
        )
    for column in TEXT_COLUMNS:
        dataframe[column] = dataframe[column].astype("string").str.strip()
    dataframe["participant_id"] = dataframe["participant_id"].str.lower()
    dataframe["session_id"] = dataframe["session_id"].str.lower()
    dataframe["state"] = dataframe["state"].str.lower()
    dataframe["condition"] = dataframe["condition"].str.lower()
    return dataframe


def class_distribution(dataframe: pd.DataFrame) -> pd.DataFrame:
    result = (
        dataframe[TARGET_COLUMN].value_counts(dropna=False)
        .rename_axis(TARGET_COLUMN).reset_index(name="samples")
    )
    result["percentage"] = result["samples"] / result["samples"].sum()
    return result


def feature_summary(dataframe: pd.DataFrame, dataset_name: str) -> pd.DataFrame:
    rows = []
    for feature in NUMERIC_COLUMNS:
        values = pd.to_numeric(dataframe[feature], errors="coerce").dropna()
        if values.empty:
            continue
        rows.append({
            "dataset": dataset_name,
            "feature": feature,
            "samples": int(len(values)),
            "missing": int(dataframe[feature].isna().sum()),
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "minimum": float(values.min()),
            "p01": float(values.quantile(0.01)),
            "p05": float(values.quantile(0.05)),
            "median": float(values.median()),
            "p95": float(values.quantile(0.95)),
            "p99": float(values.quantile(0.99)),
            "maximum": float(values.max()),
        })
    return pd.DataFrame(rows)


def participant_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for participant, group in dataframe.groupby("participant_id", sort=True):
        counts = group[TARGET_COLUMN].value_counts().to_dict()
        rows.append({
            "participant_id": participant,
            "samples": int(len(group)),
            "sessions": int(group["session_id"].nunique()),
            "alert_samples": int(counts.get("Alert", 0)),
            "mild_samples": int(counts.get("Mild Fatigue", 0)),
            "moderate_samples": int(counts.get("Moderate Fatigue", 0)),
            "conditions": " | ".join(sorted(group["condition"].dropna().astype(str).unique())),
            "states": " | ".join(sorted(group["state"].dropna().astype(str).unique())),
        })
    return pd.DataFrame(rows)


def session_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for session_id, group in dataframe.groupby("session_id", sort=True):
        timestamps = group["timestamp"].dropna().sort_values()
        duration_seconds = (
            float((timestamps.iloc[-1] - timestamps.iloc[0]).total_seconds())
            if len(timestamps) >= 2 else 0.0
        )
        counts = group[TARGET_COLUMN].value_counts().to_dict()
        rows.append({
            "session_id": session_id,
            "participant_id": str(group["participant_id"].iloc[0]),
            "samples": int(len(group)),
            "duration_seconds": duration_seconds,
            "condition": " | ".join(sorted(group["condition"].dropna().astype(str).unique())),
            "state": " | ".join(sorted(group["state"].dropna().astype(str).unique())),
            "alert_samples": int(counts.get("Alert", 0)),
            "mild_samples": int(counts.get("Mild Fatigue", 0)),
            "moderate_samples": int(counts.get("Moderate Fatigue", 0)),
            "ear_mean": float(group["ear"].mean()),
            "yawn_score_mean": float(group["yawn_score"].mean()),
            "head_tilt_mean": float(group["head_tilt"].mean()),
            "blink_count_mean": float(group["blink_count"].mean()),
        })
    return pd.DataFrame(rows)


def add_reason(reasons: pd.Series, mask: pd.Series, text: str) -> pd.Series:
    current = reasons.loc[mask]
    reasons.loc[mask] = np.where(current == "", text, current + " | " + text)
    return reasons


def identify_invalid_rows(dataframe: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    reasons = pd.Series("", index=dataframe.index, dtype="string")
    critical_columns = [
        "timestamp", "participant_id", "session_id", "ear", "yawn_score",
        "head_tilt", "hands_detected", "condition", "low_light",
        "face_confidence", "fatigue_level", "blink_count",
    ]
    for column in critical_columns:
        reasons = add_reason(reasons, dataframe[column].isna(), f"missing_{column}")
    reasons = add_reason(
        reasons,
        ~dataframe[TARGET_COLUMN].isin(ALLOWED_TARGETS),
        "invalid_fatigue_level",
    )
    reasons = add_reason(reasons, ~dataframe["state"].isin(ALLOWED_STATES), "invalid_state")
    reasons = add_reason(
        reasons,
        ~dataframe["condition"].isin(ALLOWED_CONDITIONS),
        "invalid_condition",
    )
    for column, (minimum, maximum) in HARD_VALID_RANGES.items():
        values = dataframe[column]
        mask = values.notna() & ((values < minimum) | (values > maximum))
        reasons = add_reason(reasons, mask, f"{column}_outside_hard_range")
    invalid_source_row = dataframe["source_row"].isna() | (dataframe["source_row"] < 0)
    reasons = add_reason(reasons, invalid_source_row, "invalid_source_row")
    return reasons != "", reasons


def identify_suspicious_rows(dataframe: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
    reasons = pd.Series("", index=dataframe.index, dtype="string")
    for column, (minimum, maximum) in SOFT_EXPECTED_RANGES.items():
        values = dataframe[column]
        mask = values.notna() & ((values < minimum) | (values > maximum))
        reasons = add_reason(reasons, mask, f"{column}_outside_soft_range")
    reasons = add_reason(
        reasons,
        dataframe["state"].eq("normal")
        & dataframe[TARGET_COLUMN].isin(["Moderate Fatigue", "Severe Fatigue"]),
        "normal_session_with_high_fatigue_label",
    )
    reasons = add_reason(
        reasons,
        dataframe["state"].eq("drowsy")
        & dataframe[TARGET_COLUMN].eq("Alert"),
        "drowsy_session_with_alert_label",
    )
    return reasons != "", reasons


def duplicate_summary(dataframe: pd.DataFrame) -> Dict:
    original_columns = [
        "timestamp", "ear", "yawn_score", "head_tilt", "hands_detected",
        "state", "condition", "low_light", "fatigue_score",
        "face_confidence", "fatigue_level", "blink_count",
    ]
    return {
        "rows": int(len(dataframe)),
        "exact_duplicates_all_columns": int(dataframe.duplicated().sum()),
        "duplicate_original_rows": int(dataframe[original_columns].duplicated().sum()),
        "repeated_feature_vectors": int(dataframe[MODEL_FEATURE_COLUMNS].duplicated().sum()),
        "repeated_feature_vectors_with_label": int(
            dataframe[[*MODEL_FEATURE_COLUMNS, TARGET_COLUMN]].duplicated().sum()
        ),
        "important_note": (
            "Repeated feature vectors are retained because adjacent video frames may "
            "legitimately contain identical values. No duplicate rows are removed automatically."
        ),
    }


def main() -> None:
    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Dataset Cleaning")
    print("=" * 72)

    RESULTS_DIRECTORY.mkdir(parents=True, exist_ok=True)
    OUTPUT_DATASET_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("\nLoading merged raw dataset...")
    dataframe = prepare_types(load_dataset())
    original_rows = len(dataframe)
    print(f"Rows loaded: {original_rows}")

    before_distribution = class_distribution(dataframe)
    before_features = feature_summary(dataframe, "before_cleaning")
    duplicates = duplicate_summary(dataframe)

    invalid_mask, invalid_reasons = identify_invalid_rows(dataframe)
    removed_rows = dataframe.loc[invalid_mask].copy()
    if not removed_rows.empty:
        removed_rows["removal_reason"] = invalid_reasons.loc[invalid_mask].values
    cleaned = dataframe.loc[~invalid_mask].copy()

    severe_rows = int(cleaned[TARGET_COLUMN].eq("Severe Fatigue").sum())
    cleaned[TARGET_COLUMN] = cleaned[TARGET_COLUMN].replace(
        {"Severe Fatigue": "Moderate Fatigue"}
    )

    suspicious_mask, suspicious_reasons = identify_suspicious_rows(cleaned)
    suspicious_rows = cleaned.loc[suspicious_mask].copy()
    if not suspicious_rows.empty:
        suspicious_rows["suspicion_reason"] = suspicious_reasons.loc[suspicious_mask].values

    cleaned = cleaned.sort_values(
        ["participant_id", "session_id", "timestamp", "source_row"],
        kind="stable",
    ).reset_index(drop=True)

    cleaned.to_csv(OUTPUT_DATASET_PATH, index=False)
    removed_rows.to_csv(RESULTS_DIRECTORY / "removed_rows.csv", index=False)
    suspicious_rows.to_csv(RESULTS_DIRECTORY / "suspicious_rows.csv", index=False)

    after_distribution = class_distribution(cleaned)
    after_features = feature_summary(cleaned, "after_cleaning")
    participants = participant_summary(cleaned)
    sessions = session_summary(cleaned)

    before_distribution.to_csv(
        RESULTS_DIRECTORY / "class_distribution_before.csv", index=False
    )
    after_distribution.to_csv(
        RESULTS_DIRECTORY / "class_distribution_after.csv", index=False
    )
    before_features.to_csv(
        RESULTS_DIRECTORY / "feature_summary_before.csv", index=False
    )
    after_features.to_csv(
        RESULTS_DIRECTORY / "feature_summary_after.csv", index=False
    )
    participants.to_csv(RESULTS_DIRECTORY / "participant_summary.csv", index=False)
    sessions.to_csv(RESULTS_DIRECTORY / "session_summary.csv", index=False)

    with (RESULTS_DIRECTORY / "duplicate_summary.json").open("w", encoding="utf-8") as file:
        json.dump(duplicates, file, indent=4)

    summary = {
        "project": "DriverGuardianAI V2",
        "input_dataset": str(INPUT_DATASET_PATH),
        "output_dataset": str(OUTPUT_DATASET_PATH),
        "rows_before_cleaning": int(original_rows),
        "rows_removed": int(len(removed_rows)),
        "rows_after_cleaning": int(len(cleaned)),
        "rows_retained_percentage": float(len(cleaned) / original_rows),
        "severe_rows_merged_into_moderate": severe_rows,
        "suspicious_rows_retained": int(len(suspicious_rows)),
        "participants": int(cleaned["participant_id"].nunique()),
        "sessions": int(cleaned["session_id"].nunique()),
        "class_distribution_after": {
            str(row[TARGET_COLUMN]): int(row["samples"])
            for _, row in after_distribution.iterrows()
        },
        "important_notes": [
            "No feature scaling, clipping, normalisation, encoding, or balancing was performed.",
            "Repeated frame-level feature vectors were retained.",
            "Severe Fatigue was merged into Moderate Fatigue because it had too few samples.",
            "Suspicious rows were retained and reported rather than silently removed.",
        ],
    }

    with (RESULTS_DIRECTORY / "cleaning_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 72)
    print("Dataset cleaning completed successfully.")
    print("=" * 72)
    print(f"\nRows before cleaning: {original_rows}")
    print(f"Rows removed: {len(removed_rows)}")
    print(f"Rows after cleaning: {len(cleaned)}")
    print(f"Rows retained: {len(cleaned) / original_rows:.2%}")
    print(f"Severe rows merged: {severe_rows}")
    print(f"Suspicious rows retained: {len(suspicious_rows)}")
    print(f"Participants: {cleaned['participant_id'].nunique()}")
    print(f"Sessions: {cleaned['session_id'].nunique()}")
    print("\nClass distribution after cleaning:")
    print(after_distribution.to_string(index=False))
    print("\nParticipant summary:")
    print(participants.to_string(index=False))
    if removed_rows.empty:
        print("\nNo invalid rows required removal.")
    else:
        print("\nRemoval reasons:")
        print(removed_rows["removal_reason"].value_counts())
    print("\nClean dataset saved to:")
    print(OUTPUT_DATASET_PATH)
    print("\nCleaning reports saved to:")
    print(RESULTS_DIRECTORY)


if __name__ == "__main__":
    main()