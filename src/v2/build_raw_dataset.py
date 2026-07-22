"""
Build the DriverGuardianAI V2 raw master dataset.

This script scans:

    data/raw/

It finds the original participant collection CSV files, validates
their structure, adds participant/session metadata, merges them, and
saves one raw master dataset.

Important
---------
This script does NOT:

- scale features;
- normalise features;
- clip values;
- encode categories;
- split training and validation data;
- train any model.

Its purpose is to preserve the original collected measurements exactly
as they were recorded, while adding reliable metadata for later
participant-aware and session-aware splitting.

Input
-----
data/raw/*.csv

Output
------
data/processed/driver_guardian_raw_merged.csv

Also creates:
------
results/v2/raw_dataset_build/
    build_summary.json
    file_manifest.csv
    rejected_files.csv
    participant_summary.csv
    session_summary.csv
    class_distribution.csv
    missing_values.csv
    duplicate_summary.json

Run from the project root:

    python src/v2/build_raw_dataset.py

Or from Jupyter:

    %run src/v2/build_raw_dataset.py
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(
    __file__
).resolve().parents[2]

RAW_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "raw"
)

PROCESSED_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "processed"
)

OUTPUT_DATASET_PATH = (
    PROCESSED_DIRECTORY
    / "driver_guardian_raw_merged.csv"
)

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "raw_dataset_build"
)

FILE_MANIFEST_PATH = (
    RESULTS_DIRECTORY
    / "file_manifest.csv"
)

REJECTED_FILES_PATH = (
    RESULTS_DIRECTORY
    / "rejected_files.csv"
)

PARTICIPANT_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "participant_summary.csv"
)

SESSION_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "session_summary.csv"
)

CLASS_DISTRIBUTION_PATH = (
    RESULTS_DIRECTORY
    / "class_distribution.csv"
)

MISSING_VALUES_PATH = (
    RESULTS_DIRECTORY
    / "missing_values.csv"
)

BUILD_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "build_summary.json"
)

DUPLICATE_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "duplicate_summary.json"
)


# ============================================================
# EXPECTED RAW DATASET STRUCTURE
# ============================================================

REQUIRED_COLUMNS = [
    "timestamp",
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "state",
    "condition",
    "low_light",
    "fatigue_score",
    "face_confidence",
    "fatigue_level",
    "blink_count",
]

RAW_FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count",
]

NUMERIC_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "fatigue_score",
    "face_confidence",
    "blink_count",
]

BOOLEAN_COLUMNS = [
    "hands_detected",
    "low_light",
]

TEXT_COLUMNS = [
    "state",
    "condition",
    "fatigue_level",
]

METADATA_COLUMNS = [
    "participant_id",
    "session_id",
    "source_file",
    "source_path",
    "source_row",
]

OUTPUT_COLUMN_ORDER = [
    "timestamp",
    "participant_id",
    "session_id",
    "source_file",
    "source_path",
    "source_row",
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "state",
    "condition",
    "low_light",
    "fatigue_score",
    "face_confidence",
    "fatigue_level",
    "blink_count",
]


# ============================================================
# FILE FILTERING
# ============================================================

EXCLUDED_FILENAME_PARTS = [
    "_features",
    "cleaned",
    "processed",
    "merged",
    "combined",
    "dataset_exp",
    "driver_guardian_raw_merged",
    "training",
    "validation",
    "testing",
    "split",
]


def should_ignore_file(
    filepath: Path,
) -> Tuple[bool, Optional[str]]:
    """
    Determine whether a CSV should be ignored.

    Returns
    -------
    tuple
        (should_ignore, reason)
    """

    filename_lower = filepath.name.lower()

    for excluded_part in EXCLUDED_FILENAME_PARTS:
        if excluded_part in filename_lower:
            return (
                True,
                f"Filename contains excluded term: {excluded_part}",
            )

    if filepath.name.startswith("~$"):
        return (
            True,
            "Temporary file",
        )

    return (
        False,
        None,
    )


# ============================================================
# METADATA EXTRACTION
# ============================================================

def normalise_identifier(
    value: str,
) -> str:
    """
    Convert text into a stable lowercase identifier.
    """

    value = str(
        value
    ).strip().lower()

    value = re.sub(
        r"[^a-z0-9]+",
        "_",
        value,
    )

    value = value.strip(
        "_"
    )

    return value


def extract_participant_id(
    filepath: Path,
) -> str:
    """
    Extract participant ID from the beginning of the filename.

    Expected examples
    -----------------
    alex_drowsy_dark_ef9eebfa_20250819_124144.csv
    vinicius_normal_none_12345678_20250819_100000.csv

    Result
    ------
    alex
    vinicius
    """

    stem = filepath.stem

    first_component = stem.split(
        "_",
        maxsplit=1,
    )[0]

    participant_id = normalise_identifier(
        first_component
    )

    if not participant_id:
        raise ValueError(
            "Could not extract participant ID from filename."
        )

    return participant_id


def extract_session_id(
    filepath: Path,
) -> str:
    """
    Create a stable session ID from the full filename.
    """

    session_id = normalise_identifier(
        filepath.stem
    )

    if not session_id:
        raise ValueError(
            "Could not create session ID from filename."
        )

    return session_id


# ============================================================
# TYPE CONVERSION
# ============================================================

def convert_boolean_series(
    series: pd.Series,
    column_name: str,
) -> pd.Series:
    """
    Convert common Boolean representations to pandas Boolean values.
    """

    text_values = (
        series
        .astype(str)
        .str.strip()
        .str.lower()
    )

    mapping = {
        "true": True,
        "false": False,
        "yes": True,
        "no": False,
        "1": True,
        "0": False,
        "1.0": True,
        "0.0": False,
    }

    converted = text_values.map(
        mapping
    )

    invalid_mask = (
        converted.isna()
        & series.notna()
    )

    if invalid_mask.any():
        invalid_examples = (
            series[
                invalid_mask
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            f"Column '{column_name}' contains unsupported Boolean "
            f"values: {invalid_examples[:10]}"
        )

    return converted.astype(
        "boolean"
    )


def prepare_dataframe_types(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert raw columns to consistent data types without scaling them.
    """

    dataframe = dataframe.copy()

    dataframe["timestamp"] = pd.to_datetime(
        dataframe["timestamp"],
        errors="coerce",
    )

    for column in NUMERIC_COLUMNS:
        dataframe[column] = pd.to_numeric(
            dataframe[column],
            errors="coerce",
        )

    for column in BOOLEAN_COLUMNS:
        dataframe[column] = convert_boolean_series(
            dataframe[column],
            column,
        )

    for column in TEXT_COLUMNS:
        dataframe[column] = (
            dataframe[column]
            .astype("string")
            .str.strip()
        )

    dataframe["condition"] = (
        dataframe["condition"]
        .str.lower()
    )

    dataframe["state"] = (
        dataframe["state"]
        .str.lower()
    )

    return dataframe


# ============================================================
# RAW FILE VALIDATION
# ============================================================

def validate_columns(
    dataframe: pd.DataFrame,
    filepath: Path,
) -> None:
    """
    Validate required columns in one raw CSV.
    """

    missing_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Missing required columns: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(
            "CSV contains no rows."
        )


def validate_critical_values(
    dataframe: pd.DataFrame,
) -> None:
    """
    Validate important fields after type conversion.
    """

    critical_columns = [
        "timestamp",
        "ear",
        "yawn_score",
        "head_tilt",
        "fatigue_level",
        "blink_count",
    ]

    completely_missing = [
        column
        for column in critical_columns
        if dataframe[column].isna().all()
    ]

    if completely_missing:
        raise ValueError(
            "Critical columns contain no usable values: "
            f"{completely_missing}"
        )


def validate_raw_ranges(
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Check raw ranges and return warnings.

    These are warnings rather than hard failures because raw recordings
    may contain genuine outliers that should be audited later.
    """

    warnings = []

    range_expectations = {
        "ear": (
            0.0,
            1.0,
        ),
        "yawn_score": (
            0.0,
            2.0,
        ),
        "head_tilt": (
            0.0,
            500.0,
        ),
        "fatigue_score": (
            -10.0,
            20.0,
        ),
        "face_confidence": (
            0.0,
            1.0,
        ),
        "blink_count": (
            0.0,
            10000.0,
        ),
    }

    for column, (
        minimum_expected,
        maximum_expected,
    ) in range_expectations.items():

        values = dataframe[
            column
        ].dropna()

        if values.empty:
            continue

        observed_minimum = float(
            values.min()
        )

        observed_maximum = float(
            values.max()
        )

        if (
            observed_minimum < minimum_expected
            or observed_maximum > maximum_expected
        ):
            warnings.append(
                f"{column} range "
                f"[{observed_minimum}, {observed_maximum}] "
                "falls outside the broad expected raw range "
                f"[{minimum_expected}, {maximum_expected}]."
            )

    return warnings


# ============================================================
# FILE LOADING
# ============================================================

def load_one_raw_file(
    filepath: Path,
) -> Tuple[
    pd.DataFrame,
    Dict,
]:
    """
    Load, validate, and enrich one raw collection file.
    """

    dataframe = pd.read_csv(
        filepath
    )

    validate_columns(
        dataframe,
        filepath,
    )

    dataframe = dataframe[
        REQUIRED_COLUMNS
    ].copy()

    dataframe = prepare_dataframe_types(
        dataframe
    )

    validate_critical_values(
        dataframe
    )

    range_warnings = validate_raw_ranges(
        dataframe
    )

    participant_id = extract_participant_id(
        filepath
    )

    session_id = extract_session_id(
        filepath
    )

    dataframe.insert(
        1,
        "participant_id",
        participant_id,
    )

    dataframe.insert(
        2,
        "session_id",
        session_id,
    )

    dataframe.insert(
        3,
        "source_file",
        filepath.name,
    )

    dataframe.insert(
        4,
        "source_path",
        str(
            filepath.relative_to(
                PROJECT_ROOT
            )
        ),
    )

    dataframe.insert(
        5,
        "source_row",
        np.arange(
            len(dataframe),
            dtype=int,
        ),
    )

    dataframe = dataframe[
        OUTPUT_COLUMN_ORDER
    ]

    duplicate_rows_inside_file = int(
        dataframe[
            REQUIRED_COLUMNS
        ]
        .duplicated()
        .sum()
    )

    manifest_row = {
        "source_file": filepath.name,
        "source_path": str(
            filepath.relative_to(
                PROJECT_ROOT
            )
        ),
        "participant_id": participant_id,
        "session_id": session_id,
        "rows": int(
            len(dataframe)
        ),
        "timestamp_missing": int(
            dataframe["timestamp"]
            .isna()
            .sum()
        ),
        "duplicate_raw_rows_inside_file": (
            duplicate_rows_inside_file
        ),
        "fatigue_levels": " | ".join(
            sorted(
                dataframe[
                    "fatigue_level"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        ),
        "conditions": " | ".join(
            sorted(
                dataframe[
                    "condition"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        ),
        "states": " | ".join(
            sorted(
                dataframe[
                    "state"
                ]
                .dropna()
                .astype(str)
                .unique()
                .tolist()
            )
        ),
        "range_warnings": " | ".join(
            range_warnings
        ),
    }

    return (
        dataframe,
        manifest_row,
    )


# ============================================================
# SUMMARIES
# ============================================================

def create_class_distribution(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create fatigue-level counts and percentages.
    """

    counts = (
        dataframe[
            "fatigue_level"
        ]
        .value_counts(
            dropna=False
        )
        .rename_axis(
            "fatigue_level"
        )
        .reset_index(
            name="samples"
        )
    )

    counts["percentage"] = (
        counts["samples"]
        / counts["samples"].sum()
    )

    return counts


def create_participant_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise rows, sessions, and labels per participant.
    """

    rows = []

    for participant_id, group in dataframe.groupby(
        "participant_id",
        sort=True,
    ):

        fatigue_counts = (
            group[
                "fatigue_level"
            ]
            .value_counts()
            .to_dict()
        )

        rows.append(
            {
                "participant_id": participant_id,
                "samples": int(
                    len(group)
                ),
                "sessions": int(
                    group[
                        "session_id"
                    ].nunique()
                ),
                "conditions": " | ".join(
                    sorted(
                        group[
                            "condition"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                ),
                "states": " | ".join(
                    sorted(
                        group[
                            "state"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                ),
                "alert_samples": int(
                    fatigue_counts.get(
                        "Alert",
                        0,
                    )
                ),
                "mild_samples": int(
                    fatigue_counts.get(
                        "Mild Fatigue",
                        0,
                    )
                ),
                "moderate_samples": int(
                    fatigue_counts.get(
                        "Moderate Fatigue",
                        0,
                    )
                ),
                "severe_samples": int(
                    fatigue_counts.get(
                        "Severe Fatigue",
                        0,
                    )
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        "participant_id"
    )


def create_session_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise every recording session.
    """

    rows = []

    for session_id, group in dataframe.groupby(
        "session_id",
        sort=True,
    ):

        fatigue_counts = (
            group[
                "fatigue_level"
            ]
            .value_counts()
            .to_dict()
        )

        valid_timestamps = (
            group[
                "timestamp"
            ]
            .dropna()
            .sort_values()
        )

        duration_seconds = float(
            (
                valid_timestamps.iloc[-1]
                - valid_timestamps.iloc[0]
            ).total_seconds()
        ) if len(
            valid_timestamps
        ) >= 2 else 0.0

        rows.append(
            {
                "session_id": session_id,
                "participant_id": str(
                    group[
                        "participant_id"
                    ].iloc[0]
                ),
                "source_file": str(
                    group[
                        "source_file"
                    ].iloc[0]
                ),
                "samples": int(
                    len(group)
                ),
                "duration_seconds": (
                    duration_seconds
                ),
                "condition": " | ".join(
                    sorted(
                        group[
                            "condition"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                ),
                "state": " | ".join(
                    sorted(
                        group[
                            "state"
                        ]
                        .dropna()
                        .astype(str)
                        .unique()
                        .tolist()
                    )
                ),
                "alert_samples": int(
                    fatigue_counts.get(
                        "Alert",
                        0,
                    )
                ),
                "mild_samples": int(
                    fatigue_counts.get(
                        "Mild Fatigue",
                        0,
                    )
                ),
                "moderate_samples": int(
                    fatigue_counts.get(
                        "Moderate Fatigue",
                        0,
                    )
                ),
                "severe_samples": int(
                    fatigue_counts.get(
                        "Severe Fatigue",
                        0,
                    )
                ),
                "ear_mean": float(
                    group[
                        "ear"
                    ].mean()
                ),
                "yawn_score_mean": float(
                    group[
                        "yawn_score"
                    ].mean()
                ),
                "head_tilt_mean": float(
                    group[
                        "head_tilt"
                    ].mean()
                ),
                "blink_count_mean": float(
                    group[
                        "blink_count"
                    ].mean()
                ),
            }
        )

    return pd.DataFrame(
        rows
    ).sort_values(
        [
            "participant_id",
            "session_id",
        ]
    )


def create_missing_value_summary(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise missing values by column.
    """

    summary = pd.DataFrame(
        {
            "column": dataframe.columns,
            "missing_values": [
                int(
                    dataframe[
                        column
                    ].isna().sum()
                )
                for column in dataframe.columns
            ],
        }
    )

    summary["missing_percentage"] = (
        summary["missing_values"]
        / len(dataframe)
    )

    return summary.sort_values(
        "missing_values",
        ascending=False,
    )


def create_duplicate_summary(
    dataframe: pd.DataFrame,
) -> Dict:
    """
    Calculate several duplicate definitions.
    """

    exact_duplicates = int(
        dataframe.duplicated().sum()
    )

    raw_value_columns = [
        column
        for column in REQUIRED_COLUMNS
        if column != "timestamp"
    ]

    duplicates_without_metadata = int(
        dataframe[
            REQUIRED_COLUMNS
        ]
        .duplicated()
        .sum()
    )

    repeated_feature_vectors = int(
        dataframe[
            RAW_FEATURE_COLUMNS
        ]
        .duplicated()
        .sum()
    )

    repeated_features_and_label = int(
        dataframe[
            [
                *RAW_FEATURE_COLUMNS,
                "fatigue_level",
            ]
        ]
        .duplicated()
        .sum()
    )

    return {
        "rows": int(
            len(dataframe)
        ),
        "exact_duplicates_with_metadata": (
            exact_duplicates
        ),
        "duplicate_original_rows": (
            duplicates_without_metadata
        ),
        "repeated_feature_vectors": (
            repeated_feature_vectors
        ),
        "repeated_feature_vectors_with_same_label": (
            repeated_features_and_label
        ),
        "important_note": (
            "Repeated feature vectors are not automatically errors. "
            "Adjacent video frames may legitimately contain identical "
            "measurements. No duplicates are removed by this script."
        ),
    }


# ============================================================
# MAIN BUILD PROCESS
# ============================================================

def discover_csv_files() -> List[Path]:
    """
    Find all candidate CSV files recursively under data/raw.
    """

    if not RAW_DIRECTORY.exists():
        raise FileNotFoundError(
            "Raw data directory does not exist: "
            f"{RAW_DIRECTORY}"
        )

    return sorted(
        RAW_DIRECTORY.rglob(
            "*.csv"
        )
    )


def main() -> None:
    """
    Build the raw merged V2 dataset.
    """

    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Raw Dataset Builder")
    print("=" * 72)

    PROCESSED_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    csv_files = discover_csv_files()

    print(
        f"\nCSV files discovered: {len(csv_files)}"
    )

    if not csv_files:
        raise FileNotFoundError(
            "No CSV files were found inside "
            f"{RAW_DIRECTORY}"
        )

    accepted_dataframes = []
    manifest_rows = []
    rejected_rows = []

    ignored_count = 0

    for file_number, filepath in enumerate(
        csv_files,
        start=1,
    ):

        relative_path = filepath.relative_to(
            PROJECT_ROOT
        )

        should_ignore, ignore_reason = (
            should_ignore_file(
                filepath
            )
        )

        if should_ignore:
            ignored_count += 1

            rejected_rows.append(
                {
                    "source_file": filepath.name,
                    "source_path": str(
                        relative_path
                    ),
                    "status": "ignored",
                    "reason": ignore_reason,
                }
            )

            print(
                f"[{file_number}/{len(csv_files)}] "
                f"Ignored: {relative_path}"
            )

            continue

        try:
            (
                dataframe,
                manifest_row,
            ) = load_one_raw_file(
                filepath
            )

            accepted_dataframes.append(
                dataframe
            )

            manifest_rows.append(
                manifest_row
            )

            print(
                f"[{file_number}/{len(csv_files)}] "
                f"Accepted: {relative_path} "
                f"({len(dataframe)} rows)"
            )

        except Exception as error:

            rejected_rows.append(
                {
                    "source_file": filepath.name,
                    "source_path": str(
                        relative_path
                    ),
                    "status": "rejected",
                    "reason": str(
                        error
                    ),
                }
            )

            print(
                f"[{file_number}/{len(csv_files)}] "
                f"Rejected: {relative_path}"
            )

            print(
                f"    Reason: {error}"
            )

    if not accepted_dataframes:
        raise RuntimeError(
            "No valid raw collection CSV files were accepted."
        )

    print(
        "\nMerging accepted files..."
    )

    merged = pd.concat(
        accepted_dataframes,
        ignore_index=True,
    )

    merged = merged.sort_values(
        [
            "participant_id",
            "session_id",
            "timestamp",
            "source_row",
        ],
        kind="stable",
    ).reset_index(
        drop=True
    )

    merged.to_csv(
        OUTPUT_DATASET_PATH,
        index=False,
    )

    manifest = pd.DataFrame(
        manifest_rows
    )

    rejected = pd.DataFrame(
        rejected_rows
    )

    participant_summary = (
        create_participant_summary(
            merged
        )
    )

    session_summary = (
        create_session_summary(
            merged
        )
    )

    class_distribution = (
        create_class_distribution(
            merged
        )
    )

    missing_summary = (
        create_missing_value_summary(
            merged
        )
    )

    duplicate_summary = (
        create_duplicate_summary(
            merged
        )
    )

    manifest.to_csv(
        FILE_MANIFEST_PATH,
        index=False,
    )

    rejected.to_csv(
        REJECTED_FILES_PATH,
        index=False,
    )

    participant_summary.to_csv(
        PARTICIPANT_SUMMARY_PATH,
        index=False,
    )

    session_summary.to_csv(
        SESSION_SUMMARY_PATH,
        index=False,
    )

    class_distribution.to_csv(
        CLASS_DISTRIBUTION_PATH,
        index=False,
    )

    missing_summary.to_csv(
        MISSING_VALUES_PATH,
        index=False,
    )

    with DUPLICATE_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            duplicate_summary,
            file,
            indent=4,
        )

    build_summary = {
        "project": "DriverGuardianAI V2",
        "raw_directory": str(
            RAW_DIRECTORY
        ),
        "output_dataset": str(
            OUTPUT_DATASET_PATH
        ),
        "csv_files_discovered": int(
            len(csv_files)
        ),
        "files_accepted": int(
            len(manifest_rows)
        ),
        "files_ignored_or_rejected": int(
            len(rejected_rows)
        ),
        "files_ignored": int(
            ignored_count
        ),
        "merged_rows": int(
            len(merged)
        ),
        "participants": int(
            merged[
                "participant_id"
            ].nunique()
        ),
        "sessions": int(
            merged[
                "session_id"
            ].nunique()
        ),
        "conditions": sorted(
            merged[
                "condition"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "states": sorted(
            merged[
                "state"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "fatigue_levels": sorted(
            merged[
                "fatigue_level"
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        ),
        "missing_values_total": int(
            merged.isna().sum().sum()
        ),
        "duplicate_summary": (
            duplicate_summary
        ),
        "important_note": (
            "This dataset contains raw collected measurements. "
            "No clipping, scaling, normalisation, encoding, row "
            "removal, balancing, or train-validation splitting was "
            "performed."
        ),
    }

    with BUILD_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            build_summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("Raw dataset build completed successfully.")
    print("=" * 72)

    print(
        f"\nFiles accepted: {len(manifest_rows)}"
    )

    print(
        "Files ignored/rejected: "
        f"{len(rejected_rows)}"
    )

    print(
        f"Merged rows: {len(merged)}"
    )

    print(
        "Participants: "
        f"{merged['participant_id'].nunique()}"
    )

    print(
        "Sessions: "
        f"{merged['session_id'].nunique()}"
    )

    print("\nClass distribution:")

    print(
        class_distribution.to_string(
            index=False
        )
    )

    print("\nParticipant summary:")

    print(
        participant_summary.to_string(
            index=False
        )
    )

    print("\nMissing-value summary:")

    print(
        missing_summary[
            missing_summary[
                "missing_values"
            ] > 0
        ].to_string(
            index=False
        )
        if (
            missing_summary[
                "missing_values"
            ] > 0
        ).any()
        else "No missing values detected."
    )

    print("\nDuplicate summary:")

    for key, value in duplicate_summary.items():
        print(
            f"{key}: {value}"
        )

    print(
        "\nMerged raw dataset saved to:"
    )

    print(
        OUTPUT_DATASET_PATH
    )

    print(
        "\nBuild reports saved to:"
    )

    print(
        RESULTS_DIRECTORY
    )


if __name__ == "__main__":
    main()