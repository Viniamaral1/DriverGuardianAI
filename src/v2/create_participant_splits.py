"""
Create participant-aware DriverGuardianAI V2 splits.

Input
-----
data/processed/driver_guardian_v2_clean.csv

Outputs
-------
data/splits/v2/train.csv
data/splits/v2/calibration.csv
data/splits/v2/test.csv

Reports
-------
results/v2/participant_splits/
    split_summary.json
    participant_assignment.csv
    split_class_distribution.csv
    split_feature_summary.csv
    alias_changes.csv

Important
---------
- "vini" is normalised to "vinicius".
- Every participant belongs to exactly one split.
- Every session belongs to exactly one split.
- No scaling, encoding, balancing, or training is performed.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DATASET_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "driver_guardian_v2_clean.csv"
)

SPLIT_DIRECTORY = (
    PROJECT_ROOT
    / "data"
    / "splits"
    / "v2"
)

TRAIN_PATH = SPLIT_DIRECTORY / "train.csv"
CALIBRATION_PATH = SPLIT_DIRECTORY / "calibration.csv"
TEST_PATH = SPLIT_DIRECTORY / "test.csv"

RESULTS_DIRECTORY = (
    PROJECT_ROOT
    / "results"
    / "v2"
    / "participant_splits"
)

SPLIT_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "split_summary.json"
)

PARTICIPANT_ASSIGNMENT_PATH = (
    RESULTS_DIRECTORY
    / "participant_assignment.csv"
)

CLASS_DISTRIBUTION_PATH = (
    RESULTS_DIRECTORY
    / "split_class_distribution.csv"
)

FEATURE_SUMMARY_PATH = (
    RESULTS_DIRECTORY
    / "split_feature_summary.csv"
)

ALIAS_CHANGES_PATH = (
    RESULTS_DIRECTORY
    / "alias_changes.csv"
)


# ============================================================
# SETTINGS
# ============================================================

TARGET_COLUMN = "fatigue_level"
PARTICIPANT_COLUMN = "participant_id"
SESSION_COLUMN = "session_id"

PARTICIPANT_ALIASES = {
    "vini": "vinicius",
}

EXPECTED_PARTICIPANTS = {
    "alex",
    "karys",
    "lewis",
    "rochelle",
    "runnel",
    "shayna",
    "teresa",
    "vinicius",
}

# Fixed and reproducible split.
#
# The earlier untouched test participants are preserved:
#   karys, shayna
#
# All recordings belonging to vini/vinicius stay together.
FIXED_SPLIT_ASSIGNMENT = {
    "alex": "train",
    "lewis": "train",
    "rochelle": "train",
    "runnel": "train",
    "teresa": "train",
    "vinicius": "calibration",
    "karys": "test",
    "shayna": "test",
}

MODEL_FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count",
]

NUMERIC_FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count",
]


# ============================================================
# LOADING
# ============================================================

def load_dataset() -> pd.DataFrame:
    """
    Load the cleaned V2 dataset.
    """

    if not INPUT_DATASET_PATH.exists():
        raise FileNotFoundError(
            "Clean V2 dataset was not found: "
            f"{INPUT_DATASET_PATH}"
        )

    dataframe = pd.read_csv(
        INPUT_DATASET_PATH
    )

    required_columns = [
        PARTICIPANT_COLUMN,
        SESSION_COLUMN,
        TARGET_COLUMN,
        "timestamp",
        "source_row",
        "source_file",
        *MODEL_FEATURE_COLUMNS,
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in dataframe.columns
    ]

    if missing_columns:
        raise ValueError(
            "Clean dataset is missing columns: "
            f"{missing_columns}"
        )

    if dataframe.empty:
        raise ValueError(
            "Clean dataset contains no rows."
        )

    return dataframe


# ============================================================
# PARTICIPANT NORMALISATION
# ============================================================

def normalise_participant_ids(
    dataframe: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Apply participant aliases and return an audit table.
    """

    dataframe = dataframe.copy()

    original_ids = (
        dataframe[PARTICIPANT_COLUMN]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    normalised_ids = original_ids.replace(
        PARTICIPANT_ALIASES
    )

    changed_mask = (
        original_ids
        != normalised_ids
    )

    alias_changes = dataframe.loc[
        changed_mask,
        [
            SESSION_COLUMN,
            "source_file",
            "source_row",
        ],
    ].copy()

    if not alias_changes.empty:
        alias_changes.insert(
            0,
            "original_participant_id",
            original_ids.loc[
                changed_mask
            ].values,
        )

        alias_changes.insert(
            1,
            "normalised_participant_id",
            normalised_ids.loc[
                changed_mask
            ].values,
        )

    dataframe[
        PARTICIPANT_COLUMN
    ] = normalised_ids

    return (
        dataframe,
        alias_changes,
    )


def validate_participants(
    dataframe: pd.DataFrame,
) -> List[str]:
    """
    Validate the participant list after aliasing.
    """

    participants = sorted(
        dataframe[
            PARTICIPANT_COLUMN
        ]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    participant_set = set(
        participants
    )

    if participant_set != EXPECTED_PARTICIPANTS:
        raise ValueError(
            "Unexpected participants after alias normalisation.\n"
            f"Expected: {sorted(EXPECTED_PARTICIPANTS)}\n"
            f"Found: {participants}"
        )

    assignment_participants = set(
        FIXED_SPLIT_ASSIGNMENT.keys()
    )

    if assignment_participants != participant_set:
        raise ValueError(
            "The fixed split assignment does not match "
            "the dataset participants."
        )

    return participants


# ============================================================
# SPLIT ASSIGNMENT AND LEAKAGE CHECKS
# ============================================================

def assign_splits(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add dataset_split from the fixed participant assignment.
    """

    dataframe = dataframe.copy()

    dataframe[
        "dataset_split"
    ] = dataframe[
        PARTICIPANT_COLUMN
    ].map(
        FIXED_SPLIT_ASSIGNMENT
    )

    if dataframe[
        "dataset_split"
    ].isna().any():
        missing = sorted(
            dataframe.loc[
                dataframe[
                    "dataset_split"
                ].isna(),
                PARTICIPANT_COLUMN,
            ]
            .astype(str)
            .unique()
            .tolist()
        )

        raise ValueError(
            "Participants without a split assignment: "
            f"{missing}"
        )

    return dataframe


def validate_no_participant_leakage(
    dataframe: pd.DataFrame,
) -> None:
    """
    Ensure no participant appears in multiple splits.
    """

    split_counts = (
        dataframe.groupby(
            PARTICIPANT_COLUMN
        )[
            "dataset_split"
        ]
        .nunique()
    )

    leakage = split_counts[
        split_counts > 1
    ]

    if not leakage.empty:
        raise RuntimeError(
            "Participant leakage detected: "
            f"{leakage.index.tolist()}"
        )


def validate_no_session_leakage(
    dataframe: pd.DataFrame,
) -> None:
    """
    Ensure no recording session appears in multiple splits.
    """

    split_counts = (
        dataframe.groupby(
            SESSION_COLUMN
        )[
            "dataset_split"
        ]
        .nunique()
    )

    leakage = split_counts[
        split_counts > 1
    ]

    if not leakage.empty:
        raise RuntimeError(
            "Session leakage detected: "
            f"{leakage.index.tolist()}"
        )


# ============================================================
# REPORTS
# ============================================================

def create_participant_assignment_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise samples, sessions, classes, and split by participant.
    """

    rows = []

    for participant, group in dataframe.groupby(
        PARTICIPANT_COLUMN,
        sort=True,
    ):

        counts = (
            group[
                TARGET_COLUMN
            ]
            .value_counts()
            .to_dict()
        )

        rows.append(
            {
                "participant_id": participant,
                "dataset_split": str(
                    group[
                        "dataset_split"
                    ].iloc[0]
                ),
                "samples": int(
                    len(group)
                ),
                "sessions": int(
                    group[
                        SESSION_COLUMN
                    ].nunique()
                ),
                "alert_samples": int(
                    counts.get(
                        "Alert",
                        0,
                    )
                ),
                "mild_samples": int(
                    counts.get(
                        "Mild Fatigue",
                        0,
                    )
                ),
                "moderate_samples": int(
                    counts.get(
                        "Moderate Fatigue",
                        0,
                    )
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
            }
        )

    report = pd.DataFrame(
        rows
    )

    split_order = pd.CategoricalDtype(
        categories=[
            "train",
            "calibration",
            "test",
        ],
        ordered=True,
    )

    report[
        "dataset_split"
    ] = report[
        "dataset_split"
    ].astype(
        split_order
    )

    return report.sort_values(
        [
            "dataset_split",
            "participant_id",
        ]
    )


def create_class_distribution_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Create class counts and percentages for each split.
    """

    rows = []

    for split_name in [
        "train",
        "calibration",
        "test",
    ]:

        split_group = dataframe[
            dataframe[
                "dataset_split"
            ]
            == split_name
        ]

        total = len(
            split_group
        )

        counts = (
            split_group[
                TARGET_COLUMN
            ]
            .value_counts()
        )

        for class_name, samples in counts.items():

            rows.append(
                {
                    "dataset_split": (
                        split_name
                    ),
                    "fatigue_level": (
                        class_name
                    ),
                    "samples": int(
                        samples
                    ),
                    "percentage_within_split": float(
                        samples / total
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def create_feature_summary_report(
    dataframe: pd.DataFrame,
) -> pd.DataFrame:
    """
    Summarise raw numeric feature distributions by split.
    """

    rows = []

    for split_name in [
        "train",
        "calibration",
        "test",
    ]:

        split_group = dataframe[
            dataframe[
                "dataset_split"
            ]
            == split_name
        ]

        for feature in NUMERIC_FEATURE_COLUMNS:

            values = pd.to_numeric(
                split_group[
                    feature
                ],
                errors="coerce",
            )

            values = (
                values.astype(
                    "float64"
                )
                .dropna()
            )

            if values.empty:
                continue

            rows.append(
                {
                    "dataset_split": (
                        split_name
                    ),
                    "feature": feature,
                    "samples": int(
                        len(values)
                    ),
                    "mean": float(
                        values.mean()
                    ),
                    "std": float(
                        values.std(
                            ddof=1
                        )
                    )
                    if len(values) > 1
                    else 0.0,
                    "minimum": float(
                        values.min()
                    ),
                    "p05": float(
                        values.quantile(
                            0.05
                        )
                    ),
                    "median": float(
                        values.median()
                    ),
                    "p95": float(
                        values.quantile(
                            0.95
                        )
                    ),
                    "maximum": float(
                        values.max()
                    ),
                }
            )

    return pd.DataFrame(
        rows
    )


def build_split_summary(
    dataframe: pd.DataFrame,
    alias_changes: pd.DataFrame,
) -> Dict:
    """
    Build JSON metadata for the final split.
    """

    splits = {}

    for split_name in [
        "train",
        "calibration",
        "test",
    ]:

        group = dataframe[
            dataframe[
                "dataset_split"
            ]
            == split_name
        ]

        class_counts = (
            group[
                TARGET_COLUMN
            ]
            .value_counts()
            .to_dict()
        )

        splits[
            split_name
        ] = {
            "samples": int(
                len(group)
            ),
            "participants": sorted(
                group[
                    PARTICIPANT_COLUMN
                ]
                .astype(str)
                .unique()
                .tolist()
            ),
            "participant_count": int(
                group[
                    PARTICIPANT_COLUMN
                ].nunique()
            ),
            "sessions": int(
                group[
                    SESSION_COLUMN
                ].nunique()
            ),
            "class_distribution": {
                str(key): int(value)
                for key, value
                in class_counts.items()
            },
        }

    return {
        "project": "DriverGuardianAI V2",
        "input_dataset": str(
            INPUT_DATASET_PATH
        ),
        "participant_aliases": (
            PARTICIPANT_ALIASES
        ),
        "alias_rows_changed": int(
            len(alias_changes)
        ),
        "unique_participants_after_aliasing": int(
            dataframe[
                PARTICIPANT_COLUMN
            ].nunique()
        ),
        "fixed_split_assignment": (
            FIXED_SPLIT_ASSIGNMENT
        ),
        "splits": splits,
        "participant_leakage": False,
        "session_leakage": False,
        "important_notes": [
            (
                "vini and vinicius are treated as one participant."
            ),
            (
                "Each participant appears in exactly one split."
            ),
            (
                "Each session appears in exactly one split."
            ),
            (
                "No scaling, encoding, balancing, or training "
                "was performed."
            ),
        ],
    }


# ============================================================
# SAVE DATASETS
# ============================================================

def save_split(
    dataframe: pd.DataFrame,
    split_name: str,
    filepath: Path,
) -> pd.DataFrame:
    """
    Save one split in stable participant/session/time order.
    """

    split_dataframe = dataframe[
        dataframe[
            "dataset_split"
        ]
        == split_name
    ].copy()

    split_dataframe = (
        split_dataframe.sort_values(
            [
                PARTICIPANT_COLUMN,
                SESSION_COLUMN,
                "timestamp",
                "source_row",
            ],
            kind="stable",
        )
        .reset_index(
            drop=True
        )
    )

    split_dataframe.to_csv(
        filepath,
        index=False,
    )

    return split_dataframe


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    """
    Create the V2 participant-aware splits.
    """

    print("=" * 72)
    print("DriverGuardianAI V2")
    print("Participant-Aware Dataset Splits")
    print("=" * 72)

    SPLIT_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(
        "\nLoading cleaned V2 dataset..."
    )

    dataframe = load_dataset()

    print(
        f"Rows loaded: {len(dataframe)}"
    )

    print(
        "\nNormalising participant aliases..."
    )

    (
        dataframe,
        alias_changes,
    ) = normalise_participant_ids(
        dataframe
    )

    print(
        "Rows changed from vini to vinicius: "
        f"{len(alias_changes)}"
    )

    participants = validate_participants(
        dataframe
    )

    print(
        "Participants after aliasing "
        f"({len(participants)}):"
    )

    print(
        participants
    )

    dataframe = assign_splits(
        dataframe
    )

    validate_no_participant_leakage(
        dataframe
    )

    validate_no_session_leakage(
        dataframe
    )

    train_dataframe = save_split(
        dataframe,
        "train",
        TRAIN_PATH,
    )

    calibration_dataframe = save_split(
        dataframe,
        "calibration",
        CALIBRATION_PATH,
    )

    test_dataframe = save_split(
        dataframe,
        "test",
        TEST_PATH,
    )

    participant_assignment = (
        create_participant_assignment_report(
            dataframe
        )
    )

    class_distribution = (
        create_class_distribution_report(
            dataframe
        )
    )

    feature_summary = (
        create_feature_summary_report(
            dataframe
        )
    )

    split_summary = build_split_summary(
        dataframe,
        alias_changes,
    )

    participant_assignment.to_csv(
        PARTICIPANT_ASSIGNMENT_PATH,
        index=False,
    )

    class_distribution.to_csv(
        CLASS_DISTRIBUTION_PATH,
        index=False,
    )

    feature_summary.to_csv(
        FEATURE_SUMMARY_PATH,
        index=False,
    )

    alias_changes.to_csv(
        ALIAS_CHANGES_PATH,
        index=False,
    )

    with SPLIT_SUMMARY_PATH.open(
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            split_summary,
            file,
            indent=4,
        )

    print("\n" + "=" * 72)
    print("Participant-aware splits created successfully.")
    print("=" * 72)

    print("\nParticipant assignment:")

    print(
        participant_assignment.to_string(
            index=False
        )
    )

    print("\nSplit sizes:")

    print(
        f"Train       : {len(train_dataframe)}"
    )

    print(
        "Calibration : "
        f"{len(calibration_dataframe)}"
    )

    print(
        f"Test        : {len(test_dataframe)}"
    )

    print("\nClass distribution by split:")

    print(
        class_distribution.to_string(
            index=False
        )
    )

    print("\nLeakage checks:")
    print("Participant leakage: none")
    print("Session leakage: none")

    print("\nSaved split files:")
    print(TRAIN_PATH)
    print(CALIBRATION_PATH)
    print(TEST_PATH)

    print("\nReports saved to:")
    print(RESULTS_DIRECTORY)


if __name__ == "__main__":
    main()