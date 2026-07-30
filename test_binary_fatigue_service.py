"""
Test the complete binary DriverGuardianAI service.

Pipeline
--------
Real dataset sample
→ ClassicalPredictor
→ BinaryDecisionAgent
→ LoggingAgent

Run in Jupyter:

    %run test_binary_fatigue_service.py

Or from Anaconda Prompt:

    python test_binary_fatigue_service.py
"""

import pandas as pd

from src.binary_fatigue_service import (
    BinaryFatigueService,
)


DATASET_PATH = "data/dataset_exp3.csv"

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


def convert_actual_label(label):
    """
    Convert original dataset labels into binary labels.
    """

    if label == "Alert":
        return "Alert"

    if label in {
        "Mild Fatigue",
        "Moderate Fatigue",
        "Severe Fatigue",
    }:
        return "Fatigue"

    raise ValueError(
        f"Unexpected fatigue label: {label}"
    )


def print_result(
    sample_number,
    actual_label,
    result,
):
    """
    Print one complete binary service result.
    """

    prediction = result["prediction"]
    decision = result["decision"]

    print("\n" + "=" * 72)
    print(f"Sample {sample_number}")
    print("=" * 72)

    print(
        f"Actual label                : "
        f"{actual_label}"
    )

    print(
        f"Model prediction            : "
        f"{prediction['prediction']}"
    )

    print(
        f"Fatigue probability         : "
        f"{prediction['fatigue_probability']:.2%}"
    )

    print(
        f"Decision threshold          : "
        f"{prediction['threshold']:.2f}"
    )

    print(
        f"Temporal state              : "
        f"{decision['state']}"
    )

    print(
        f"Alert level                 : "
        f"{decision['alert_level']}"
    )

    print(
        f"Trigger alert               : "
        f"{decision['trigger_alert']}"
    )

    print(
        f"Fatigue ratio               : "
        f"{decision['fatigue_ratio']:.2%}"
    )

    print(
        f"Average Fatigue probability : "
        f"{decision['average_fatigue_probability']:.2%}"
    )

    print(
        f"Consecutive Fatigue         : "
        f"{decision['consecutive_fatigue']}"
    )

    print(
        f"History size                : "
        f"{decision['history_size']}"
    )

    print(
        f"Reason                      : "
        f"{decision['reason']}"
    )


def main():
    """
    Run the binary service using real dataset samples.
    """

    service = BinaryFatigueService(
        model_path=(
            "models/"
            "driver_guardian_calibrated_hgb.joblib"
        ),
        enable_logging=True,
        log_directory="logs",
        window_size=12,
        minimum_history=5,
        warning_ratio_threshold=0.50,
        critical_ratio_threshold=0.70,
        consecutive_fatigue_required=3,
        alert_cooldown_seconds=10.0,
    )

    print("\nModel and service information:")

    information = service.model_information()

    print(
        information
    )

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    fatigue_rows = dataframe[
        dataframe["fatigue_level"].isin(
            [
                "Moderate Fatigue",
                "Severe Fatigue",
                "Mild Fatigue",
            ]
        )
    ]

    if len(fatigue_rows) < 7:
        raise ValueError(
            "Not enough Fatigue rows were found "
            "for the temporal test."
        )

    selected_rows = fatigue_rows.sample(
        n=7,
        random_state=42,
    ).reset_index(
        drop=True
    )

    print("\n" + "=" * 72)
    print(
        "DriverGuardianAI Binary Fatigue Service Test"
    )
    print("=" * 72)

    for sample_number, row in selected_rows.iterrows():
        features = row[
            FEATURE_COLUMNS
        ].to_dict()

        actual_label = convert_actual_label(
            row["fatigue_level"]
        )

        result = service.process(
            features
        )

        print_result(
            sample_number=sample_number + 1,
            actual_label=actual_label,
            result=result,
        )

    print("\nLog file:")

    if result["log_path"]:
        print(
            result["log_path"]
        )

    print("\n" + "=" * 72)
    print(
        "Binary Fatigue Service test completed."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()