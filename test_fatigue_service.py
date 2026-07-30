"""
Test the complete DriverGuardianAI fatigue service.

Run in Jupyter with:

    %run test_fatigue_service.py
"""

import pandas as pd

from src.fatigue_service import FatigueService


DATASET_PATH = "data/dataset_exp3.csv"

FEATURE_COLUMNS = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count"
]


def print_result(index, actual_label, result):
    """
    Print one complete service result.
    """

    prediction = result["prediction"]
    decision = result["decision"]

    print("\n" + "=" * 70)
    print(f"Sample {index}")
    print("=" * 70)

    print(f"Actual label   : {actual_label}")
    print(f"Model output   : {prediction['prediction']}")
    print(f"Confidence     : {prediction['confidence']:.2%}")
    print(f"Temporal state : {decision['state']}")
    print(f"Alert level    : {decision['alert_level']}")
    print(f"Trigger alert  : {decision['trigger_alert']}")
    print(f"Reason         : {decision['reason']}")


def main():
    """
    Run the service using real dataset samples.
    """

    service = FatigueService(
        model_path="models/driver_guardian_best.pth",
        preprocessing_path="models/preprocessing.pkl",
        hidden_dims=[
            256,
            128,
            64
        ],
        dropout=0.30,
        num_classes=3,
        enable_logging=True,
        log_directory="logs"
    )

    dataset = pd.read_csv(
        DATASET_PATH
    )

    moderate_rows = dataset[
        dataset["fatigue_level"].isin(
            [
                "Moderate Fatigue",
                "Severe Fatigue"
            ]
        )
    ]

    if len(moderate_rows) < 6:
        raise ValueError(
            "Not enough Moderate/Severe rows were found "
            "to test the temporal workflow."
        )

    selected_rows = moderate_rows.head(6)

    print("=" * 70)
    print("DriverGuardianAI Fatigue Service Test")
    print("=" * 70)

    for index, (_, row) in enumerate(
        selected_rows.iterrows(),
        start=1
    ):
        actual_label = row[
            "fatigue_level"
        ]

        features = row[
            FEATURE_COLUMNS
        ].to_dict()

        result = service.process(
            features
        )

        print_result(
            index=index,
            actual_label=actual_label,
            result=result
        )

    print("\n" + "=" * 70)
    print("Fatigue Service test completed.")
    print("=" * 70)


if __name__ == "__main__":

    main()