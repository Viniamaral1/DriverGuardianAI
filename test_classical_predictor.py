"""
Test the calibrated classical predictor using real dataset samples.

Run from Jupyter:

    %run test_classical_predictor.py

Or from Anaconda Prompt:

    python test_classical_predictor.py
"""

import pandas as pd

from src.classical_predictor import ClassicalPredictor


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
    Convert the original three/four-class label into binary form.
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
    features,
    result,
):
    """
    Print one prediction result.
    """

    print("\n" + "=" * 72)
    print(f"Sample {sample_number}")
    print("=" * 72)

    print(
        f"Actual label        : {actual_label}"
    )

    print(
        f"Predicted label     : {result['prediction']}"
    )

    print(
        f"Predicted class     : {result['predicted_class']}"
    )

    print(
        f"Decision threshold  : {result['threshold']:.2f}"
    )

    print(
        f"Fatigue probability : "
        f"{result['fatigue_probability']:.2%}"
    )

    print(
        f"Alert probability   : "
        f"{result['alert_probability']:.2%}"
    )

    print(
        f"Confidence          : "
        f"{result['confidence']:.2%}"
    )

    print(
        f"Threshold margin    : "
        f"{result['probability_margin']:.4f}"
    )

    print("\nInput features:")

    for feature_name, value in features.items():
        print(
            f"  {feature_name:<20} {value}"
        )


def main():
    """
    Load the model and test real dataset rows.
    """

    predictor = ClassicalPredictor(
        model_path=(
            "models/"
            "driver_guardian_calibrated_hgb.joblib"
        )
    )

    print("\nModel information:")

    model_information = (
        predictor.model_information()
    )

    for key, value in model_information.items():
        print(
            f"{key}: {value}"
        )

    print("\nLoading dataset...")

    dataframe = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"Dataset loaded: {len(dataframe)} samples"
    )

    alert_rows = dataframe[
        dataframe["fatigue_level"]
        == "Alert"
    ]

    fatigue_rows = dataframe[
        dataframe["fatigue_level"].isin(
            [
                "Mild Fatigue",
                "Moderate Fatigue",
                "Severe Fatigue",
            ]
        )
    ]

    if alert_rows.empty:
        raise ValueError(
            "No Alert rows were found."
        )

    if fatigue_rows.empty:
        raise ValueError(
            "No Fatigue rows were found."
        )

    selected_rows = pd.concat(
        [
            alert_rows.sample(
                n=2,
                random_state=42,
            ),
            fatigue_rows.sample(
                n=2,
                random_state=42,
            ),
        ],
        ignore_index=True,
    )

    print("\nRunning individual predictions...")

    for sample_number, row in enumerate(
        selected_rows.itertuples(
            index=False
        ),
        start=1,
    ):
        row_series = pd.Series(
            row._asdict()
        )

        features = row_series[
            FEATURE_COLUMNS
        ].to_dict()

        actual_label = convert_actual_label(
            row_series["fatigue_level"]
        )

        result = predictor.predict(
            features
        )

        print_result(
            sample_number=sample_number,
            actual_label=actual_label,
            features=features,
            result=result,
        )

    print("\n" + "=" * 72)
    print("Testing batch prediction")
    print("=" * 72)

    batch_features = selected_rows[
        FEATURE_COLUMNS
    ]

    batch_results = predictor.predict_batch(
        batch_features
    )

    print(
        f"Batch predictions returned: "
        f"{len(batch_results)}"
    )

    for index, result in enumerate(
        batch_results,
        start=1,
    ):
        print(
            f"{index}. "
            f"{result['prediction']} | "
            f"Fatigue probability: "
            f"{result['fatigue_probability']:.2%}"
        )

    print("\n" + "=" * 72)
    print(
        "Classical Predictor test completed successfully."
    )
    print("=" * 72)


if __name__ == "__main__":
    main()