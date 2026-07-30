"""
Test DriverGuardianAI predictions using real dataset samples.

This verifies that:
- the production predictor loads correctly;
- preprocessing matches training;
- predictions can be made on realistic feature combinations.
"""

import pandas as pd

from src.predict import Predictor


DATASET_PATH = "data/dataset_exp3.csv"

MODEL_PATH = "models/driver_guardian_best.pth"

PREPROCESSING_PATH = "models/preprocessing.pkl"

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


def print_prediction(
    sample_number,
    actual_label,
    sample,
    result
):
    """
    Print one prediction and its original feature values.
    """

    print("\n" + "=" * 70)

    print(
        f"Sample {sample_number}"
    )

    print("=" * 70)

    print(
        f"Actual label    : {actual_label}"
    )

    print(
        f"Predicted label : {result['prediction']}"
    )

    print(
        f"Predicted class : {result['predicted_class']}"
    )

    print(
        f"Confidence      : {result['confidence']:.2%}"
    )

    print("\nOriginal features:")

    for feature_name, value in sample.items():

        print(
            f"  {feature_name:<20} {value}"
        )

    print("\nClass probabilities:")

    for label, probability in result[
        "probabilities"
    ].items():

        print(
            f"  {label:<28} {probability:.4%}"
        )


def main():
    """
    Load the model and test it using real dataset rows.
    """

    predictor = Predictor(
        model_path=MODEL_PATH,
        preprocessing_path=PREPROCESSING_PATH,
        hidden_dims=[
            256,
            128,
            64
        ],
        dropout=0.30,
        num_classes=3
    )

    print("\nLoading dataset...")

    dataset = pd.read_csv(
        DATASET_PATH
    )

    missing_columns = [
        column
        for column in FEATURE_COLUMNS
        if column not in dataset.columns
    ]

    if missing_columns:

        raise ValueError(
            "Dataset is missing required features: "
            f"{missing_columns}"
        )

    if "fatigue_level" not in dataset.columns:

        raise ValueError(
            "The dataset does not contain fatigue_level."
        )

    print(
        f"Dataset loaded: {len(dataset)} samples"
    )

    print("\nCondition values found in the dataset:")

    print(
        dataset["condition"]
        .astype(str)
        .value_counts()
        .head(20)
    )

    # Select three reproducible real rows.
    #
    # One row is selected from the beginning,
    # one from the middle,
    # and one from the end of the dataset.

    selected_indices = [
        0,
        len(dataset) // 2,
        len(dataset) - 1
    ]

    for sample_number, row_index in enumerate(
        selected_indices,
        start=1
    ):

        row = dataset.iloc[
            row_index
        ]

        actual_label = row[
            "fatigue_level"
        ]

        sample = row[
            FEATURE_COLUMNS
        ].to_dict()

        result = predictor.predict(
            sample
        )

        print_prediction(
            sample_number=sample_number,
            actual_label=actual_label,
            sample=sample,
            result=result
        )

    print("\n" + "=" * 70)

    print(
        "Real-sample predictor test completed."
    )

    print("=" * 70)


if __name__ == "__main__":

    main()