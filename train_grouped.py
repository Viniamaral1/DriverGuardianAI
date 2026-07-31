"""
Participant-aware training experiment for DriverGuardianAI.

Experiment 9
------------
This experiment avoids participant and temporal leakage by ensuring
that every participant belongs entirely to either:

- the training set; or
- the validation set.

It does not replace the existing Experiment 8 model.

Outputs
-------
models/driver_guardian_grouped_best.pth
models/preprocessing_grouped.pkl
results/experiment9_participant_group_split/
"""

import json
import os
from pathlib import Path

import pandas as pd
import torch
import torch.optim as optim

from sklearn.model_selection import GroupShuffleSplit

from src.config import load_config
from src.dataset import load_data
from src.preprocess import (
    fit_preprocessor,
    transform_dataset
)
from src.dataloader import create_dataloaders
from src.fatigue_model import FatigueResidualNN
from src.losses import get_loss
from src.trainer import Trainer
from src.evaluate import evaluate_model
from src.utils import (
    save_preprocessor,
    set_seed
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

CONFIG_PATH = "config/config.yaml"

EXPERIMENT_NAME = (
    "experiment9_participant_group_split"
)

MODEL_PATH = (
    "models/driver_guardian_grouped_best.pth"
)

PREPROCESSING_PATH = (
    "models/preprocessing_grouped.pkl"
)

SPLIT_INFORMATION_PATH = (
    "results/"
    f"{EXPERIMENT_NAME}/participant_split.json"
)

# With eight participants, 25% produces approximately
# six training participants and two validation participants.
VALIDATION_PARTICIPANT_FRACTION = 0.25


# ============================================================
# PARTICIPANT EXTRACTION
# ============================================================

def extract_participant(
    source_file
):
    """
    Extract the participant name from source_file.

    Example
    -------
    vinicius_normal_none_123.csv -> vinicius
    """

    if pd.isna(source_file):
        raise ValueError(
            "A missing source_file value was found."
        )

    source_file = str(
        source_file
    ).strip()

    if not source_file:
        raise ValueError(
            "An empty source_file value was found."
        )

    participant = source_file.split(
        "_",
        maxsplit=1
    )[0].lower()

    return participant


def add_participant_column(
    dataframe
):
    """
    Add a participant column without modifying source_file.
    """

    dataframe = dataframe.copy()

    if "source_file" not in dataframe.columns:
        raise ValueError(
            "The dataset does not contain source_file. "
            "Participant-aware splitting cannot be performed."
        )

    dataframe["participant"] = dataframe[
        "source_file"
    ].apply(
        extract_participant
    )

    return dataframe


# ============================================================
# GROUP-AWARE SPLIT
# ============================================================

def create_participant_split(
    dataframe,
    random_state
):
    """
    Split the dataset by participant.

    No participant can appear in both sets.
    """

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_PARTICIPANT_FRACTION,
        random_state=random_state
    )

    groups = dataframe[
        "participant"
    ]

    train_indices, validation_indices = next(
        splitter.split(
            dataframe,
            groups=groups
        )
    )

    train_dataframe = dataframe.iloc[
        train_indices
    ].copy()

    validation_dataframe = dataframe.iloc[
        validation_indices
    ].copy()

    train_participants = sorted(
        train_dataframe[
            "participant"
        ].unique().tolist()
    )

    validation_participants = sorted(
        validation_dataframe[
            "participant"
        ].unique().tolist()
    )

    participant_overlap = set(
        train_participants
    ).intersection(
        validation_participants
    )

    if participant_overlap:
        raise RuntimeError(
            "Participant leakage detected: "
            f"{sorted(participant_overlap)}"
        )

    return (
        train_dataframe,
        validation_dataframe,
        train_participants,
        validation_participants
    )


# ============================================================
# SPLIT REPORT
# ============================================================

def print_split_report(
    train_dataframe,
    validation_dataframe,
    train_participants,
    validation_participants,
    target_column
):
    """
    Print participant and class distributions.
    """

    print("\n" + "=" * 60)
    print("Participant-aware split")
    print("=" * 60)

    print("\nTraining participants:")

    for participant in train_participants:
        print(
            f"  - {participant}"
        )

    print("\nValidation participants:")

    for participant in validation_participants:
        print(
            f"  - {participant}"
        )

    print(
        f"\nTraining samples: "
        f"{len(train_dataframe)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataframe)}"
    )

    print("\nTraining class distribution:")

    print(
        train_dataframe[
            target_column
        ].value_counts()
    )

    print("\nValidation class distribution:")

    print(
        validation_dataframe[
            target_column
        ].value_counts()
    )

    print("\nTraining recordings:")

    print(
        train_dataframe[
            "source_file"
        ].nunique()
    )

    print("\nValidation recordings:")

    print(
        validation_dataframe[
            "source_file"
        ].nunique()
    )


def save_split_information(
    train_dataframe,
    validation_dataframe,
    train_participants,
    validation_participants,
    target_column
):
    """
    Save the exact participant split for reproducibility.
    """

    filepath = Path(
        SPLIT_INFORMATION_PATH
    )

    filepath.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    information = {
        "experiment": EXPERIMENT_NAME,
        "split_type": "participant_group_split",
        "validation_participant_fraction": (
            VALIDATION_PARTICIPANT_FRACTION
        ),
        "training_participants": (
            train_participants
        ),
        "validation_participants": (
            validation_participants
        ),
        "training_samples": int(
            len(train_dataframe)
        ),
        "validation_samples": int(
            len(validation_dataframe)
        ),
        "training_recordings": int(
            train_dataframe[
                "source_file"
            ].nunique()
        ),
        "validation_recordings": int(
            validation_dataframe[
                "source_file"
            ].nunique()
        ),
        "training_class_distribution": {
            str(key): int(value)
            for key, value
            in train_dataframe[
                target_column
            ].value_counts().items()
        },
        "validation_class_distribution": {
            str(key): int(value)
            for key, value
            in validation_dataframe[
                target_column
            ].value_counts().items()
        }
    }

    with filepath.open(
        "w",
        encoding="utf-8"
    ) as file:
        json.dump(
            information,
            file,
            indent=4
        )

    print(
        f"\nParticipant split saved to: "
        f"{filepath}"
    )


# ============================================================
# MAIN TRAINING PIPELINE
# ============================================================

def main():
    """
    Run Experiment 9.
    """

    print("=" * 60)
    print("DriverGuardianAI")
    print("Experiment 9: Participant Group Split")
    print("=" * 60)

    # --------------------------------------------------
    # Configuration and reproducibility
    # --------------------------------------------------

    config = load_config(
        CONFIG_PATH
    )

    seed = config[
        "training"
    ].get(
        "seed",
        42
    )

    set_seed(
        seed
    )

    print("\nConfiguration loaded.")

    # --------------------------------------------------
    # Device
    # --------------------------------------------------

    device = (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nUsing device: {device}"
    )

    # --------------------------------------------------
    # Directories
    # --------------------------------------------------

    Path(
        MODEL_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    Path(
        PREPROCESSING_PATH
    ).parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Load dataset
    # --------------------------------------------------

    print("\nLoading dataset...")

    dataframe = load_data(
        config[
            "dataset"
        ][
            "path"
        ]
    )

    print(
        f"Dataset shape: {dataframe.shape}"
    )

    dataframe = add_participant_column(
        dataframe
    )

    participants = sorted(
        dataframe[
            "participant"
        ].unique().tolist()
    )

    print(
        f"Participants detected: "
        f"{len(participants)}"
    )

    print(
        participants
    )

    # --------------------------------------------------
    # Participant-aware split
    # --------------------------------------------------

    (
        train_dataframe,
        validation_dataframe,
        train_participants,
        validation_participants
    ) = create_participant_split(
        dataframe=dataframe,
        random_state=config[
            "dataset"
        ][
            "random_state"
        ]
    )

    target_column = config[
        "dataset"
    ][
        "target"
    ]

    print_split_report(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        train_participants=train_participants,
        validation_participants=validation_participants,
        target_column=target_column
    )

    save_split_information(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        train_participants=train_participants,
        validation_participants=validation_participants,
        target_column=target_column
    )

    # --------------------------------------------------
    # Remove helper column
    #
    # source_file will already be removed by preprocess.py.
    # participant is only used for splitting and must not
    # become a model feature.
    # --------------------------------------------------

    train_dataframe.drop(
        columns=[
            "participant"
        ],
        inplace=True
    )

    validation_dataframe.drop(
        columns=[
            "participant"
        ],
        inplace=True
    )

    # --------------------------------------------------
    # Fit preprocessing on training participants only
    # --------------------------------------------------

    print(
        "\nFitting preprocessing on training "
        "participants only..."
    )

    train_dataframe, preprocessing = (
        fit_preprocessor(
            train_dataframe
        )
    )

    print(
        "Training preprocessing complete."
    )

    print(
        "\nTransforming validation participants..."
    )

    validation_dataframe = transform_dataset(
        validation_dataframe,
        preprocessing
    )

    print(
        "Validation transformation complete."
    )

    # --------------------------------------------------
    # Save grouped preprocessing pipeline
    # --------------------------------------------------

    save_preprocessor(
        preprocessing,
        PREPROCESSING_PATH
    )

    print(
        f"\nGrouped preprocessing saved to: "
        f"{PREPROCESSING_PATH}"
    )

    # --------------------------------------------------
    # Features and labels
    # --------------------------------------------------

    X_train = train_dataframe.drop(
        target_column,
        axis=1
    )

    y_train = train_dataframe[
        target_column
    ]

    X_validation = validation_dataframe.drop(
        target_column,
        axis=1
    )

    y_validation = validation_dataframe[
        target_column
    ]

    print("\nModel feature order:")

    print(
        X_train.columns.tolist()
    )

    print(
        f"\nInput dimensions: "
        f"{X_train.shape[1]}"
    )

    # --------------------------------------------------
    # Controlled class weighting
    #
    # Keep the same values as Experiment 8 so the split
    # strategy is the main experimental change.
    # --------------------------------------------------

    class_weights = torch.tensor(
        [
            1.0,
            1.2,
            1.8
        ],
        dtype=torch.float32,
        device=device
    )

    print(
        "\nClass weights:"
    )

    print(
        class_weights
    )

    # --------------------------------------------------
    # DataLoaders
    # --------------------------------------------------

    train_loader, validation_loader = (
        create_dataloaders(
            X_train,
            X_validation,
            y_train,
            y_validation,
            sampler=None,
            batch_size=config[
                "training"
            ][
                "batch_size"
            ]
        )
    )

    print(
        "\nDataLoaders created."
    )

    # --------------------------------------------------
    # Model
    # --------------------------------------------------

    model = FatigueResidualNN(
        input_dim=X_train.shape[1],
        hidden_dims=config[
            "model"
        ][
            "hidden_dims"
        ],
        dropout=config[
            "model"
        ][
            "dropout"
        ],
        num_classes=config[
            "model"
        ][
            "num_classes"
        ]
    )

    model.to(
        device
    )

    print(
        "\nModel created."
    )

    # --------------------------------------------------
    # Loss
    # --------------------------------------------------

    criterion = get_loss(
        config[
            "loss"
        ][
            "type"
        ],
        class_weights
    )

    # --------------------------------------------------
    # Optimizer
    # --------------------------------------------------

    optimizer = optim.Adam(
        model.parameters(),
        lr=config[
            "training"
        ][
            "learning_rate"
        ],
        weight_decay=config[
            "training"
        ].get(
            "weight_decay",
            0.0
        )
    )

    # --------------------------------------------------
    # Scheduler
    # --------------------------------------------------

    scheduler = None

    scheduler_config = config.get(
        "scheduler",
        {}
    )

    if scheduler_config.get(
        "enabled",
        False
    ):
        scheduler = (
            optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode="min",
                factor=scheduler_config.get(
                    "factor",
                    0.5
                ),
                patience=scheduler_config.get(
                    "patience",
                    5
                ),
                min_lr=scheduler_config.get(
                    "min_lr",
                    1e-6
                )
            )
        )

        print(
            "Learning-rate scheduler enabled."
        )

    # --------------------------------------------------
    # Trainer
    # --------------------------------------------------

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device
    )

    # --------------------------------------------------
    # Training
    # --------------------------------------------------

    print(
        "\nStarting participant-aware training...\n"
    )

    trainer.fit(
        epochs=config[
            "training"
        ][
            "epochs"
        ],
        patience=config[
            "training"
        ][
            "patience"
        ],
        save_path=MODEL_PATH
    )

    # --------------------------------------------------
    # Load best grouped model
    # --------------------------------------------------

    print(
        "\nLoading best grouped model..."
    )

    model.load_state_dict(
        torch.load(
            MODEL_PATH,
            map_location=device
        )
    )

    model.to(
        device
    )

    model.eval()

    # --------------------------------------------------
    # Evaluation
    # --------------------------------------------------

    print(
        "\nEvaluating held-out participants..."
    )

    (
        metrics,
        report,
        y_true,
        y_pred
    ) = evaluate_model(
        trainer,
        experiment_name=EXPERIMENT_NAME
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "Experiment 9 completed successfully!"
    )

    print(
        "=" * 60
    )

    print(
        "\nFinal participant-aware metrics:"
    )

    for key, value in metrics.items():
        print(
            f"{key}: {value}"
        )

    print(
        f"\nModel saved to: "
        f"{MODEL_PATH}"
    )

    print(
        f"Preprocessing saved to: "
        f"{PREPROCESSING_PATH}"
    )


if __name__ == "__main__":
    main()