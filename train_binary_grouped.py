"""
Binary participant-aware training for DriverGuardianAI.

Experiment 10
-------------
This experiment converts the original fatigue target into:

    0 = Alert
    1 = Fatigue

The Fatigue class combines:

- Mild Fatigue
- Moderate Fatigue
- Severe Fatigue

The dataset is split by participant so that no participant appears
in both training and validation. This prevents participant leakage
and reduces temporal leakage between related video frames.

This experiment does not overwrite the existing three-class models.

Outputs
-------
models/driver_guardian_binary_grouped_best.pth
models/preprocessing_binary_grouped.pkl

results/experiment10_binary_participant_split/
    participant_split.json
    metrics.json
    classification_report.txt
    confusion_matrix.csv
    confusion_matrix.png
    roc_curve.png
    precision_recall_curve.png
    training_history.json
    training_history.png
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve
)

from sklearn.model_selection import GroupShuffleSplit

from src.config import load_config
from src.dataset import load_data

from src.preprocess import (
    fit_preprocessor,
    transform_dataset
)

from src.dataloader import create_dataloaders
from src.fatigue_model import FatigueResidualNN
from src.trainer import Trainer

from src.utils import (
    save_preprocessor,
    set_seed
)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

CONFIG_PATH = "config/config.yaml"

EXPERIMENT_NAME = (
    "experiment10_binary_participant_split"
)

MODEL_PATH = Path(
    "models/driver_guardian_binary_grouped_best.pth"
)

PREPROCESSING_PATH = Path(
    "models/preprocessing_binary_grouped.pkl"
)

RESULTS_DIRECTORY = Path(
    "results"
) / EXPERIMENT_NAME

SPLIT_INFORMATION_PATH = (
    RESULTS_DIRECTORY
    / "participant_split.json"
)

METRICS_PATH = (
    RESULTS_DIRECTORY
    / "metrics.json"
)

CLASSIFICATION_REPORT_PATH = (
    RESULTS_DIRECTORY
    / "classification_report.txt"
)

CONFUSION_MATRIX_CSV_PATH = (
    RESULTS_DIRECTORY
    / "confusion_matrix.csv"
)

CONFUSION_MATRIX_IMAGE_PATH = (
    RESULTS_DIRECTORY
    / "confusion_matrix.png"
)

ROC_CURVE_PATH = (
    RESULTS_DIRECTORY
    / "roc_curve.png"
)

PRECISION_RECALL_CURVE_PATH = (
    RESULTS_DIRECTORY
    / "precision_recall_curve.png"
)

TRAINING_HISTORY_JSON_PATH = (
    RESULTS_DIRECTORY
    / "training_history.json"
)

TRAINING_HISTORY_IMAGE_PATH = (
    RESULTS_DIRECTORY
    / "training_history.png"
)

VALIDATION_PARTICIPANT_FRACTION = 0.25

TARGET_COLUMN = "fatigue_level"

CLASS_NAMES = [
    "Alert",
    "Fatigue"
]


# ============================================================
# PARTICIPANT EXTRACTION
# ============================================================

def extract_participant(source_file):
    """
    Extract participant name from source_file.

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


def add_participant_column(dataframe):
    """
    Add a participant column for group-aware splitting.
    """

    dataframe = dataframe.copy()

    if "source_file" not in dataframe.columns:
        raise ValueError(
            "The dataset does not contain source_file."
        )

    dataframe["participant"] = dataframe[
        "source_file"
    ].apply(
        extract_participant
    )

    return dataframe


# ============================================================
# BINARY TARGET
# ============================================================

def create_binary_target(dataframe):
    """
    Convert the original target into Alert versus Fatigue.

    Original labels
    ---------------
    Alert              -> Alert
    Mild Fatigue       -> Fatigue
    Moderate Fatigue   -> Fatigue
    Severe Fatigue     -> Fatigue
    """

    dataframe = dataframe.copy()

    if TARGET_COLUMN not in dataframe.columns:
        raise ValueError(
            f"The dataset does not contain {TARGET_COLUMN}."
        )

    label_mapping = {
        "Alert": "Alert",
        "Mild Fatigue": "Fatigue",
        "Moderate Fatigue": "Fatigue",
        "Severe Fatigue": "Fatigue"
    }

    original_values = set(
        dataframe[
            TARGET_COLUMN
        ].dropna().astype(str).unique()
    )

    unknown_values = original_values.difference(
        label_mapping.keys()
    )

    if unknown_values:
        raise ValueError(
            "Unknown fatigue labels were found: "
            f"{sorted(unknown_values)}"
        )

    dataframe[TARGET_COLUMN] = (
        dataframe[TARGET_COLUMN]
        .astype(str)
        .map(label_mapping)
    )

    if dataframe[TARGET_COLUMN].isna().any():
        raise ValueError(
            "Some target values could not be converted "
            "to the binary target."
        )

    return dataframe


# ============================================================
# GROUP SPLIT
# ============================================================

def create_participant_split(
    dataframe,
    random_state
):
    """
    Split the dataset by participant.

    No participant can appear in both training and validation.
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
# CLASS WEIGHTS
# ============================================================

def calculate_binary_class_weights(
    labels,
    device
):
    """
    Calculate balanced binary class weights from training labels.

    The returned tensor order matches:

        0 = Alert
        1 = Fatigue
    """

    label_array = np.asarray(
        labels,
        dtype=int
    )

    class_counts = np.bincount(
        label_array,
        minlength=2
    )

    if np.any(class_counts == 0):
        raise ValueError(
            "Both binary classes must be present "
            "in the training set."
        )

    total_samples = class_counts.sum()
    number_of_classes = len(class_counts)

    weights = (
        total_samples
        / (
            number_of_classes
            * class_counts
        )
    )

    weight_tensor = torch.tensor(
        weights,
        dtype=torch.float32,
        device=device
    )

    print("\nBinary class counts:")

    print(
        f"Alert   : {class_counts[0]}"
    )

    print(
        f"Fatigue : {class_counts[1]}"
    )

    print("\nCalculated class weights:")

    print(
        weight_tensor
    )

    return weight_tensor


# ============================================================
# SPLIT REPORT
# ============================================================

def print_split_report(
    train_dataframe,
    validation_dataframe,
    train_participants,
    validation_participants
):
    """
    Print participant and binary class distributions.
    """

    print("\n" + "=" * 60)
    print("Binary participant-aware split")
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

    print("\nTraining binary class distribution:")

    print(
        train_dataframe[
            TARGET_COLUMN
        ].value_counts()
    )

    print("\nValidation binary class distribution:")

    print(
        validation_dataframe[
            TARGET_COLUMN
        ].value_counts()
    )

    print(
        "\nTraining recordings: "
        f"{train_dataframe['source_file'].nunique()}"
    )

    print(
        "Validation recordings: "
        f"{validation_dataframe['source_file'].nunique()}"
    )


def save_split_information(
    train_dataframe,
    validation_dataframe,
    train_participants,
    validation_participants
):
    """
    Save the exact split for reproducibility.
    """

    information = {
        "experiment": EXPERIMENT_NAME,
        "task": "binary_alert_vs_fatigue",
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
                TARGET_COLUMN
            ].value_counts().items()
        },
        "validation_class_distribution": {
            str(key): int(value)
            for key, value
            in validation_dataframe[
                TARGET_COLUMN
            ].value_counts().items()
        }
    }

    with SPLIT_INFORMATION_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            information,
            file,
            indent=4
        )


# ============================================================
# BINARY EVALUATION
# ============================================================

def collect_binary_predictions(
    model,
    validation_loader,
    criterion,
    device
):
    """
    Collect binary predictions and Fatigue probabilities.
    """

    model.eval()

    total_loss = 0.0
    total_batches = 0

    true_labels = []
    predicted_labels = []
    fatigue_probabilities = []

    with torch.inference_mode():

        for features, labels in validation_loader:

            features = features.to(
                device
            )

            labels = labels.to(
                device
            )

            logits = model(
                features
            )

            loss = criterion(
                logits,
                labels
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            total_loss += float(
                loss.item()
            )

            total_batches += 1

            true_labels.extend(
                labels.cpu().numpy().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().numpy().tolist()
            )

            fatigue_probabilities.extend(
                probabilities[
                    :,
                    1
                ].cpu().numpy().tolist()
            )

    if total_batches == 0:
        raise ValueError(
            "The validation loader contained no batches."
        )

    average_loss = (
        total_loss / total_batches
    )

    return (
        average_loss,
        np.asarray(
            true_labels,
            dtype=int
        ),
        np.asarray(
            predicted_labels,
            dtype=int
        ),
        np.asarray(
            fatigue_probabilities,
            dtype=float
        )
    )


def calculate_binary_metrics(
    validation_loss,
    y_true,
    y_pred,
    fatigue_probabilities
):
    """
    Calculate binary classification metrics.
    """

    matrix = confusion_matrix(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ]
    )

    true_negative = int(
        matrix[0, 0]
    )

    false_positive = int(
        matrix[0, 1]
    )

    false_negative = int(
        matrix[1, 0]
    )

    true_positive = int(
        matrix[1, 1]
    )

    sensitivity = (
        true_positive
        / (
            true_positive
            + false_negative
        )
        if (
            true_positive
            + false_negative
        ) > 0
        else 0.0
    )

    specificity = (
        true_negative
        / (
            true_negative
            + false_positive
        )
        if (
            true_negative
            + false_positive
        ) > 0
        else 0.0
    )

    false_positive_rate = (
        false_positive
        / (
            false_positive
            + true_negative
        )
        if (
            false_positive
            + true_negative
        ) > 0
        else 0.0
    )

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "task": "Alert vs Fatigue",
        "samples": int(
            len(y_true)
        ),
        "validation_loss": float(
            validation_loss
        ),
        "accuracy": float(
            accuracy_score(
                y_true,
                y_pred
            )
        ),
        "precision": float(
            precision_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "recall_sensitivity": float(
            sensitivity
        ),
        "specificity": float(
            specificity
        ),
        "f1_score": float(
            f1_score(
                y_true,
                y_pred,
                pos_label=1,
                zero_division=0
            )
        ),
        "roc_auc": float(
            roc_auc_score(
                y_true,
                fatigue_probabilities
            )
        ),
        "average_precision": float(
            average_precision_score(
                y_true,
                fatigue_probabilities
            )
        ),
        "false_positive_rate": float(
            false_positive_rate
        ),
        "true_negative": true_negative,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "true_positive": true_positive
    }

    return (
        metrics,
        matrix
    )


# ============================================================
# SAVE EVALUATION OUTPUTS
# ============================================================

def save_metrics(metrics):
    """
    Save metrics as JSON.
    """

    with METRICS_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            metrics,
            file,
            indent=4
        )


def save_classification_report(
    y_true,
    y_pred
):
    """
    Save and return the binary classification report.
    """

    report = classification_report(
        y_true,
        y_pred,
        labels=[
            0,
            1
        ],
        target_names=CLASS_NAMES,
        zero_division=0
    )

    with CLASSIFICATION_REPORT_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            report
        )

    return report


def save_confusion_matrix(
    matrix
):
    """
    Save binary confusion matrix as CSV and PNG.
    """

    matrix_dataframe = pd.DataFrame(
        matrix,
        index=[
            "Actual Alert",
            "Actual Fatigue"
        ],
        columns=[
            "Predicted Alert",
            "Predicted Fatigue"
        ]
    )

    matrix_dataframe.to_csv(
        CONFUSION_MATRIX_CSV_PATH
    )

    plt.figure(
        figsize=(
            7,
            6
        )
    )

    plt.imshow(
        matrix
    )

    plt.colorbar()

    plt.xticks(
        [
            0,
            1
        ],
        CLASS_NAMES
    )

    plt.yticks(
        [
            0,
            1
        ],
        CLASS_NAMES
    )

    for row_index in range(
        2
    ):

        for column_index in range(
            2
        ):

            plt.text(
                column_index,
                row_index,
                str(
                    matrix[
                        row_index,
                        column_index
                    ]
                ),
                ha="center",
                va="center",
                fontsize=12
            )

    plt.xlabel(
        "Predicted class"
    )

    plt.ylabel(
        "Actual class"
    )

    plt.title(
        "Binary Participant-Aware Confusion Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_IMAGE_PATH,
        dpi=300
    )

    plt.close()


def save_roc_curve(
    y_true,
    fatigue_probabilities
):
    """
    Save ROC curve.
    """

    false_positive_rates, true_positive_rates, _ = (
        roc_curve(
            y_true,
            fatigue_probabilities
        )
    )

    auc_value = roc_auc_score(
        y_true,
        fatigue_probabilities
    )

    plt.figure(
        figsize=(
            8,
            6
        )
    )

    plt.plot(
        false_positive_rates,
        true_positive_rates,
        label=(
            f"ROC AUC = {auc_value:.3f}"
        )
    )

    plt.plot(
        [
            0,
            1
        ],
        [
            0,
            1
        ],
        linestyle="--",
        label="Random baseline"
    )

    plt.xlabel(
        "False Positive Rate"
    )

    plt.ylabel(
        "True Positive Rate"
    )

    plt.title(
        "Binary Alert vs Fatigue ROC Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        ROC_CURVE_PATH,
        dpi=300
    )

    plt.close()


def save_precision_recall_curve(
    y_true,
    fatigue_probabilities
):
    """
    Save binary precision-recall curve.
    """

    precision_values, recall_values, _ = (
        precision_recall_curve(
            y_true,
            fatigue_probabilities
        )
    )

    average_precision = (
        average_precision_score(
            y_true,
            fatigue_probabilities
        )
    )

    plt.figure(
        figsize=(
            8,
            6
        )
    )

    plt.plot(
        recall_values,
        precision_values,
        label=(
            "Average precision = "
            f"{average_precision:.3f}"
        )
    )

    plt.xlabel(
        "Recall"
    )

    plt.ylabel(
        "Precision"
    )

    plt.title(
        "Binary Alert vs Fatigue Precision-Recall Curve"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        PRECISION_RECALL_CURVE_PATH,
        dpi=300
    )

    plt.close()


def save_training_history(
    history
):
    """
    Save training history as JSON and PNG.
    """

    serialisable_history = {
        key: [
            float(value)
            for value in values
        ]
        for key, values in history.items()
    }

    with TRAINING_HISTORY_JSON_PATH.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            serialisable_history,
            file,
            indent=4
        )

    plt.figure(
        figsize=(
            9,
            6
        )
    )

    plt.plot(
        history[
            "train_loss"
        ],
        label="Training Loss"
    )

    plt.plot(
        history[
            "val_loss"
        ],
        label="Validation Loss"
    )

    plt.xlabel(
        "Epoch"
    )

    plt.ylabel(
        "Loss"
    )

    plt.title(
        "Binary Participant-Aware Training History"
    )

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        TRAINING_HISTORY_IMAGE_PATH,
        dpi=300
    )

    plt.close()


# ============================================================
# MAIN PIPELINE
# ============================================================

def main():
    """
    Run Experiment 10.
    """

    print("=" * 60)
    print("DriverGuardianAI")
    print("Experiment 10: Binary Participant Split")
    print("=" * 60)

    # --------------------------------------------------
    # Directories
    # --------------------------------------------------

    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    PREPROCESSING_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    RESULTS_DIRECTORY.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------
    # Configuration and seed
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
    # Load and prepare dataset
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
        f"Original dataset shape: "
        f"{dataframe.shape}"
    )

    dataframe = add_participant_column(
        dataframe
    )

    dataframe = create_binary_target(
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

    print(
        "\nComplete binary class distribution:"
    )

    print(
        dataframe[
            TARGET_COLUMN
        ].value_counts()
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

    print_split_report(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        train_participants=train_participants,
        validation_participants=validation_participants
    )

    save_split_information(
        train_dataframe=train_dataframe,
        validation_dataframe=validation_dataframe,
        train_participants=train_participants,
        validation_participants=validation_participants
    )

    # --------------------------------------------------
    # Remove helper participant column
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
    # Fit preprocessing only on training participants
    # --------------------------------------------------

    print(
        "\nFitting binary preprocessing on "
        "training participants only..."
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

    save_preprocessor(
        preprocessing,
        str(
            PREPROCESSING_PATH
        )
    )

    print(
        "\nBinary preprocessing saved to: "
        f"{PREPROCESSING_PATH}"
    )

    # --------------------------------------------------
    # Features and labels
    # --------------------------------------------------

    X_train = train_dataframe.drop(
        TARGET_COLUMN,
        axis=1
    )

    y_train = train_dataframe[
        TARGET_COLUMN
    ].astype(int)

    X_validation = validation_dataframe.drop(
        TARGET_COLUMN,
        axis=1
    )

    y_validation = validation_dataframe[
        TARGET_COLUMN
    ].astype(int)

    print("\nModel feature order:")

    print(
        X_train.columns.tolist()
    )

    print(
        f"\nInput dimensions: "
        f"{X_train.shape[1]}"
    )

    print("\nEncoded target classes:")

    print(
        preprocessing[
            "target_encoder"
        ].classes_
    )

    expected_classes = [
        "Alert",
        "Fatigue"
    ]

    encoded_classes = (
        preprocessing[
            "target_encoder"
        ].classes_.tolist()
    )

    if encoded_classes != expected_classes:
        raise ValueError(
            "Unexpected target encoding order. "
            f"Expected {expected_classes}, "
            f"received {encoded_classes}."
        )

    # --------------------------------------------------
    # Binary class weights
    # --------------------------------------------------

    class_weights = (
        calculate_binary_class_weights(
            y_train,
            device
        )
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
    # Binary model
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
        num_classes=2
    )

    model.to(
        device
    )

    print(
        "\nBinary model created."
    )

    # --------------------------------------------------
    # Loss
    #
    # Cross-entropy provides a clean binary baseline.
    # --------------------------------------------------

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    print(
        "Using weighted CrossEntropyLoss."
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
        "\nStarting binary participant-aware training...\n"
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
        save_path=str(
            MODEL_PATH
        )
    )

    # --------------------------------------------------
    # Load best model
    # --------------------------------------------------

    print(
        "\nLoading best binary model..."
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
    # Binary evaluation
    # --------------------------------------------------

    print(
        "\nEvaluating held-out participants..."
    )

    (
        validation_loss,
        y_true,
        y_pred,
        fatigue_probabilities
    ) = collect_binary_predictions(
        model=model,
        validation_loader=validation_loader,
        criterion=criterion,
        device=device
    )

    (
        metrics,
        matrix
    ) = calculate_binary_metrics(
        validation_loss=validation_loss,
        y_true=y_true,
        y_pred=y_pred,
        fatigue_probabilities=(
            fatigue_probabilities
        )
    )

    report = save_classification_report(
        y_true,
        y_pred
    )

    save_metrics(
        metrics
    )

    save_confusion_matrix(
        matrix
    )

    save_roc_curve(
        y_true,
        fatigue_probabilities
    )

    save_precision_recall_curve(
        y_true,
        fatigue_probabilities
    )

    save_training_history(
        trainer.history
    )

    # --------------------------------------------------
    # Final output
    # --------------------------------------------------

    print("\nConfusion Matrix:")

    print(
        matrix
    )

    print("\nClassification Report:")

    print(
        report
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "Experiment 10 completed successfully!"
    )

    print(
        "=" * 60
    )

    print(
        "\nFinal binary participant-aware metrics:"
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
        "Preprocessing saved to: "
        f"{PREPROCESSING_PATH}"
    )

    print(
        f"Results saved to: "
        f"{RESULTS_DIRECTORY}"
    )


if __name__ == "__main__":

    main()