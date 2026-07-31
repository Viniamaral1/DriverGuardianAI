"""
DriverGuardianAI Experiment 11: binary participant-aware model
without the hands_detected feature.

Creates:
- models/driver_guardian_binary_no_hands_best.pth
- models/preprocessing_binary_no_hands.pkl
- results/experiment11_binary_no_hands/
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
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import GroupShuffleSplit

from src.config import load_config
from src.dataloader import create_dataloaders
from src.dataset import load_data
from src.fatigue_model import FatigueResidualNN
from src.preprocess import fit_preprocessor, transform_dataset
from src.trainer import Trainer
from src.utils import save_preprocessor, set_seed


CONFIG_PATH = "config/config.yaml"
EXPERIMENT_NAME = "experiment11_binary_no_hands"
MODEL_PATH = Path("models/driver_guardian_binary_no_hands_best.pth")
PREPROCESSING_PATH = Path("models/preprocessing_binary_no_hands.pkl")
RESULTS_DIR = Path("results") / EXPERIMENT_NAME
TARGET_COLUMN = "fatigue_level"
REMOVED_FEATURE = "hands_detected"
CLASS_NAMES = ["Alert", "Fatigue"]
VALIDATION_FRACTION = 0.25


def extract_participant(source_file):
    if pd.isna(source_file):
        raise ValueError("Missing source_file value found.")
    source_file = str(source_file).strip()
    if not source_file:
        raise ValueError("Empty source_file value found.")
    return source_file.split("_", maxsplit=1)[0].lower()


def prepare_dataset(dataframe):
    dataframe = dataframe.copy()

    if "source_file" not in dataframe.columns:
        raise ValueError("Dataset does not contain source_file.")

    dataframe["participant"] = dataframe["source_file"].apply(
        extract_participant
    )

    mapping = {
        "Alert": "Alert",
        "Mild Fatigue": "Fatigue",
        "Moderate Fatigue": "Fatigue",
        "Severe Fatigue": "Fatigue",
    }

    unknown = set(
        dataframe[TARGET_COLUMN].dropna().astype(str).unique()
    ).difference(mapping)

    if unknown:
        raise ValueError(f"Unknown labels found: {sorted(unknown)}")

    dataframe[TARGET_COLUMN] = (
        dataframe[TARGET_COLUMN].astype(str).map(mapping)
    )

    if REMOVED_FEATURE not in dataframe.columns:
        raise ValueError(f"Dataset does not contain {REMOVED_FEATURE}.")

    dataframe.drop(columns=[REMOVED_FEATURE], inplace=True)
    return dataframe


def participant_split(dataframe, random_state):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=VALIDATION_FRACTION,
        random_state=random_state,
    )

    train_idx, val_idx = next(
        splitter.split(dataframe, groups=dataframe["participant"])
    )

    train_df = dataframe.iloc[train_idx].copy()
    val_df = dataframe.iloc[val_idx].copy()

    train_people = sorted(train_df["participant"].unique().tolist())
    val_people = sorted(val_df["participant"].unique().tolist())

    overlap = set(train_people).intersection(val_people)
    if overlap:
        raise RuntimeError(f"Participant leakage detected: {sorted(overlap)}")

    return train_df, val_df, train_people, val_people


def calculate_class_weights(y_train, device):
    labels = np.asarray(y_train, dtype=int)
    counts = np.bincount(labels, minlength=2)

    if np.any(counts == 0):
        raise ValueError("Both binary classes must be present in training.")

    weights = counts.sum() / (2 * counts)

    print("\nBinary class counts:")
    print(f"Alert   : {counts[0]}")
    print(f"Fatigue : {counts[1]}")

    tensor = torch.tensor(weights, dtype=torch.float32, device=device)
    print("\nCalculated class weights:")
    print(tensor)
    return tensor


def collect_predictions(model, loader, criterion, device):
    model.eval()
    losses = []
    y_true = []
    y_pred = []
    fatigue_probs = []

    with torch.inference_mode():
        for features, labels in loader:
            features = features.to(device)
            labels = labels.to(device)

            logits = model(features)
            loss = criterion(logits, labels)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1)

            losses.append(float(loss.item()))
            y_true.extend(labels.cpu().numpy().tolist())
            y_pred.extend(predictions.cpu().numpy().tolist())
            fatigue_probs.extend(probabilities[:, 1].cpu().numpy().tolist())

    if not losses:
        raise ValueError("Validation loader contained no batches.")

    return (
        float(np.mean(losses)),
        np.asarray(y_true, dtype=int),
        np.asarray(y_pred, dtype=int),
        np.asarray(fatigue_probs, dtype=float),
    )


def calculate_metrics(val_loss, y_true, y_pred, fatigue_probs):
    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()

    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    false_positive_rate = fp / (fp + tn) if fp + tn else 0.0

    metrics = {
        "experiment": EXPERIMENT_NAME,
        "task": "Alert vs Fatigue",
        "removed_feature": REMOVED_FEATURE,
        "samples": int(len(y_true)),
        "validation_loss": float(val_loss),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_score": float(
            f1_score(y_true, y_pred, pos_label=1, zero_division=0)
        ),
        "roc_auc": float(roc_auc_score(y_true, fatigue_probs)),
        "average_precision": float(
            average_precision_score(y_true, fatigue_probs)
        ),
        "false_positive_rate": float(false_positive_rate),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }

    return metrics, matrix


def save_outputs(metrics, matrix, report, y_true, fatigue_probs, history):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with (RESULTS_DIR / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    with (RESULTS_DIR / "classification_report.txt").open(
        "w", encoding="utf-8"
    ) as file:
        file.write(report)

    pd.DataFrame(
        matrix,
        index=["Actual Alert", "Actual Fatigue"],
        columns=["Predicted Alert", "Predicted Fatigue"],
    ).to_csv(RESULTS_DIR / "confusion_matrix.csv")

    plt.figure(figsize=(7, 6))
    plt.imshow(matrix)
    plt.colorbar()
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    for row in range(2):
        for column in range(2):
            plt.text(column, row, str(matrix[row, column]), ha="center", va="center")
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title("Binary Participant Split — No Hands")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "confusion_matrix.png", dpi=300)
    plt.close()

    fpr, tpr, _ = roc_curve(y_true, fatigue_probs)
    auc_value = roc_auc_score(y_true, fatigue_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC AUC = {auc_value:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", label="Random baseline")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Alert vs Fatigue ROC — No Hands")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "roc_curve.png", dpi=300)
    plt.close()

    precision_values, recall_values, _ = precision_recall_curve(
        y_true, fatigue_probs
    )
    average_precision = average_precision_score(y_true, fatigue_probs)
    plt.figure(figsize=(8, 6))
    plt.plot(
        recall_values,
        precision_values,
        label=f"Average precision = {average_precision:.3f}",
    )
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Alert vs Fatigue Precision-Recall — No Hands")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "precision_recall_curve.png", dpi=300)
    plt.close()

    serialisable_history = {
        key: [float(value) for value in values]
        for key, values in history.items()
    }
    with (RESULTS_DIR / "training_history.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(serialisable_history, file, indent=4)

    plt.figure(figsize=(9, 6))
    plt.plot(history["train_loss"], label="Training Loss")
    plt.plot(history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Binary Participant Training — No Hands")
    plt.legend()
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / "training_history.png", dpi=300)
    plt.close()


def save_split_details(train_df, val_df, train_people, val_people):
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    details = {
        "experiment": EXPERIMENT_NAME,
        "removed_feature": REMOVED_FEATURE,
        "split_type": "participant_group_split",
        "training_participants": train_people,
        "validation_participants": val_people,
        "training_samples": int(len(train_df)),
        "validation_samples": int(len(val_df)),
        "training_recordings": int(train_df["source_file"].nunique()),
        "validation_recordings": int(val_df["source_file"].nunique()),
        "training_class_distribution": {
            str(key): int(value)
            for key, value in train_df[TARGET_COLUMN].value_counts().items()
        },
        "validation_class_distribution": {
            str(key): int(value)
            for key, value in val_df[TARGET_COLUMN].value_counts().items()
        },
    }

    with (RESULTS_DIR / "participant_split.json").open(
        "w", encoding="utf-8"
    ) as file:
        json.dump(details, file, indent=4)


def main():
    print("=" * 60)
    print("DriverGuardianAI")
    print("Experiment 11: Binary Without Hands")
    print("=" * 60)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    PREPROCESSING_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    config = load_config(CONFIG_PATH)
    seed = config["training"].get("seed", 42)
    set_seed(seed)

    print("\nConfiguration loaded.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"\nUsing device: {device}")

    print("\nLoading dataset...")
    dataframe = load_data(config["dataset"]["path"])
    print(f"Original dataset shape: {dataframe.shape}")

    dataframe = prepare_dataset(dataframe)

    participants = sorted(dataframe["participant"].unique().tolist())
    print(f"Participants detected: {len(participants)}")
    print(participants)

    print("\nComplete binary class distribution:")
    print(dataframe[TARGET_COLUMN].value_counts())

    train_df, val_df, train_people, val_people = participant_split(
        dataframe,
        config["dataset"]["random_state"],
    )

    print("\nTraining participants:")
    for person in train_people:
        print(f"  - {person}")

    print("\nValidation participants:")
    for person in val_people:
        print(f"  - {person}")

    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    print("\nTraining binary class distribution:")
    print(train_df[TARGET_COLUMN].value_counts())

    print("\nValidation binary class distribution:")
    print(val_df[TARGET_COLUMN].value_counts())

    save_split_details(
        train_df,
        val_df,
        train_people,
        val_people,
    )

    train_df.drop(columns=["participant"], inplace=True)
    val_df.drop(columns=["participant"], inplace=True)

    print("\nRemoved feature: hands_detected")

    print("\nFitting preprocessing on training participants only...")
    train_df, preprocessing = fit_preprocessor(train_df)
    print("Training preprocessing complete.")

    print("\nTransforming validation participants...")
    val_df = transform_dataset(val_df, preprocessing)
    print("Validation transformation complete.")

    save_preprocessor(preprocessing, str(PREPROCESSING_PATH))
    print(f"\nPreprocessing saved to: {PREPROCESSING_PATH}")

    X_train = train_df.drop(TARGET_COLUMN, axis=1)
    y_train = train_df[TARGET_COLUMN].astype(int)
    X_val = val_df.drop(TARGET_COLUMN, axis=1)
    y_val = val_df[TARGET_COLUMN].astype(int)

    print("\nModel feature order:")
    print(X_train.columns.tolist())
    print(f"\nInput dimensions: {X_train.shape[1]}")

    if REMOVED_FEATURE in X_train.columns:
        raise RuntimeError("hands_detected was not removed.")

    encoded_classes = preprocessing["target_encoder"].classes_.tolist()
    print("\nEncoded target classes:")
    print(encoded_classes)

    if encoded_classes != ["Alert", "Fatigue"]:
        raise ValueError(
            "Unexpected target encoding order: "
            f"{encoded_classes}"
        )

    class_weights = calculate_class_weights(y_train, device)

    train_loader, val_loader = create_dataloaders(
        X_train,
        X_val,
        y_train,
        y_val,
        sampler=None,
        batch_size=config["training"]["batch_size"],
    )

    print("\nDataLoaders created.")

    model = FatigueResidualNN(
        input_dim=X_train.shape[1],
        hidden_dims=config["model"]["hidden_dims"],
        dropout=config["model"]["dropout"],
        num_classes=2,
    )
    model.to(device)

    print("\nBinary model created.")

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    print("Using weighted CrossEntropyLoss.")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 0.0),
    )

    scheduler = None
    scheduler_config = config.get("scheduler", {})
    if scheduler_config.get("enabled", False):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_config.get("factor", 0.5),
            patience=scheduler_config.get("patience", 5),
            min_lr=scheduler_config.get("min_lr", 1e-6),
        )
        print("Learning-rate scheduler enabled.")

    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
    )

    print("\nStarting binary no-hands training...\n")

    trainer.fit(
        epochs=config["training"]["epochs"],
        patience=config["training"]["patience"],
        save_path=str(MODEL_PATH),
    )

    print("\nLoading best no-hands model...")
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device)
    model.eval()

    print("\nEvaluating held-out participants...")

    val_loss, y_true, y_pred, fatigue_probs = collect_predictions(
        model,
        val_loader,
        criterion,
        device,
    )

    metrics, matrix = calculate_metrics(
        val_loss,
        y_true,
        y_pred,
        fatigue_probs,
    )

    report = classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=CLASS_NAMES,
        zero_division=0,
    )

    save_outputs(
        metrics,
        matrix,
        report,
        y_true,
        fatigue_probs,
        trainer.history,
    )

    print("\nConfusion Matrix:")
    print(matrix)

    print("\nClassification Report:")
    print(report)

    print("\n" + "=" * 60)
    print("Experiment 11 completed successfully!")
    print("=" * 60)

    print("\nFinal binary no-hands metrics:")
    for key, value in metrics.items():
        print(f"{key}: {value}")

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Preprocessing saved to: {PREPROCESSING_PATH}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()