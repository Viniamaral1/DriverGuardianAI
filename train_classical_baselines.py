"""Experiment 12: participant-aware classical baselines for DriverGuardianAI."""

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATASET_PATH = Path("data/dataset_exp3.csv")
RESULTS_DIR = Path("results/experiment12_classical_baselines")
MODEL_PATH = Path("models/driver_guardian_best_classical.joblib")
RANDOM_STATE = 42
TEST_SIZE = 0.25

FEATURES = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "condition",
    "low_light",
    "face_confidence",
    "blink_count",
]
NUMERIC_FEATURES = [
    "ear",
    "yawn_score",
    "head_tilt",
    "hands_detected",
    "low_light",
    "face_confidence",
    "blink_count",
]
CATEGORICAL_FEATURES = ["condition"]
CLASS_NAMES = ["Alert", "Fatigue"]


def extract_participant(source_file):
    if pd.isna(source_file):
        raise ValueError("Missing source_file value.")
    value = str(source_file).strip()
    if not value:
        raise ValueError("Empty source_file value.")
    return value.split("_", 1)[0].lower()


def load_dataset():
    if not DATASET_PATH.exists():
        raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

    df = pd.read_csv(DATASET_PATH)
    required = set(FEATURES + ["source_file", "fatigue_level"])
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df = df.copy()
    df["participant"] = df["source_file"].apply(extract_participant)

    mapping = {
        "Alert": 0,
        "Mild Fatigue": 1,
        "Moderate Fatigue": 1,
        "Severe Fatigue": 1,
    }
    unknown = set(df["fatigue_level"].dropna().astype(str).unique()) - set(mapping)
    if unknown:
        raise ValueError(f"Unknown labels: {sorted(unknown)}")

    df["binary_target"] = df["fatigue_level"].astype(str).map(mapping)

    df["low_light"] = (
        df["low_light"]
        .astype(str)
        .str.lower()
        .map({"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0, "1.0": 1.0, "0.0": 0.0})
    )

    for column in NUMERIC_FEATURES:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["condition"] = df["condition"].astype(str)
    return df


def participant_split(df):
    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_idx, val_idx = next(splitter.split(df, groups=df["participant"]))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()

    train_people = sorted(train_df["participant"].unique().tolist())
    val_people = sorted(val_df["participant"].unique().tolist())
    overlap = set(train_people).intersection(val_people)
    if overlap:
        raise RuntimeError(f"Participant leakage: {sorted(overlap)}")

    return train_df, val_df, train_people, val_people


def make_preprocessor():
    numeric = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )


def build_models():
    return {
        "logistic_regression": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=2000,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    RandomForestClassifier(
                        n_estimators=400,
                        max_depth=18,
                        min_samples_leaf=5,
                        class_weight="balanced",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "hist_gradient_boosting": Pipeline(
            [
                ("preprocessor", make_preprocessor()),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        learning_rate=0.06,
                        max_iter=300,
                        max_leaf_nodes=31,
                        min_samples_leaf=30,
                        l2_regularization=1.0,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def metrics_dict(name, y_true, y_pred, probabilities):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    return {
        "model": name,
        "samples": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_fatigue": float(precision_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "recall_sensitivity": float(sensitivity),
        "specificity": float(specificity),
        "f1_fatigue": float(f1_score(y_true, y_pred, pos_label=1, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
        "average_precision": float(average_precision_score(y_true, probabilities)),
        "false_positive_rate": float(1.0 - specificity),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def save_confusion_matrix(cm, model_name):
    plt.figure(figsize=(7, 6))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks([0, 1], CLASS_NAMES)
    plt.yticks([0, 1], CLASS_NAMES)
    for row in range(2):
        for col in range(2):
            plt.text(col, row, str(cm[row, col]), ha="center", va="center", fontsize=12)
    plt.xlabel("Predicted class")
    plt.ylabel("Actual class")
    plt.title(f"Participant-aware baseline: {model_name}")
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f"{model_name}_confusion_matrix.png", dpi=300)
    plt.close()


def main():
    print("=" * 72)
    print("DriverGuardianAI")
    print("Experiment 12: Classical Participant-Aware Baselines")
    print("=" * 72)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    df = load_dataset()
    train_df, val_df, train_people, val_people = participant_split(df)

    print("\nTraining participants:")
    for person in train_people:
        print(f"  - {person}")
    print("\nValidation participants:")
    for person in val_people:
        print(f"  - {person}")
    print(f"\nTraining samples: {len(train_df)}")
    print(f"Validation samples: {len(val_df)}")

    with (RESULTS_DIR / "participant_split.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "training_participants": train_people,
                "validation_participants": val_people,
                "training_samples": int(len(train_df)),
                "validation_samples": int(len(val_df)),
            },
            file,
            indent=4,
        )

    X_train = train_df[FEATURES]
    y_train = train_df["binary_target"].astype(int)
    X_val = val_df[FEATURES]
    y_val = val_df["binary_target"].astype(int)

    rows = []
    details = {}
    best_name = None
    best_model = None
    best_auc = -np.inf

    for name, model in build_models().items():
        print("\n" + "-" * 72)
        print(f"Training: {name}")
        print("-" * 72)

        model.fit(X_train, y_train)
        predictions = model.predict(X_val)
        probabilities = model.predict_proba(X_val)[:, 1]

        metrics = metrics_dict(name, y_val.to_numpy(), predictions, probabilities)
        cm = confusion_matrix(y_val, predictions, labels=[0, 1])
        report = classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=CLASS_NAMES,
            zero_division=0,
        )

        rows.append(metrics)
        details[name] = {
            "metrics": metrics,
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        with (RESULTS_DIR / f"{name}_classification_report.txt").open("w", encoding="utf-8") as file:
            file.write(report)
        save_confusion_matrix(cm, name)

        print("\nConfusion Matrix:")
        print(cm)
        print("\nClassification Report:")
        print(report)
        print("\nMetrics:")
        for key, value in metrics.items():
            print(f"{key}: {value}")

        if metrics["roc_auc"] > best_auc:
            best_auc = metrics["roc_auc"]
            best_name = name
            best_model = model

    comparison = pd.DataFrame(rows).sort_values(
        ["roc_auc", "f1_fatigue", "accuracy"],
        ascending=False,
    )
    comparison.to_csv(RESULTS_DIR / "model_comparison.csv", index=False)

    with (RESULTS_DIR / "model_comparison.json").open("w", encoding="utf-8") as file:
        json.dump({"best_model": best_name, "results": details}, file, indent=4)

    if best_model is None:
        raise RuntimeError("No model completed training.")

    joblib.dump(
        {
            "model_name": best_name,
            "pipeline": best_model,
            "feature_columns": FEATURES,
            "class_names": CLASS_NAMES,
        },
        MODEL_PATH,
    )

    print("\n" + "=" * 72)
    print("Experiment 12 completed successfully.")
    print("=" * 72)
    print("\nModel comparison:")
    print(comparison.to_string(index=False))
    print(f"\nBest model by ROC-AUC: {best_name}")
    print(f"Best model saved to: {MODEL_PATH}")
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()