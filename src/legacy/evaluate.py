"""
Evaluation utilities for DriverGuardianAI.

NOTE:
This version is compatible with Trainer.validate_one_epoch() returning:
(avg_loss, accuracy, f1, recall, targets, predictions, probabilities)
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
)

from sklearn.metrics import (
    roc_curve,
    auc,
    precision_recall_curve,
    average_precision_score
)

from sklearn.preprocessing import label_binarize

from sklearn.calibration import calibration_curve

def save_metrics(metrics, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    with open(os.path.join(save_dir, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

def save_training_history(history, save_dir):
    with open(os.path.join(save_dir, "training_history.json"), "w") as f:
        json.dump(history, f, indent=4)

def save_classification_report(y_true, y_pred, save_dir):
    report = classification_report(
        y_true,
        y_pred,
        target_names=["Alert","Mild","Moderate"]
    )
    with open(os.path.join(save_dir,"classification_report.txt"),"w") as f:
        f.write(report)
    return report

def save_confusion_matrix(y_true, y_pred, save_dir):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(6,5))
    plt.imshow(cm)
    plt.colorbar()
    plt.xticks([0,1,2],["Alert","Mild","Moderate"])
    plt.yticks([0,1,2],["Alert","Mild","Moderate"])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j,i,str(cm[i,j]),ha="center",va="center")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"confusion_matrix.png"),dpi=300)
    plt.close()
    print("\nConfusion Matrix:")
    print(cm)

def plot_training_history(history, save_dir):
    plt.figure(figsize=(8,5))
    plt.plot(history["train_loss"],label="Train")
    plt.plot(history["val_loss"],label="Validation")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"training_history.png"),dpi=300)
    plt.close()

def plot_confidence_distribution(probabilities, save_dir):
    confidence = np.max(np.asarray(probabilities), axis=1)
    plt.figure(figsize=(8,5))
    plt.hist(confidence,bins=20)
    plt.xlabel("Confidence")
    plt.ylabel("Samples")
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir,"confidence_distribution.png"),dpi=300)
    plt.close()

def plot_roc_curves(
    y_true,
    probabilities,
    save_dir
):
    plot_precision_recall_curves(
    y_true,
    np.asarray(probabilities),
    save_dir
    )
    """
    Plot One-vs-Rest ROC curves for all fatigue classes.
    """

    class_names = [
        "Alert",
        "Mild",
        "Moderate"
    ]

    y_true_bin = label_binarize(
        y_true,
        classes=[0, 1, 2]
    )

    plt.figure(figsize=(8, 6))

    for i in range(3):

        fpr, tpr, _ = roc_curve(
            y_true_bin[:, i],
            probabilities[:, i]
        )

        roc_auc = auc(
            fpr,
            tpr
        )

        plt.plot(
            fpr,
            tpr,
            label=f"{class_names[i]} (AUC = {roc_auc:.3f})"
        )

    plt.plot(
        [0, 1],
        [0, 1],
        linestyle="--"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "roc_curves.png"
        ),
        dpi=300
    )

    plt.close()

def plot_precision_recall_curves(
    y_true,
    probabilities,
    save_dir
):
    """
    Plot Precision-Recall curves for all classes.
    """

    class_names = [
        "Alert",
        "Mild",
        "Moderate"
    ]

    y_true_bin = label_binarize(
        y_true,
        classes=[0,1,2]
    )

    plt.figure(figsize=(8,6))

    for i in range(3):

        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, i],
            probabilities[:, i]
        )

        ap = average_precision_score(
            y_true_bin[:, i],
            probabilities[:, i]
        )

        plt.plot(
            recall,
            precision,
            label=f"{class_names[i]} (AP={ap:.3f})"
        )

    plt.xlabel("Recall")
    plt.ylabel("Precision")

    plt.title("Precision-Recall Curves")

    plt.legend()

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "precision_recall_curves.png"
        ),
        dpi=300
    )

    plt.close()

def evaluate_model(trainer, experiment_name="experiment"):
    save_dir=os.path.join("results",experiment_name)
    os.makedirs(save_dir,exist_ok=True)
    print("\nRunning validation...")
    result=trainer.validate_one_epoch(return_preds=True)
    print(f"Validation output values: {len(result)}")
    if len(result)!=7:
        raise ValueError(f"Expected 7 outputs, got {len(result)}")
    val_loss,acc,f1,recall,y_true,y_pred,probabilities=result
    precision=precision_score(y_true,y_pred,average="weighted",zero_division=0)
    metrics={
        "experiment":experiment_name,
        "samples":len(y_true),
        "validation_loss":float(val_loss),
        "accuracy":float(acc),
        "precision":float(precision),
        "recall":float(recall),
        "f1_score":float(f1)
    }
    save_metrics(metrics,save_dir)
    save_training_history(trainer.history,save_dir)
    report=save_classification_report(y_true,y_pred,save_dir)
    save_confusion_matrix(y_true,y_pred,save_dir)
    plot_training_history(trainer.history,save_dir)
    plot_confidence_distribution(probabilities,save_dir)
    plot_roc_curves(
    y_true,
    np.asarray(probabilities),
    save_dir
    )
    print("\n")
    print(report)
    print(f"\n✅ Evaluation complete! Results saved to '{save_dir}'")
    return metrics,report,y_true,y_pred
