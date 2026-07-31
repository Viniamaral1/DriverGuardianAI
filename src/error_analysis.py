"""
Error analysis utilities for DriverGuardianAI.

Provides tools for analysing model mistakes after evaluation.
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


def create_error_dataframe(
    y_true,
    y_pred,
    probabilities=None
):
    """
    Create a dataframe containing every prediction.
    """

    df = pd.DataFrame({
        "Actual": y_true,
        "Predicted": y_pred
    })

    df["Correct"] = df["Actual"] == df["Predicted"]

    if probabilities is not None:
        df["Confidence"] = probabilities.max(axis=1)

    return df


def save_error_csv(
    error_df,
    save_dir
):
    """
    Save every prediction to CSV.
    """

    os.makedirs(save_dir, exist_ok=True)

    error_df.to_csv(
        os.path.join(save_dir, "prediction_analysis.csv"),
        index=False
    )


def save_misclassified_samples(
    error_df,
    save_dir
):
    """
    Save only incorrect predictions.
    """

    mistakes = error_df[
        error_df["Correct"] == False
    ]

    mistakes.to_csv(
        os.path.join(save_dir, "misclassified_samples.csv"),
        index=False
    )

    return mistakes


def plot_prediction_distribution(
    error_df,
    save_dir
):
    """
    Plot number of predictions per class.
    """

    counts = (
        error_df["Predicted"]
        .value_counts()
        .sort_index()
    )

    plt.figure(figsize=(6,4))

    counts.plot(kind="bar")

    plt.xlabel("Predicted Class")
    plt.ylabel("Count")
    plt.title("Prediction Distribution")

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            save_dir,
            "prediction_distribution.png"
        ),
        dpi=300
    )

    plt.close()


def analyse_errors(
    y_true,
    y_pred,
    probabilities,
    save_dir
):
    """
    Complete error analysis pipeline.
    """

    print("\nRunning error analysis...")

    error_df = create_error_dataframe(
        y_true,
        y_pred,
        probabilities
    )

    save_error_csv(
        error_df,
        save_dir
    )

    mistakes = save_misclassified_samples(
        error_df,
        save_dir
    )

    plot_prediction_distribution(
        error_df,
        save_dir
    )

    print(f"Total samples: {len(error_df)}")
    print(f"Correct predictions: {error_df['Correct'].sum()}")
    print(f"Incorrect predictions: {len(mistakes)}")

    return error_df