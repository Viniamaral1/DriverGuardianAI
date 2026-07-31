"""
Class weighting utilities for DriverGuardianAI.

Handles:
- Class weight calculation
- WeightedRandomSampler creation

Used to address fatigue class imbalance.
"""


import torch

import numpy as np

from torch.utils.data import WeightedRandomSampler



def compute_class_weights(y_train):
    """
    Compute class weights for focal loss.

    Classes:

    0 -> Alert
    1 -> Mild Fatigue
    2 -> Moderate/Severe Fatigue

    Higher weights increase the penalty
    for misclassifying important fatigue states.
    """


    classes, counts = np.unique(
        y_train,
        return_counts=True
    )


    print("\nClass distribution:")

    for c, count in zip(classes, counts):

        print(
            f"Class {c}: {count}"
        )


    total = len(y_train)


    weights = []


    for c in classes:

        weight = total / (
            len(classes) * counts[c]
        )

        weights.append(weight)



    weights = torch.tensor(
        weights,
        dtype=torch.float32
    )


    print(
        "\nCalculated class weights:"
    )

    print(
        weights
    )


    return weights




def create_weighted_sampler(y_train):
    """
    Create WeightedRandomSampler.

    This balances the training batches by
    increasing the probability of minority
    fatigue classes being sampled.
    """


    class_weights = compute_class_weights(
        y_train
    )


    sample_weights = class_weights[
        torch.tensor(
            y_train.values,
            dtype=torch.long
        )
    ]


    sampler = WeightedRandomSampler(

        weights=sample_weights,

        num_samples=len(sample_weights),

        replacement=True

    )


    return sampler