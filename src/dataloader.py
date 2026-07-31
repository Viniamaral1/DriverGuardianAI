"""
PyTorch DataLoader utilities for DriverGuardianAI.
"""

import torch

from torch.utils.data import TensorDataset, DataLoader



def create_dataloaders(
    X_train,
    X_val,
    y_train,
    y_val,
    sampler=None,
    batch_size=128
):

    """
    Create PyTorch DataLoaders for training and validation.
    """

    # -------------------------------
    # Ensure numeric features
    # -------------------------------

    X_train = X_train.astype("float32")
    X_val = X_val.astype("float32")


    # Convert numpy arrays to tensors

    X_train = torch.tensor(
        X_train.values,
        dtype=torch.float32
    )


    X_val = torch.tensor(
        X_val.values,
        dtype=torch.float32
    )


    # -------------------------------
    # Labels
    # -------------------------------

    y_train = torch.tensor(
        y_train.values,
        dtype=torch.long
    )


    y_val = torch.tensor(
        y_val.values,
        dtype=torch.long
    )


    # -------------------------------
    # Dataset objects
    # -------------------------------

    train_dataset = TensorDataset(
        X_train,
        y_train
    )


    val_dataset = TensorDataset(
        X_val,
        y_val
    )


    # -------------------------------
    # DataLoaders
    # -------------------------------

    if sampler is not None:

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            sampler=sampler
        )

    else:

        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )


    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False
    )


    return train_loader, val_loader