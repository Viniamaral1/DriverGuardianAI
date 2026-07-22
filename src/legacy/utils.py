"""
Utility functions used throughout DriverGuardianAI.
"""

import os
import random
import pickle

import numpy as np
import torch
import yaml


# =====================================================
# File Utilities
# =====================================================

def create_directory(path):
    """
    Create directory if it does not exist.
    """
    if path:
        os.makedirs(path, exist_ok=True)


# =====================================================
# Pickle Utilities
# =====================================================

def save_pickle(obj, filepath):
    """
    Save a Python object.
    """

    create_directory(os.path.dirname(filepath))

    with open(filepath, "wb") as f:
        pickle.dump(obj, f)


def load_pickle(filepath):
    """
    Load a pickled object.
    """

    with open(filepath, "rb") as f:
        return pickle.load(f)


# =====================================================
# YAML Utilities
# =====================================================

def save_yaml(config, filepath):
    """
    Save YAML configuration.
    """

    create_directory(os.path.dirname(filepath))

    with open(filepath, "w") as f:
        yaml.safe_dump(
            config,
            f,
            sort_keys=False
        )


def load_yaml(filepath):
    """
    Load YAML configuration.
    """

    with open(filepath, "r") as f:
        return yaml.safe_load(f)


# =====================================================
# Random Seed
# =====================================================

def set_seed(seed=42):
    """
    Make experiments reproducible.
    """

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed(seed)

        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# =====================================================
# Label Encoders
# =====================================================

def save_encoders(encoders, filepath):
    """
    Save fitted label encoders.
    """

    save_pickle(encoders, filepath)


def load_encoders(filepath):
    """
    Load fitted label encoders.
    """

    return load_pickle(filepath)


# =====================================================
# Full Preprocessing Pipeline
# =====================================================

def save_preprocessor(preprocessing, filepath):
    """
    Save the fitted preprocessing pipeline.
    """

    save_pickle(preprocessing, filepath)


def load_preprocessor(filepath):
    """
    Load the fitted preprocessing pipeline.
    """

    return load_pickle(filepath)