"""
Configuration loader.

Loads project settings from config/config.yaml.
"""

import yaml


def load_config(config_path):
    """
    Load the YAML configuration file.

    Parameters
    ----------
    config_path : str
        Path to the configuration file.

    Returns
    -------
    dict
        Configuration dictionary.
    """

    with open(config_path, "r") as file:

        config = yaml.safe_load(file)

    return config