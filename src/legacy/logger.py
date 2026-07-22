"""
Logging utilities for DriverGuardianAI.

Creates a logger that writes messages to both the console
and a log file inside the logs/ directory.
"""

import logging
import os


def setup_logger(log_file="logs/training.log"):
    """
    Configure and return the project logger.

    Parameters
    ----------
    log_file : str
        Path to the log file.

    Returns
    -------
    logging.Logger
        Configured logger.
    """

    os.makedirs(
        os.path.dirname(log_file),
        exist_ok=True
    )

    logger = logging.getLogger("DriverGuardianAI")

    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(message)s"
    )

    file_handler = logging.FileHandler(log_file)

    file_handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()

    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    logger.addHandler(console_handler)

    return logger