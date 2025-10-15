"""Utility functions for configuration, logging and reproducibility."""

from __future__ import annotations

import json
import logging
import os
import random
from typing import Any, Dict

import numpy as np
import yaml


def load_config(path: str) -> Dict[str, Any]:
    """Load a YAML configuration file.

    Args:
        path: Path to the YAML file.

    Returns:
        Parsed configuration as a dictionary.
    """
    with open(path, "r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def save_json(data: Dict[str, Any], path: str) -> None:
    """Save a Python dictionary to a JSON file.

    Args:
        data: Dictionary of serialisable data.
        path: Destination path for the JSON file.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=4)


def get_logger(name: str, log_file: str) -> logging.Logger:
    """Create or retrieve a logger that logs to both a file and the console.

    Args:
        name: Name of the logger.
        log_file: Path to the log file.

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    # Prevent duplicate handlers when this function is called multiple times
    if logger.handlers:
        return logger
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    # File handler
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    # Console handler
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(ch)
    return logger


def set_seed(seed: int) -> None:
    """Fix random seeds for Python, NumPy and the hash seed.

    Args:
        seed: Seed value to use.
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)