"""Tests for data loading and splitting using unittest."""

from __future__ import annotations

import unittest

import numpy as np

from src import data
from src.utils import load_config


class TestData(unittest.TestCase):
    """Unit tests for the data module."""

    def test_data_split_proportions(self) -> None:
        """Ensure the stratified split proportions roughly match the configuration."""
        X, y = data.load_dataset()
        config = load_config("config/config.yaml")
        split_cfg = config["split"]
        splits = data.stratified_split(
            X,
            y,
            train_size=split_cfg["train_size"],
            val_size=split_cfg["val_size"],
            test_size=split_cfg["test_size"],
            random_state=42,
        )
        total = len(X)
        train_prop = len(splits["train"][0]) / total
        val_prop = len(splits["val"][0]) / total
        test_prop = len(splits["test"][0]) / total
        self.assertAlmostEqual(train_prop + val_prop + test_prop, 1.0, places=3)
        self.assertLess(abs(train_prop - split_cfg["train_size"]), 0.05)
        self.assertLess(abs(val_prop - split_cfg["val_size"]), 0.05)
        self.assertLess(abs(test_prop - split_cfg["test_size"]), 0.05)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()