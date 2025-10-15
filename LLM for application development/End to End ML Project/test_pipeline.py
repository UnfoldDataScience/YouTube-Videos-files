"""Tests for pipeline construction and basic model fitting using unittest."""

from __future__ import annotations

import unittest

from sklearn.pipeline import Pipeline

from src.data import load_dataset, stratified_split
from src.features import build_preprocessor
from src.models import get_model
from src.utils import load_config


class TestPipeline(unittest.TestCase):
    """Unit tests for pipeline functionality."""

    def test_pipeline_fit_predict(self) -> None:
        """Ensure a simple pipeline can fit and produce predictions without errors."""
        X, y = load_dataset()
        config = load_config("config/config.yaml")
        split_cfg = config["split"]
        splits = stratified_split(
            X,
            y,
            train_size=split_cfg["train_size"],
            val_size=split_cfg["val_size"],
            test_size=split_cfg["test_size"],
            random_state=42,
        )
        X_train, y_train = splits["train"]
        X_test, _ = splits["test"]
        preprocessor = build_preprocessor(use_pca=False, use_poly=False)
        model = get_model("linear")
        pipeline = Pipeline([
            ("pre", preprocessor),
            ("model", model),
        ])
        pipeline.fit(X_train, y_train)
        preds = pipeline.predict(X_test)
        self.assertEqual(len(preds), len(X_test))


if __name__ == "__main__":  # pragma: no cover
    unittest.main()