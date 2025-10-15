"""Tests for metrics generation and validity using unittest."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import unittest


class TestMetrics(unittest.TestCase):
    """Unit tests for training and evaluation metrics."""

    def test_metrics_file_exists_after_eval(self) -> None:
        """Run the training and evaluation scripts and verify that the metrics file exists and contains expected keys."""
        # Create a temporary working directory
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = os.getcwd()
            # Copy project into tmpdir
            dest = os.path.join(tmpdir, "proj")
            shutil.copytree(project_root, dest, dirs_exist_ok=True)
            cwd = os.getcwd()
            os.chdir(dest)
            try:
                # Train model (fast linear to minimise runtime)
                subprocess.check_call([
                    "python",
                    "-m",
                    "src.train",
                    "--model",
                    "linear",
                    "--fast",
                ], timeout=300)
                # Evaluate model (skip SHAP)
                subprocess.check_call([
                    "python",
                    "-m",
                    "src.evaluate",
                    "--fast",
                    "--skip-shap",
                ], timeout=300)
                metrics_path = os.path.join("artifacts", "metrics.json")
                self.assertTrue(os.path.exists(metrics_path))
                with open(metrics_path, "r", encoding="utf-8") as fh:
                    metrics = json.load(fh)
                for key in ["rmse", "mae", "r2"]:
                    self.assertIn(key, metrics)
                    self.assertGreaterEqual(metrics[key], 0.0)
            finally:
                os.chdir(cwd)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()