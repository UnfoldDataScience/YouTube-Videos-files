"""Tests for command line interfaces using unittest."""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import unittest


class TestCLI(unittest.TestCase):
    """Unit tests for the CLI scripts."""

    def test_cli_train_and_eval(self) -> None:
        """Run the train and eval commands and check for artifact existence."""
        with tempfile.TemporaryDirectory() as tmpdir:
            project_root = os.getcwd()
            dest = os.path.join(tmpdir, "proj")
            shutil.copytree(project_root, dest, dirs_exist_ok=True)
            cwd = os.getcwd()
            os.chdir(dest)
            try:
                # Train using random forest (fast) to exercise hyperparameter search
                subprocess.check_call([
                    "python",
                    "-m",
                    "src.train",
                    "--model",
                    "random_forest",
                    "--fast",
                ], timeout=300)
                self.assertTrue(os.path.exists(os.path.join("artifacts", "model.joblib")))
                # Evaluate
                subprocess.check_call([
                    "python",
                    "-m",
                    "src.evaluate",
                    "--fast",
                    "--skip-shap",
                ], timeout=300)
                self.assertTrue(os.path.exists(os.path.join("artifacts", "metrics.json")))
            finally:
                os.chdir(cwd)


if __name__ == "__main__":  # pragma: no cover
    unittest.main()