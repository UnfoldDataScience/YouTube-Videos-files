"""Evaluation script for the California Housing Regressor.

This module loads a trained model, evaluates it on the test set, writes
metrics to disk and generates a suite of diagnostic plots.  Optional SHAP
explanations are computed to provide interpretability if the shap package is
available and the user has not disabled it.
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .data import load_dataset, stratified_split
from .utils import get_logger, load_config, save_json, set_seed
from .viz import (
    parity_plot,
    residuals_plot,
    error_hist_plot,
    feature_importance_plot,
)


def parse_args() -> argparse.Namespace:
    """Define and parse command line arguments for evaluation."""
    parser = argparse.ArgumentParser(description="Evaluate a trained model.")
    parser.add_argument(
        "--skip-shap",
        action="store_true",
        help="Skip SHAP explainability plots even if shap is installed.",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use fewer samples for SHAP explanations to speed up evaluation.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the evaluation routine.

    Args:
        args: Parsed command line arguments.

    Returns:
        A dictionary containing test metrics.
    """
    config = load_config(args.config)
    split_cfg = config["split"]
    train_cfg = config["training"]
    set_seed(train_cfg["random_state"])
    logger = get_logger("evaluate", os.path.join("logs", "evaluate.log"))
    logger.info("Starting evaluation; skip_shap=%s, fast=%s", args.skip_shap, args.fast)
    # Load dataset and model
    X, y = load_dataset()
    splits = stratified_split(
        X,
        y,
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )
    X_test, y_test = splits["test"]
    # Load the trained model
    model_path = os.path.join("artifacts", "model.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Trained model not found at {model_path}.  Run the training script first."
        )
    pipeline = joblib.load(model_path)
    # Predict
    y_pred = pipeline.predict(X_test)
    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    logger.info(
        "Test metrics - RMSE: %.4f, MAE: %.4f, R^2: %.4f", metrics["rmse"], metrics["mae"], metrics["r2"]
    )
    # Write metrics to JSON
    save_json(metrics, os.path.join("artifacts", "metrics.json"))
    # Create plots directory
    plots_dir = os.path.join("artifacts", "plots")
    os.makedirs(plots_dir, exist_ok=True)
    # Generate diagnostic plots
    parity_plot(y_test, y_pred, os.path.join(plots_dir, "parity.png"))
    residuals_plot(y_test, y_pred, os.path.join(plots_dir, "residuals.png"))
    error_hist_plot(y_test, y_pred, os.path.join(plots_dir, "error_hist.png"))
    # Feature importance plot (may silently skip if not supported)
    feature_importance_plot(
        getattr(pipeline, "named_steps", {}).get("model", pipeline),
        feature_names=X_test.columns.tolist(),
        filename=os.path.join(plots_dir, "feature_importance.png"),
    )
    # SHAP explanations
    if not args.skip_shap:
        try:
            import shap  # type: ignore

            # Use a small sample for efficiency in fast mode
            sample_size = 200 if args.fast else min(1000, len(X_test))
            shap_X = X_test.sample(n=sample_size, random_state=train_cfg["random_state"])
            # For shap.Explainer to work correctly, feed the underlying model after preprocessing
            model = getattr(pipeline, "named_steps", {}).get("model", None)
            preprocessor = getattr(pipeline, "named_steps", {}).get("preprocessor", None)
            if model is None or preprocessor is None:
                logger.warning("Unable to locate model or preprocessor for SHAP; skipping.")
            else:
                X_processed = preprocessor.transform(shap_X)
                explainer = shap.Explainer(model, X_processed)
                shap_values = explainer(X_processed)
                # Global summary bar plot
                shap_summary_path = os.path.join(plots_dir, "shap_summary.png")
                shap.plots.bar(shap_values, show=False)
                import matplotlib.pyplot as plt

                plt.tight_layout()
                plt.savefig(shap_summary_path)
                plt.close()
                # Local explanation for the first test instance
                first_processed = X_processed[0]
                first_raw = shap_X.iloc[[0]]
                shap_waterfall_path = os.path.join(plots_dir, "shap_waterfall.png")
                shap.plots.waterfall(shap_values[0], show=False)
                plt.tight_layout()
                plt.savefig(shap_waterfall_path)
                plt.close()
                logger.info("Generated SHAP explanations.")
        except Exception as exc:  # noqa: BLE001
            logger.info("SHAP explanations unavailable or failed: %s", exc)
    else:
        logger.info("Skipping SHAP explanations as requested.")
    return metrics


def main() -> None:
    args = parse_args()
    evaluate(args)


if __name__ == "__main__":  # pragma: no cover
    main()