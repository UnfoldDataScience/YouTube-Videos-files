"""Training script for the California Housing Regressor.

This module implements a command line interface to train a regression model
using the California Housing dataset.  It supports multiple models,
preprocessing options and hyperparameter tuning via randomised search.

Example usage:

```
python -m src.train --model random_forest --use-pca --fast
```
"""

from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from .data import load_dataset, stratified_split
from .features import build_preprocessor
from .models import get_model, get_param_distributions
from .utils import get_logger, load_config, save_json, set_seed


def parse_args() -> argparse.Namespace:
    """Define and parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train a regression model.")
    parser.add_argument(
        "--model",
        type=str,
        default="random_forest",
        choices=["linear", "random_forest", "gradient_boosting", "xgboost"],
        help="Which model to train.",
    )
    parser.add_argument(
        "--use-pca",
        action="store_true",
        help="Include a PCA step in the preprocessing pipeline.",
    )
    parser.add_argument(
        "--use-poly",
        action="store_true",
        help="Include polynomial feature expansion (degree 2).",
    )
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Use a reduced hyperparameter search and fewer CV folds for faster runs.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> Dict[str, Any]:
    """Run the training routine.

    Args:
        args: Parsed command line arguments.

    Returns:
        A dictionary containing validation metrics and best hyperparameters.
    """
    config = load_config(args.config)
    # Extract split and training parameters
    split_cfg = config["split"]
    train_cfg = config["training"]

    # Determine number of iterations and CV folds
    n_iter = train_cfg["fast_n_iter"] if args.fast else train_cfg["n_iter"]
    cv_folds = train_cfg["fast_cv_folds"] if args.fast else train_cfg["cv_folds"]

    # Fix random seeds
    set_seed(train_cfg["random_state"])

    # Set up logging
    logger = get_logger("train", os.path.join("logs", "train.log"))
    logger.info("Starting training with model=%s, use_pca=%s, use_poly=%s, fast=%s", args.model, args.use_pca, args.use_poly, args.fast)

    # Load and split the dataset
    X, y = load_dataset()
    splits = stratified_split(
        X,
        y,
        train_size=split_cfg["train_size"],
        val_size=split_cfg["val_size"],
        test_size=split_cfg["test_size"],
        random_state=train_cfg["random_state"],
    )
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    # Build preprocessing pipeline
    preprocessor = build_preprocessor(use_pca=args.use_pca, use_poly=args.use_poly)
    # Instantiate model
    try:
        model = get_model(args.model)
    except ValueError as exc:
        logger.error(str(exc))
        raise
    # Assemble full pipeline
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    # Hyperparameter search space
    param_distributions = get_param_distributions(args.model)
    # Perform RandomizedSearchCV if there are hyperparameters
    if param_distributions:
        logger.info("Starting hyperparameter search with %d iterations and %d CV folds", n_iter, cv_folds)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv_folds,
            scoring="neg_root_mean_squared_error",
            n_jobs=-1,
            random_state=train_cfg["random_state"],
            verbose=0,
        )
        search.fit(X_train, y_train)
        best_pipeline = search.best_estimator_
        best_params = search.best_params_
        logger.info("Best hyperparameters: %s", best_params)
    else:
        # No hyperparameter search needed
        logger.info("No hyperparameters to tune for model '%s'", args.model)
        best_pipeline = pipeline
        best_pipeline.fit(X_train, y_train)
        best_params = {}
    # Evaluate on validation set
    y_val_pred = best_pipeline.predict(X_val)
    rmse = mean_squared_error(y_val, y_val_pred, squared=False)
    mae = mean_absolute_error(y_val, y_val_pred)
    r2 = r2_score(y_val, y_val_pred)
    metrics = {"rmse": float(rmse), "mae": float(mae), "r2": float(r2)}
    logger.info(
        "Validation metrics - RMSE: %.4f, MAE: %.4f, R^2: %.4f", metrics["rmse"], metrics["mae"], metrics["r2"]
    )
    # Retrain on combined train+val before saving
    X_combined = pd.concat([X_train, X_val], ignore_index=True)
    y_combined = pd.concat([y_train, y_val], ignore_index=True)
    best_pipeline.fit(X_combined, y_combined)
    # Persist model and metadata
    os.makedirs("artifacts", exist_ok=True)
    model_path = os.path.join("artifacts", "model.joblib")
    joblib.dump(best_pipeline, model_path)
    logger.info("Saved model to %s", model_path)
    # Save metrics and params for reference (will be overwritten by evaluate.py)
    save_json(metrics, os.path.join("artifacts", "metrics.json"))
    save_json(best_params, os.path.join("artifacts", "params.json"))
    return {"metrics": metrics, "params": best_params}


def main() -> None:
    args = parse_args()
    train(args)


if __name__ == "__main__":  # pragma: no cover
    main()