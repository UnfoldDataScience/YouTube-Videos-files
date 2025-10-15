"""Plotting utilities for regression diagnostics and feature importance."""

from __future__ import annotations

from typing import Iterable, List

import numpy as np
import matplotlib.pyplot as plt


def parity_plot(y_true: Iterable[float], y_pred: Iterable[float], filename: str) -> None:
    """Save a parity plot comparing true and predicted values.

    Args:
        y_true: Iterable of ground truth values.
        y_pred: Iterable of predicted values.
        filename: Path to save the resulting plot.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.5)
    min_val = float(min(y_true.min(), y_pred.min()))
    max_val = float(max(y_true.max(), y_pred.max()))
    ax.plot([min_val, max_val], [min_val, max_val], linestyle="--", linewidth=1)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual (Parity Plot)")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def residuals_plot(y_true: Iterable[float], y_pred: Iterable[float], filename: str) -> None:
    """Save a residuals plot showing prediction residuals against predicted values.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        filename: Path to save the resulting plot.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    residuals = y_true - y_pred
    fig, ax = plt.subplots()
    ax.scatter(y_pred, residuals, alpha=0.5)
    ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Residual (Actual - Predicted)")
    ax.set_title("Residuals vs Predicted")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def error_hist_plot(y_true: Iterable[float], y_pred: Iterable[float], filename: str) -> None:
    """Save a histogram of prediction errors.

    Args:
        y_true: Ground truth values.
        y_pred: Predicted values.
        filename: Path to save the histogram.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    errors = y_true - y_pred
    fig, ax = plt.subplots()
    ax.hist(errors, bins=30, alpha=0.7)
    ax.set_xlabel("Error (Actual - Predicted)")
    ax.set_ylabel("Frequency")
    ax.set_title("Distribution of Prediction Errors")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)


def feature_importance_plot(
    model: Any, feature_names: List[str], filename: str
) -> None:
    """Save a bar chart of feature importances if the model exposes them.

    For tree-based models, the ``feature_importances_`` attribute is used.  For
    linear models, the absolute value of the coefficients is used.  If neither
    attribute is available, the function returns without error.

    Args:
        model: Trained estimator from which to extract importance values.
        feature_names: Names of the input features.
        filename: Path to save the plot.
    """
    # Tree-based models
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.array(model.feature_importances_)
    elif hasattr(model, "coef_"):
        coefs = model.coef_
        # For multi-output regressors, sum the absolute values across targets
        if len(np.shape(coefs)) > 1:
            coefs = np.sum(np.abs(coefs), axis=0)
        importances = np.abs(coefs)
    if importances is None:
        # Nothing to plot
        return
    # Limit to top 20 features for readability
    indices = np.argsort(importances)[::-1][: min(20, len(importances))]
    top_features = [feature_names[i] for i in indices]
    top_importances = importances[indices]
    fig, ax = plt.subplots()
    ax.bar(range(len(top_features)), top_importances)
    ax.set_xticks(range(len(top_features)))
    ax.set_xticklabels(top_features, rotation=90)
    ax.set_ylabel("Importance")
    ax.set_title("Feature Importance")
    fig.tight_layout()
    fig.savefig(filename)
    plt.close(fig)