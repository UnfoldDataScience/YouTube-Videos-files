"""SHAP-based explanation utilities.

This module wraps optional SHAP functionality to make it easy to compute
global and local explanations.  If SHAP is not installed, the functions
gracefully degrade by returning ``None``.
"""

from __future__ import annotations

from typing import Any, Optional, Tuple, Iterable

import numpy as np

try:
    import shap  # type: ignore
except Exception:
    shap = None  # type: ignore


def compute_shap_values(model: Any, preprocessor: Any, X_sample: np.ndarray) -> Optional[Any]:
    """Compute SHAP values for a given sample and model.

    Args:
        model: Underlying trained estimator.
        preprocessor: Fitted preprocessing pipeline used to transform raw inputs.
        X_sample: Raw feature matrix on which to compute SHAP values.

    Returns:
        A SHAP values object or ``None`` if SHAP is unavailable.
    """
    if shap is None:
        return None
    X_transformed = preprocessor.transform(X_sample)
    explainer = shap.Explainer(model, X_transformed)
    return explainer(X_transformed)


def shap_summary_plot(shap_values: Any, filename: str) -> None:
    """Create and save a global SHAP summary bar plot.

    Args:
        shap_values: SHAP values object returned from ``compute_shap_values``.
        filename: Path to save the plot.
    """
    if shap is None or shap_values is None:
        return
    shap.plots.bar(shap_values, show=False)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def shap_waterfall_plot(shap_values: Any, index: int, filename: str) -> None:
    """Create and save a SHAP waterfall plot for a specific instance.

    Args:
        shap_values: SHAP values object.
        index: Index of the sample to visualise.
        filename: Path to save the plot.
    """
    if shap is None or shap_values is None:
        return
    shap.plots.waterfall(shap_values[index], show=False)
    import matplotlib.pyplot as plt

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()