"""Model factories and hyperparameter search spaces."""

from __future__ import annotations

from typing import Dict, Any, List

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression

try:
    # Optional dependency
    from xgboost import XGBRegressor  # type: ignore
except Exception:
    XGBRegressor = None  # type: ignore


def get_model(name: str):
    """Instantiate a regression model based on its name.

    Args:
        name: Name of the model.  Supported values are ``linear``,
            ``random_forest``, ``gradient_boosting`` and ``xgboost`` (if the
            XGBoost package is installed).

    Returns:
        An instantiated scikit‑learn compatible regressor.

    Raises:
        ValueError: If an unsupported model name is given or the requested
            model requires a missing optional dependency.
    """
    name = name.lower()
    if name == "linear":
        return LinearRegression()
    if name == "random_forest":
        return RandomForestRegressor(random_state=42)
    if name == "gradient_boosting":
        return GradientBoostingRegressor(random_state=42)
    if name == "xgboost":
        if XGBRegressor is None:
            raise ValueError(
                "XGBoost is not available.  Install the xgboost package to use this model."
            )
        return XGBRegressor(random_state=42, verbosity=0, objective="reg:squarederror")
    raise ValueError(f"Unsupported model: {name}")


def get_param_distributions(name: str) -> Dict[str, List[Any]]:
    """Define hyperparameter search spaces for supported models.

    These distributions are used with ``RandomizedSearchCV``.  They are kept
    deliberately small to maintain reasonable run times while still exploring
    meaningful hyperparameter values.

    Args:
        name: Name of the model.

    Returns:
        A dictionary mapping parameter names to lists of candidate values.
    """
    name = name.lower()
    if name == "linear":
        # Linear regression has no hyperparameters; return an empty dict.
        return {}
    if name == "random_forest":
        return {
            "model__n_estimators": [50, 100, 150, 200],
            "model__max_depth": [None, 5, 10, 15],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
        }
    if name == "gradient_boosting":
        return {
            "model__n_estimators": [50, 100, 150, 200],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [2, 3, 4, 5],
            "model__subsample": [0.8, 1.0],
        }
    if name == "xgboost" and XGBRegressor is not None:
        return {
            "model__n_estimators": [50, 100, 150, 200],
            "model__learning_rate": [0.01, 0.05, 0.1, 0.2],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.8, 1.0],
            "model__colsample_bytree": [0.8, 1.0],
        }
    return {}