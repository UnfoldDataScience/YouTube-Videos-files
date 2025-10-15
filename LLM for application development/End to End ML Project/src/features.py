"""Feature engineering and preprocessing pipelines."""

from __future__ import annotations

from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler


def build_preprocessor(use_pca: bool = False, use_poly: bool = False) -> Pipeline:
    """Construct a preprocessing pipeline.

    The pipeline always standardises features.  Additional steps such as
    polynomial feature expansion and principal component analysis are included
    conditionally based on the supplied flags.

    Args:
        use_pca: Whether to append a PCA transformation.
        use_poly: Whether to include a polynomial feature expansion (degree 2).

    Returns:
        A ``sklearn.pipeline.Pipeline`` object that can be used in a larger
        modelling pipeline.
    """
    steps = []
    # Standard scaling for all numerical features
    steps.append(("scaler", StandardScaler()))
    # Optional polynomial features (only meaningful for linear models)
    if use_poly:
        steps.append(("poly", PolynomialFeatures(degree=2, include_bias=False)))
    # Optional principal component analysis for dimensionality reduction
    if use_pca:
        steps.append(("pca", PCA()))
    return Pipeline(steps)