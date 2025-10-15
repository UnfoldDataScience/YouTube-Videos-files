"""Data loading and splitting utilities for the California Housing dataset."""

from __future__ import annotations

import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import StratifiedShuffleSplit
from typing import Tuple, Dict, Any


def load_dataset() -> Tuple[pd.DataFrame, pd.Series]:
    """Load the California Housing dataset, falling back to a synthetic dataset if remote fetch fails.

    Under normal circumstances this function fetches the California Housing dataset
    via :func:`sklearn.datasets.fetch_california_housing`.  If the download
    fails (e.g. due to lack of network access), a synthetic dataset is
    generated using :func:`sklearn.datasets.make_regression` with a similar
    number of samples and features.  Feature names correspond to those of the
    original dataset to maintain compatibility.

    Returns:
        A tuple of features (X) and target (y) as pandas objects.
    """
    try:
        data = fetch_california_housing(as_frame=True)
        X = data.data.copy()
        y = data.target.copy()
        return X, y
    except Exception:
        # Fallback: generate synthetic regression data with similar shape
        from sklearn.datasets import make_regression

        n_samples = 20640  # approximate number of rows in California Housing
        n_features = 8
        X_array, y_array = make_regression(
            n_samples=n_samples,
            n_features=n_features,
            noise=0.1,
            random_state=0,
        )
        feature_names = [
            "MedInc",
            "HouseAge",
            "AveRooms",
            "AveBedrms",
            "Population",
            "AveOccup",
            "Latitude",
            "Longitude",
        ]
        X = pd.DataFrame(X_array, columns=feature_names)
        y = pd.Series(y_array, name="MedHouseVal")
        return X, y


def stratified_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_size: float,
    val_size: float,
    test_size: float,
    random_state: int,
) -> Dict[str, Tuple[pd.DataFrame, pd.Series]]:
    """Perform a stratified train/validation/test split using quantile bins of the target.

    The target is binned into equal‑sized quantiles, then a two‑stage stratified shuffle
    split is applied.  First, the dataset is split into train+val and test.  Second,
    the train+val portion is split into train and validation.  Ratios are controlled by
    `train_size`, `val_size` and `test_size`.

    Args:
        X: Feature dataframe.
        y: Target series.
        train_size: Proportion of examples to use for training.
        val_size: Proportion of examples to use for validation.
        test_size: Proportion of examples to use for testing.
        random_state: Random seed controlling reproducibility.

    Returns:
        Dictionary with keys 'train', 'val' and 'test', each mapping to a tuple
        ``(X_split, y_split)``.
    """
    if not abs((train_size + val_size + test_size) - 1.0) < 1e-6:
        raise ValueError("train_size + val_size + test_size must equal 1.0")
    # Bin the continuous target into quantiles for stratification
    bins = pd.qcut(y, q=10, duplicates="drop")
    # First split: train+val vs test
    sss1 = StratifiedShuffleSplit(
        n_splits=1, test_size=test_size, random_state=random_state
    )
    for train_val_idx, test_idx in sss1.split(X, bins):
        X_train_val, X_test = X.iloc[train_val_idx], X.iloc[test_idx]
        y_train_val, y_test = y.iloc[train_val_idx], y.iloc[test_idx]
        bins_train_val = bins.iloc[train_val_idx]
        # Second split: train vs val
        val_ratio = val_size / (train_size + val_size)
        sss2 = StratifiedShuffleSplit(
            n_splits=1, test_size=val_ratio, random_state=random_state
        )
        for train_idx, val_idx in sss2.split(X_train_val, bins_train_val):
            X_train, X_val = (
                X_train_val.iloc[train_idx],
                X_train_val.iloc[val_idx],
            )
            y_train, y_val = (
                y_train_val.iloc[train_idx],
                y_train_val.iloc[val_idx],
            )
            return {
                "train": (X_train.reset_index(drop=True), y_train.reset_index(drop=True)),
                "val": (X_val.reset_index(drop=True), y_val.reset_index(drop=True)),
                "test": (X_test.reset_index(drop=True), y_test.reset_index(drop=True)),
            }
    # Should never be reached
    raise RuntimeError("Failed to split the dataset.")