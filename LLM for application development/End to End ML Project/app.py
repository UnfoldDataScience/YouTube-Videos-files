"""Streamlit UI for the California Housing Regressor.

This app allows users to train different regression models on the California
Housing dataset, explore model performance through interactive plots, and
make single predictions with optional SHAP explainability.  Use the sidebar
to configure preprocessing steps and model hyperparameters, then press
"Train / Retrain" to fit the model.  Metrics and plots will update after
training.  You can also input feature values manually to obtain a predicted
median house price and view a SHAP waterfall plot when available.
"""

from __future__ import annotations

import functools
import os
from typing import Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from src.data import load_dataset, stratified_split
from src.features import build_preprocessor
from src.models import get_model, get_param_distributions
from src.utils import load_config, set_seed
from src.viz import parity_plot, residuals_plot, error_hist_plot, feature_importance_plot


def load_splits(config: Dict) -> Dict[str, tuple[pd.DataFrame, pd.Series]]:
    """Helper to load and split the dataset once using Streamlit caching."""
    @st.cache_data  # type: ignore[misc]
    def _load():
        X, y = load_dataset()
        splits = stratified_split(
            X,
            y,
            train_size=config["split"]["train_size"],
            val_size=config["split"]["val_size"],
            test_size=config["split"]["test_size"],
            random_state=config["training"]["random_state"],
        )
        return splits

    return _load()


def train_model(
    model_name: str,
    use_pca: bool,
    use_poly: bool,
    hyperparams: Dict[str, any],
    config: Dict,
    fast: bool,
) -> Dict[str, any]:
    """Train a model on the training data and return metrics and the fitted pipeline."""
    set_seed(config["training"]["random_state"])
    splits = load_splits(config)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]
    preprocessor = build_preprocessor(use_pca=use_pca, use_poly=use_poly)
    model = get_model(model_name)
    # Override model hyperparameters with user inputs
    for param, value in hyperparams.items():
        if value is not None:
            setattr(model, param, value)
    from sklearn.pipeline import Pipeline

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", model),
    ])
    # Fit on training and validation sets combined
    # Train on the training set
    pipeline.fit(X_train, y_train)
    # Compute predictions and metrics on test set
    y_pred = pipeline.predict(X_test)
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

    rmse = mean_squared_error(y_test, y_pred, squared=False)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    metrics = {"rmse": rmse, "mae": mae, "r2": r2}
    return {
        "pipeline": pipeline,
        "metrics": metrics,
        "y_test": y_test,
        "y_pred": y_pred,
        "X_test": X_test,
    }


def shap_available() -> bool:
    """Determine whether the shap library is installed."""
    try:
        import shap  # type: ignore

        _ = shap  # prevent unused import warning
        return True
    except Exception:
        return False


def main() -> None:
    """Entrypoint for the Streamlit app."""
    st.set_page_config(
        page_title="Housing Regressor",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("End to End Machine Learning Pipeline Demo")
    config = load_config("config/config.yaml")
    # Sidebar configuration
    st.sidebar.header("Configuration")
    # Determine available models
    available_models: List[str] = ["linear", "random_forest", "gradient_boosting"]
    try:
        # Check if XGBoost is available via get_model
        get_model("xgboost")
        available_models.append("xgboost")
    except Exception:
        pass
    model_name = st.sidebar.selectbox("Model", options=available_models, index=available_models.index("random_forest"))
    use_pca = st.sidebar.checkbox("Use PCA", value=False)
    use_poly = st.sidebar.checkbox("Use Polynomial Features", value=False)
    # Hyperparameter inputs vary by model
    hyperparams: Dict[str, any] = {}
    st.sidebar.subheader("Hyperparameters")
    if model_name == "random_forest":
        n_estimators = st.sidebar.slider("n_estimators", 50, 200, 100, step=10)
        max_depth = st.sidebar.selectbox("max_depth", options=[None, 5, 10, 15], index=0)
        min_samples_split = st.sidebar.slider("min_samples_split", 2, 10, 2)
        min_samples_leaf = st.sidebar.slider("min_samples_leaf", 1, 4, 1)
        hyperparams = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_split": min_samples_split,
            "min_samples_leaf": min_samples_leaf,
        }
    elif model_name == "gradient_boosting":
        n_estimators = st.sidebar.slider("n_estimators", 50, 200, 100, step=10)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.05, step=0.01)
        max_depth = st.sidebar.slider("max_depth", 2, 6, 3)
        subsample = st.sidebar.slider("subsample", 0.5, 1.0, 1.0, step=0.1)
        hyperparams = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
        }
    elif model_name == "xgboost":
        n_estimators = st.sidebar.slider("n_estimators", 50, 200, 100, step=10)
        learning_rate = st.sidebar.slider("learning_rate", 0.01, 0.3, 0.05, step=0.01)
        max_depth = st.sidebar.slider("max_depth", 3, 8, 4)
        subsample = st.sidebar.slider("subsample", 0.5, 1.0, 1.0, step=0.1)
        colsample_bytree = st.sidebar.slider("colsample_bytree", 0.5, 1.0, 1.0, step=0.1)
        hyperparams = {
            "n_estimators": n_estimators,
            "learning_rate": learning_rate,
            "max_depth": max_depth,
            "subsample": subsample,
            "colsample_bytree": colsample_bytree,
        }
    # Button to train/retrain
    if "trained" not in st.session_state:
        st.session_state.trained = False
    if st.sidebar.button("Train / Retrain"):
        with st.spinner("Training model..."):
            result = train_model(
                model_name=model_name,
                use_pca=use_pca,
                use_poly=use_poly,
                hyperparams=hyperparams,
                config=config,
                fast=False,
            )
        st.session_state.pipeline = result["pipeline"]
        st.session_state.metrics = result["metrics"]
        st.session_state.y_test = result["y_test"]
        st.session_state.y_pred = result["y_pred"]
        st.session_state.X_test = result["X_test"]
        st.session_state.trained = True
    # If model is trained, display metrics and plots
    if st.session_state.get("trained", False):
        metrics = st.session_state.metrics
        st.subheader("Test Metrics")
        col1, col2, col3 = st.columns(3)
        col1.metric("RMSE", f"{metrics['rmse']:.3f}")
        col2.metric("MAE", f"{metrics['mae']:.3f}")
        col3.metric("R²", f"{metrics['r2']:.3f}")
        # Show plots
        st.subheader("Diagnostics")
        plots_cols = st.columns(3)
        # Parity plot
        parity_path = os.path.join("artifacts", "plots", "parity.png")
        residuals_path = os.path.join("artifacts", "plots", "residuals.png")
        error_hist_path = os.path.join("artifacts", "plots", "error_hist.png")
        # Regenerate plots based on current predictions
        parity_plot(st.session_state.y_test, st.session_state.y_pred, parity_path)
        residuals_plot(st.session_state.y_test, st.session_state.y_pred, residuals_path)
        error_hist_plot(st.session_state.y_test, st.session_state.y_pred, error_hist_path)
        plots_cols[0].image(parity_path, caption="Parity Plot", use_column_width=True)
        plots_cols[1].image(residuals_path, caption="Residuals Plot", use_column_width=True)
        plots_cols[2].image(error_hist_path, caption="Error Histogram", use_column_width=True)
        # Feature importance
        fi_path = os.path.join("artifacts", "plots", "feature_importance.png")
        feature_importance_plot(
            getattr(st.session_state.pipeline, "named_steps", {}).get("model", st.session_state.pipeline),
            st.session_state.X_test.columns.tolist(),
            fi_path,
        )
        st.image(fi_path, caption="Feature Importance", use_column_width=True)
        # SHAP summary if available
        if shap_available():
            shap_summary_path = os.path.join("artifacts", "plots", "shap_summary.png")
            shap_waterfall_path = os.path.join("artifacts", "plots", "shap_waterfall.png")
            import shap  # type: ignore
            import matplotlib.pyplot as plt
            # Compute SHAP once per model training
            if "shap_values" not in st.session_state:
                try:
                    preprocessor = st.session_state.pipeline.named_steps["preprocessor"]
                    model = st.session_state.pipeline.named_steps["model"]
                    # Sample a subset for efficiency
                    sample_idx = np.random.choice(
                        len(st.session_state.X_test),
                        size=min(200, len(st.session_state.X_test)),
                        replace=False,
                    )
                    X_sample = st.session_state.X_test.iloc[sample_idx]
                    X_processed = preprocessor.transform(X_sample)
                    explainer = shap.Explainer(model, X_processed)
                    shap_values = explainer(X_processed)
                    st.session_state.shap_values = shap_values
                    st.session_state.shap_sample = X_sample
                    # Summary plot
                    shap.plots.bar(shap_values, show=False)
                    plt.tight_layout()
                    plt.savefig(shap_summary_path)
                    plt.close()
                    # Waterfall plot for first sample
                    shap.plots.waterfall(shap_values[0], show=False)
                    plt.tight_layout()
                    plt.savefig(shap_waterfall_path)
                    plt.close()
                except Exception as exc:
                    st.session_state.shap_values = None
                    st.warning(f"SHAP computation failed: {exc}")
            # Display SHAP plots if they exist
            if os.path.exists(shap_summary_path):
                st.image(shap_summary_path, caption="SHAP Summary", use_column_width=True)
            if os.path.exists(shap_waterfall_path):
                st.image(shap_waterfall_path, caption="SHAP Waterfall (First Sample)", use_column_width=True)
        else:
            st.info("SHAP is not installed; install it to see explanations.")
        # Single prediction form
        st.subheader("Single Prediction")
        with st.form("prediction_form"):
            feature_inputs = {}
            # Show number input for each feature
            for col in st.session_state.X_test.columns:
                # Use reasonable defaults based on test sample median
                default_val = float(st.session_state.X_test[col].median())
                feature_inputs[col] = st.number_input(col, value=default_val)
            submitted = st.form_submit_button("Predict")
            if submitted:
                # Convert to DataFrame for prediction
                input_df = pd.DataFrame([feature_inputs])
                prediction = st.session_state.pipeline.predict(input_df)[0]
                st.write(f"**Predicted Median House Value:** {prediction:.3f}")
                # Local SHAP for this instance if available
                if shap_available() and "pipeline" in st.session_state:
                    try:
                        preprocessor = st.session_state.pipeline.named_steps["preprocessor"]
                        model = st.session_state.pipeline.named_steps["model"]
                        processed = preprocessor.transform(input_df)
                        import shap  # type: ignore

                        explainer = shap.Explainer(model, processed)
                        shap_values_single = explainer(processed)
                        import matplotlib.pyplot as plt
                        shap_waterfall_temp = os.path.join("artifacts", "plots", "shap_single_temp.png")
                        shap.plots.waterfall(shap_values_single[0], show=False)
                        plt.tight_layout()
                        plt.savefig(shap_waterfall_temp)
                        plt.close()
                        st.image(shap_waterfall_temp, caption="SHAP Explanation", use_column_width=True)
                    except Exception as exc:
                        st.warning(f"Failed to compute local SHAP explanation: {exc}")
    else:
        st.info("Use the sidebar to configure your model and click 'Train / Retrain' to begin.")


if __name__ == "__main__":  # pragma: no cover
    main()