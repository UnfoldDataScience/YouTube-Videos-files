# California Housing Regressor

**California Housing Regressor** is an end‑to‑end machine learning project that predicts median house prices from the well‑known California Housing dataset.  It exposes both a command line interface and an interactive [Streamlit](https://streamlit.io/) web application for training, evaluation, and exploration of the model.  The project is designed to be reproducible, easy to run locally, and includes automated tests, plots, and optional explainability via [SHAP](https://github.com/shap/shap).

## Project Overview

This repository contains everything needed to fetch the dataset, build preprocessing pipelines, train and tune regression models, evaluate their performance, and serve predictions through a simple web UI.  Key features include:

- **Data ingestion**: Loads the California Housing dataset from `sklearn.datasets.fetch_california_housing()` and performs a stratified train/validation/test split using quantile bins on the target to preserve the distribution across splits.
- **Preprocessing**: Applies standard scaling to all numerical features, with optional principal component analysis (PCA) and optional polynomial feature expansion.  These options can be toggled from the CLI or the Streamlit app.
- **Models**: Supports multiple regressors—`LinearRegression`, `RandomForestRegressor`, `GradientBoostingRegressor`, and, if installed, `XGBRegressor`.  Hyperparameters are tuned via `RandomizedSearchCV`.
- **Evaluation**: Computes root mean squared error (RMSE), mean absolute error (MAE), and the coefficient of determination (R²) on the validation and test sets.  Generates parity plots, residual plots, error histograms, and feature importance charts in `artifacts/plots/`.
- **Explainability**: Uses SHAP for global and local interpretability.  If SHAP is not installed, the scripts will skip these visualisations gracefully and log an informational message.
- **Streamlit UI**: An interactive app (`streamlit run app.py`) that lets you select models and preprocessing options, retrain the model, view evaluation metrics and plots, and make single predictions with optional SHAP explanations.
- **Testing**: The `tests/` directory contains PyTest suites that verify data splitting proportions, pipeline construction, CLI commands, and metrics file integrity.  Running `make test` executes these tests.

## Quickstart

To get started, clone the repository and create a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Then perform a quick train and evaluation run:

```bash
make all
```

This command trains the best model using the default configuration and produces evaluation metrics and plots under the `artifacts/` directory.

Launch the Streamlit application with:

```bash
make app
```

The web interface allows you to select a model, toggle PCA and polynomial features, retrain the model on demand, and view metrics and plots.  A form is provided for making individual predictions.  SHAP visualisations will appear if the `shap` package is installed; otherwise, the app will indicate that explainability is unavailable.

## Makefile Commands

The included `Makefile` simplifies common tasks:

| Target        | Description                                      |
|-------------- |--------------------------------------------------|
| `make setup`  | Create a virtual environment, install deps, and install pre‑commit hooks |
| `make train`  | Train the model using the default configuration |
| `make eval`   | Evaluate the saved model and write plots/metrics |
| `make app`    | Launch the Streamlit application               |
| `make test`   | Run all PyTest suites                          |
| `make lint`   | Run Ruff and Black in check mode               |
| `make format` | Format code with Black and sort imports with isort |
| `make all`    | Equivalent to `make train && make eval`         |

## Configuration

Default hyperparameters and split proportions are defined in `config/config.yaml`.  You can modify this file to change the train/validation/test ratios, the number of randomised search iterations, or the number of cross‑validation folds.  Command line arguments always override values in the configuration file.

## Troubleshooting

- **SHAP not installed**: If `shap` is not installed in your environment, SHAP plots and explanations will be skipped.  Install SHAP via `pip install shap` to enable interpretability features.
- **Missing XGBoost**: The XGBoost model is optional.  If the `xgboost` Python package is unavailable, the model selector will hide the XGBoost option automatically.
- **Long training times**: Use the `--fast` flag when running `src.train` or `src.evaluate` to reduce the number of hyperparameter search iterations and cross‑validation folds, which speeds up training and evaluation for testing purposes.

## License

This project is provided for educational purposes and is not intended for production use without additional hardening and testing.


How to run - alternate way (my machine only)
# already running, but for reference:

# Finish install (inside venv)
python -m pip install --upgrade pip setuptools wheel
$env:SETUPTOOLS_USE_DISTUTILS = 'local'   # safe to leave during install
python -m pip install --no-build-isolation --only-binary=:all: -r requirements.txt

# Train + evaluate (fast mode) - CLI mode
python -m src.train --model random_forest --fast
python -m src.evaluate --fast --skip-shap

streamlit run app.py