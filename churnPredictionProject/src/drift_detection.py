# src/drift_detection.py
from alibi_detect.cd import KSDrift
import pandas as pd

# Load old and new data for drift detection
X_train = pd.read_csv('./data/preprocessed_data.csv')
new_data = pd.read_csv('./data/new_churn_data.csv')

# Create drift detector
cd = KSDrift(X_train)

def detect_drift(new_data):
    drift_preds = cd.predict(new_data)
    if drift_preds['data']['is_drift']:
        print("Data drift detected. Retraining required.")
        from src.retrain_model import retrain_pipeline
        retrain_pipeline()

# Detect drift
detect_drift(new_data)
