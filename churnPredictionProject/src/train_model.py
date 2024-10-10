import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
import mlflow

def train_model(preprocessed_data, target_var):
    # Load preprocessed data
    data = pd.DataFrame(preprocessed_data)
    print(data.columns)
    X = data.drop(target_var, axis=1)
    y = data[target_var]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model training
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Save model version
    joblib.dump(rf_model, './models/model_rf_v1.pkl')

    # Log model performance using MLflow
    mlflow.log_param("model_version", "model_rf_v1")
    mlflow.log_metric("accuracy", rf_model.score(X_test, y_test))

preprocessed_data = pd.read_csv('./data/preprocessed_data.csv')
train_model(preprocessed_data, "Churn")