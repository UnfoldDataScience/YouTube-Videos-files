from data_ingestion import fetch_new_data
from preprocessing import preprocessor
from train_model import train_model
import pandas as pd
# Retrain the model with new data
def retrain_pipeline():
    # Fetch new data
    new_data = fetch_new_data()
    new_data = new_data[["tenure", "MonthlyCharges","gender", "Contract","Churn"]]
    
    # Preprocess new data
    preprocessed_data = preprocessor.transform(new_data)
    preprocessed_data_df = pd.DataFrame(preprocessed_data)
    preprocessed_data_df.columns = ["tenure", "MonthlyCharges","gender_0","gender_1", "Contract_0","Contract_1","Churn"]
    

    # Retrain the model
    model = train_model(preprocessed_data_df, "Churn")
    print("Model retrained and saved.")
    
# Schedule or trigger retraining
retrain_pipeline()
