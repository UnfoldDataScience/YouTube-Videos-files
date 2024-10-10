# src/data_ingestion.py
import pandas as pd
import requests
import kaggle as kg

def fetch_new_data():
    kg.api.dataset_download_files('blastchar/telco-customer-churn',path='on.zip',unzip=True)
    df = pd.read_csv('on.zip/WA_Fn-UseC_-Telco-Customer-Churn.csv', encoding='ISO-8859-1')
    return(df)

# Save new data for further processing
new_data = fetch_new_data()
new_data.to_csv('./data/new_churn_data.csv', index=False)