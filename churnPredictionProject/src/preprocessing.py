# src/preprocess.py
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load new data
new_data = pd.read_csv('./data/new_churn_data.csv')

new_data = new_data[["tenure", "MonthlyCharges","gender", "Contract","Churn"]]


# Define preprocess pipeline
numerical_features = ["tenure", "MonthlyCharges"]
categorical_features = ["gender", "Contract"]

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features)
    ])

# Transform data
preprocessed_data = preprocessor.fit_transform(new_data)
preprocessed_data_df = pd.DataFrame(preprocessed_data)
preprocessed_data_df.columns = ["tenure", "MonthlyCharges","gender_0","gender_1", "Contract_0","Contract_1","Churn"]
preprocessed_data_df.to_csv('./data/preprocessed_data.csv', index=False)