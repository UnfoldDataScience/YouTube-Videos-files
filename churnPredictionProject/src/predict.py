import joblib
import numpy as np
import pandas as pd

# Load the trained model
model = joblib.load('./models/model_rf_v1.pkl')

# Function to make predictions
def predict(features):
    prediction = model.predict([features])
    return prediction

# Example new customer data for prediction
# You can replace this with actual new data from a CSV or user input
new_customer_data = np.array([25, 70, 1, 0,0,1])  # Example data: [tenure, monthly_charges, gender, contract_one_year, contract_two_year]

# Make prediction for the new customer
prediction = predict(new_customer_data)

# Output the prediction result
if prediction[0] == 1:
    print("The customer is likely to churn.")
else:
    print("The customer is not likely to churn.")
