import streamlit as st
import numpy as np
import joblib

# Load the trained model
model = joblib.load('./models/model_rf_v1.pkl')

# Title of the application
st.title("Customer Churn Prediction")

# Sidebar for user inputs
st.sidebar.header("Customer Input Features")

# Function to get user input from UI
def get_user_input():
    tenure = st.sidebar.slider("Customer Tenure (months)", 0, 72, 36)
    monthly_charges = st.sidebar.slider("Monthly Charges", 0, 120, 60)
    gender_f = st.sidebar.selectbox("Gender Female", (1,0))
    gender_m = st.sidebar.selectbox("Gender Male", (1,0))
    contract_one_year = st.sidebar.selectbox("Contract 1 year", (1,0))
    contract_two_year = st.sidebar.selectbox("Contract 2 year", (1,0))
    # gender_m = st.sidebar.selectbox("Gender Male", (1,0))
    # contract = st.sidebar.selectbox("Contract Type", ("Month-to-month", "One year", "Two year"))

    # Encode categorical features manually (since model is expecting numerical data)
    # gender = 1 if gender == "Male" else 0
    # contract_one_year = 1 if contract == "One year" else 0
    # contract_two_year = 1 if contract == "Two year" else 0

    features = np.array([tenure, monthly_charges, gender_f,gender_m, contract_one_year, contract_two_year])
    return features

# Collect user input
user_input = get_user_input()

# Make predictions based on user input
prediction = model.predict([user_input])

# Display the prediction
st.subheader("Prediction:")
if prediction[0] == 1:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is not likely to churn.")

# Display the feature inputs
# st.subheader("Input Features:")
# st.write(f"Tenure: {user_input[0]} months")
# st.write(f"Monthly Charges: ${user_input[1]}")
# st.write(f"Gender: {'Male' if user_input[2] == 1 else 'Female'}")
# st.write(f"Contract Type: {'One year' if user_input[3] == 1 else 'Two year' if user_input[4] == 1 else 'Month-to-month'}")
