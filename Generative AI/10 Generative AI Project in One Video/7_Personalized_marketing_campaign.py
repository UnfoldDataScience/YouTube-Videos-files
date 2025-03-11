import streamlit as st
import openai
import os
import pandas as pd

from dotenv import load_dotenv
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")
def generate_email(customer_data):
    messages = [
        {"role": "system", "content": "You are a marketing email generator."},
        {"role": "user", "content": f"Generate a personalized marketing email using the following customer data: {customer_data}"}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"

st.title("Personalized Marketing Emails")
uploaded_file = st.file_uploader("Upload Customer Data (CSV):", type="csv")
if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("Customer Data:", df)
    if st.button("Generate Emails"):
        for index, row in df.iterrows():
            email = generate_email(row.to_dict())
            st.write(f"Email for {row['name']}:", email)