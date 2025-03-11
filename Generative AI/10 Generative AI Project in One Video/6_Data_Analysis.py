import streamlit as st
import pandas as pd
import openai
import matplotlib.pyplot as plt
import seaborn as sns
import os
from dotenv import load_dotenv
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_insight(df, question):
    """Generates insights using an LLM."""
    prompt = f"Given the following dataset (first 5 rows):\n{df.head().to_string()}\n\nQuestion: {question}\n\nAnswer:"
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=300,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"
def generate_visualization(df, column_x, column_y=None, vis_type="hist"):
    """Generates a visualization."""
    try:
        plt.figure(figsize=(10, 6))
        if vis_type == "hist":
            sns.histplot(df[column_x])
        elif vis_type == "scatter" and column_y:
            sns.scatterplot(x=column_x, y=column_y, data=df)
        else:
            return "Invalid visualization request."
        st.pyplot(plt)
        return "Visualization generated."
    except Exception as e:
        return f"Error generating visualization: {e}"

st.title("Data Exploration and Insight Generation")

uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    st.write("### Dataset Preview")
    st.dataframe(df.head())

    st.write("### Dataset Information")
    st.write(f"Number of rows: {df.shape[0]}")
    st.write(f"Number of columns: {df.shape[1]}")
    st.write(f"Column names: {', '.join(df.columns)}")

    question = st.text_input("Ask a question about the data:")
    if question:
        insight = generate_insight(df, question)
        st.write("### Generated Insight")
        st.write(insight)

    st.write("### Data Visualization")
    col_x = st.selectbox("Select X-axis column", df.columns)
    col_y = st.selectbox("Select Y-axis column (for scatter plot)", [None] + list(df.columns))
    vis_type = st.selectbox("Select visualization type", ["hist", "scatter"])
    if st.button("Generate Visualization"):
        vis_result = generate_visualization(df, col_x, col_y if vis_type == "scatter" else None, vis_type)
        st.write(vis_result)