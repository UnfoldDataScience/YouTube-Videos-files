import streamlit as st
import openai
import os
from dotenv import load_dotenv

env_path = r'E:\YTReusable\.env' #Or what ever your .env path is.
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_email(relation, context):
    messages = [
        {"role": "system", "content": "You are a professional email writer. Create clear and concise emails based on relation and context."},
        {"role": "user", "content": f"Write an email with the following relation: {relation} and context: {context}."}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or another suitable chat model
            messages=messages,
            max_tokens=300
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        return f"An error occurred: {e}"


st.title("Email Generator")
relation = st.text_input("Relation (e.g., Colleague, Friend, Client):")
context = st.text_input("Context (e.g., Meeting Request, Follow-up, Thank You):")
if relation and context:
    email = generate_email(relation, context)
    st.write(email)