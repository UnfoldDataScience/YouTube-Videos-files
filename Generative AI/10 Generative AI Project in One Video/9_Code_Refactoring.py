import streamlit as st
import openai
import os
from dotenv import load_dotenv
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

def refactor_code(code):
    """Refactors and optimizes the given code using OpenAI's Chat Completions API."""
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or "gpt-4" if you have access
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that refactors and optimizes code. Provide the refactored code directly, without explanation unless specifically requested.",
                },
                {
                    "role": "user",
                    "content": f"Refactor and optimize the following code:\n{code}",
                },
            ],
            max_tokens=800, #increased tokens as chat api is more verbose.
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"An error occurred: {e}"
st.title("Code Refactoring")
code = st.text_area("Enter Code:")
if code:
    refactored_code = refactor_code(code)
    st.code(refactored_code)