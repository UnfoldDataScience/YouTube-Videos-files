import streamlit as st
import openai
import os

from dotenv import load_dotenv
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")


if 'chat_log' not in st.session_state:
    st.session_state.chat_log = []

def chat_with_gpt(prompt, history):
    messages = [{"role": "system", "content": "You are a helpful assistant."},]
    for message in history:
        if "You:" in message:
            messages.append({"role": "user", "content": message.replace("You:","")})
        if "Bot:" in message:
            messages.append({"role": "assistant", "content": message.replace("Bot:", "")})

    messages.append({"role": "user", "content": prompt})

    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Or another suitable model
        messages=messages,
        max_tokens=150
    )
    return response.choices[0].message.content.strip()


st.title("Contextual Chatbot")
user_input = st.text_input("You:")
if user_input:
    st.session_state.chat_log.append(f"You: {user_input}")
    response = chat_with_gpt(user_input, st.session_state.chat_log)
    st.session_state.chat_log.append(f"Bot: {response}")
    for message in st.session_state.chat_log:
        st.write(message)
