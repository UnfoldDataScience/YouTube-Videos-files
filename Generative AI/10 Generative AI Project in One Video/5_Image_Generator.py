import streamlit as st
import requests
import os
import openai
import tempfile
from PIL import Image
import io
from dotenv import load_dotenv
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_image(prompt):
    try:
        response = openai.images.generate(
            model="dall-e-3",
            prompt=prompt,
            n=1,
            size="1024x1024"
        )
        return response.data[0].url

    except openai.OpenAIError as e:
        st.error(f"Error generating image: {e}")
        return None
    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
        return None

st.title("DALL-E 3 Image Generator")
prompt = st.text_input("Enter a prompt:")

if prompt:
    with st.spinner("Generating image..."):
        image_url = generate_image(prompt)
        if image_url:
            st.image(image_url)

