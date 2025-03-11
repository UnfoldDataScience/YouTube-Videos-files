import streamlit as st
import openai
import os
from dotenv import load_dotenv
from PIL import Image
import io
import base64
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_recipe(image_file):
    try:
        # Convert uploaded image to base64
        img = Image.open(image_file)
        buffered = io.BytesIO()
        img.save(buffered, format="JPEG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        response = openai.chat.completions.create(
            model="gpt-4o", #Correct Model.
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "What is this food? Generate a recipe for it."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{img_base64}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=500,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        st.error(f"An error occurred: {e}")
        return None

st.title("Recipe from Image")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    with st.spinner("Generating recipe..."):
        recipe = generate_recipe(uploaded_file)
        if recipe:
            st.write(recipe)