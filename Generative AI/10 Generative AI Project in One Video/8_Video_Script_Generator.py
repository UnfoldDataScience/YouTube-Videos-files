import streamlit as st
import openai
import os
from dotenv import load_dotenv

env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)
openai.api_key = os.getenv("OPENAI_API_KEY")

def generate_video_script(blog_post):
    messages = [
        {"role": "system", "content": "You are a video script generator. Create engaging and concise video scripts from blog posts."},
        {"role": "user", "content": f"Create a video script from the following blog post:\n{blog_post}"}
    ]
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",  # Or another suitable chat model
            messages=messages,
            max_tokens=500
        )
        return response.choices[0].message.content.strip()
    except openai.OpenAIError as e:
        return f"An error occurred: {e}"

st.title("Video Script Generator")
blog_post = st.text_area("Enter Blog Post:")
if blog_post:
    script = generate_video_script(blog_post)
    st.write(script)