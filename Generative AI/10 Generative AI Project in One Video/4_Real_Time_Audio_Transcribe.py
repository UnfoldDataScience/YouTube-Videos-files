import streamlit as st
import openai
import os
import sounddevice as sd
import numpy as np
import soundfile as sf
import tempfile
from dotenv import load_dotenv
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")

def record_and_transcribe():
    fs = 44100  # Sample rate
    seconds = 5  # Duration of recording

    st.info("Recording... Please speak clearly.")
    try:
        myrecording = sd.rec(int(seconds * fs), samplerate=fs, channels=1)
        sd.wait()
        myrecording = np.squeeze(myrecording)
    except Exception as e:
        st.error(f"Error during recording: {e}")
        return None

    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as temp_audio:
            sf.write(temp_audio.name, myrecording, fs)
            audio_file = open(temp_audio.name, "rb")
            st.info("Transcribing...")
            transcript = openai.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file
            )
            return transcript.text
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return None
st.title("Real-time Audio Transcription")

if st.button("Record and Transcribe"):
    transcript = record_and_transcribe()
    if transcript:
        st.write("Transcript:", transcript)
        