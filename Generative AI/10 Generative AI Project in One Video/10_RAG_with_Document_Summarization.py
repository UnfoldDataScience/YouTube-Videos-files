import streamlit as st
import os
import openai
from sentence_transformers import SentenceTransformer, util
import PyPDF2

from dotenv import load_dotenv
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)

openai.api_key = os.getenv("OPENAI_API_KEY")
model = SentenceTransformer('all-MiniLM-L6-v2')

def load_pdf(uploaded_file):
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.getvalue())

    with open("temp.pdf", 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        text = ""
        for page_num in range(len(pdf_reader.pages)):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()
    return text

def chunk_text(text, chunk_size=500, overlap=100):
    chunks = []
    for i in range(0, len(text), chunk_size - overlap):
        chunks.append(text[i:i + chunk_size])
    return chunks


def create_embeddings(chunks):
    embeddings = model.encode(chunks, convert_to_tensor=True)
    return embeddings

def find_relevant_chunks(query_embedding, chunk_embeddings, chunks, top_k=3):
    cosine_scores = util.pytorch_cos_sim(query_embedding, chunk_embeddings)[0]
    top_results = sorted(range(len(cosine_scores)), key=lambda i: cosine_scores[i], reverse=True)[:top_k]
    relevant_chunks = [chunks[i] for i in top_results]
    return relevant_chunks

def generate_response(query, context):
    messages = [
        {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
        {"role": "user", "content": f"Context: {context}\nQuestion: {query}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",  # Or another suitable chat model
        messages=messages,
        max_tokens=200
    )

    return response.choices[0].message.content.strip()

st.title("Simple RAG Application (No LangChain)")
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file:
    with st.spinner("Processing PDF..."):
        pdf_text = load_pdf(uploaded_file)
        chunks = chunk_text(pdf_text)
        chunk_embeddings = create_embeddings(chunks)

        query = st.text_input("Ask a question:")
        if query:
            query_embedding = model.encode([query], convert_to_tensor=True)
            relevant_chunks = find_relevant_chunks(query_embedding, chunk_embeddings, chunks)
            context = "\n".join(relevant_chunks)
            answer = generate_response(query, context)
            st.write("Answer:", answer)