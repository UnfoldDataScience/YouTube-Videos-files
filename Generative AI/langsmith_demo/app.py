from flask import Flask, render_template, request, jsonify
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from flask_cors import CORS

import openai
from dotenv import load_dotenv
# Load the .env file from local
env_path = r'E:\YTReusable\.env'
load_dotenv(env_path)

app = Flask(__name__)
CORS(app)

def initialize_rag():
    # Load PDF documents
    loader = DirectoryLoader('documentation/', glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()

    # Split documents into chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    texts = text_splitter.split_documents(documents)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)

    # Create RAG chain
    llm = ChatOpenAI(model_name="gpt-3.5-turbo")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )

    return qa_chain

# Initialize the RAG system
qa_chain = initialize_rag()

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    question = request.form['question']
    answer = qa_chain.run(question)
    print(answer)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)



