import streamlit as st
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

# Define Streamlit layout
st.title("DeepSeek-R1: Explore AI-Powered Insights")
st.write("An interactive app to showcase the capabilities of DeepSeek-R1. Enter a question and get concise and insightful answers.")


# User input section
question = st.text_input("Enter your question:")

# Initialize the model
@st.cache_resource
def load_model():
    return OllamaLLM(model="deepseek-r1:1.5b", base_url="http://127.0.0.1:11434")

model = load_model()
# Prompt template
template = """Question: {question} 
Answer: Let's keep it concise and actionable under 200 words"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

# Handle user interaction
if st.button("Get Answer"):
    if question.strip():
        try:
            with st.spinner("Thinking..."):
                response = chain.invoke({"question": question})
            st.success("### Answer:")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question to get an answer.")

# Footer
st.markdown("---")
st.write("Made with ❤️ using Streamlit and DeepSeek-R1")
