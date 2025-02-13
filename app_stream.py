import streamlit as st
import pandas as pd
import os
from dotenv import load_dotenv
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.prompt import system_prompt

# Load environment variables
load_dotenv()

# Set page config
st.set_page_config(
    page_title="Medical Chatbot",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for enhanced UI
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
    }
    
    /* Chat message bubbles */
    .stChatMessage {
        padding: 12px 16px;
        border-radius: 12px;
        margin-bottom: 10px;
        max-width: 75%;
    }
    .user-message {
        background-color: #0078d4;
        color: white;
        text-align: right;
        padding: 12px;
        border-radius: 16px 16px 0 16px;
        margin-left: auto;
    }
    .assistant-message {
        background-color: #f1f1f1;
        color: black;
        text-align: left;
        padding: 12px;
        border-radius: 16px 16px 16px 0;
        margin-right: auto;
    }
    
    /* Sidebar customization */
    .sidebar .sidebar-content {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 12px;
    }
    .sidebar .stRadio label {
        font-weight: bold;
    }
    
    /* Header styling */
    h1, h2, h3 {
        color: #2c3e50;
        text-align: center;
    }
    
    /* Input box */
    .stChatInput {
        width: 100%;
        border-radius: 12px;
        border: 1px solid #ddd;
        padding: 10px;
    }
    </style>
    """, unsafe_allow_html=True)

# Load API keys
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Cache embeddings
@st.cache_resource
def load_embeddings():
    return download_hugging_face_embeddings()

# Cache Pinecone retriever
@st.cache_resource
def load_retriever():
    embeddings = load_embeddings()
    index_name = "medicalbot"
    docsearch = PineconeVectorStore.from_existing_index(index_name=index_name, embedding=embeddings)
    return docsearch.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Cache LLM model
@st.cache_resource
def load_llm():
    return ChatGroq(
        temperature=0.4,
        max_tokens=500,
        model_name="llama-3.3-70b-versatile"
    )

# Initialize session state
if "initialized" not in st.session_state:
    st.session_state["retriever"] = load_retriever()
    st.session_state["llm"] = load_llm()
    st.session_state["initialized"] = True

# Create RAG chain
prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}"),
])

question_answer_chain = create_stuff_documents_chain(st.session_state["llm"], prompt)
rag_chain = create_retrieval_chain(st.session_state["retriever"], question_answer_chain)

# Sidebar Navigation
st.sidebar.title("📌 Navigation")
st.sidebar.markdown("---")
page = st.sidebar.radio("Go to:", ["💬 Chat", "ℹ️ About"], index=0)

if page == "💬 Chat":
    st.title("🩺 Medical Chatbot")
    st.markdown("Ask me anything about health! 💬")

    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state["messages"] = []

    # Display chat history
    for message in st.session_state["messages"]:
        role, content = message["role"], message["content"]
        with st.chat_message(role):
            st.markdown(f'<div class="{role}-message">{content}</div>', unsafe_allow_html=True)

    # Chat input
    user_input = st.chat_input("Type your message here...")

    if user_input:
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(f'<div class="user-message">{user_input}</div>', unsafe_allow_html=True)
        
        with st.spinner("Thinking..."):
            response = rag_chain.invoke({"input": user_input})
            bot_response = response["answer"]
        
        st.session_state["messages"].append({"role": "assistant", "content": bot_response})
        with st.chat_message("assistant"):
            st.markdown(f'<div class="assistant-message">{bot_response}</div>', unsafe_allow_html=True)

elif page == "ℹ️ About":
    st.title("ℹ️ About the Medical Chatbot")
    st.markdown("""
    This AI-powered **Medical Chatbot** provides instant responses to health-related inquiries using **NLP** and **RAG (Retrieval-Augmented Generation)**.
    
    **🔹 Features:**
    - 📚 Retrieves relevant health information using **Pinecone Vector Database**
    - 🧠 Generates responses with **LLama-3.3-70B Versatile**
    - ⚡ Fast and interactive chat experience
    
    **🛠️ Built With:**
    - 🏗️ **LangChain** for response generation
    - 🏎️ **Streamlit** for UI
    - 🔎 **Pinecone** for document retrieval
    - 🤖 **Groq LLM** for AI-generated answers
    
    **⚠️ Disclaimer:**
    This chatbot provides **informational responses only**. It does **not** offer medical advice, diagnosis, or treatment. Always consult a healthcare professional for medical concerns.
    
    **👨‍💻 Developed by:** Hasini Uppaluri  
    📧 Email: [uppalurihasini@gmail.com](mailto:uppalurihasini@gmail.com)  
    🔗 GitHub: [github.com/hspsnm12](https://github.com/hspsnm12)  
    🔗 LinkedIn: [linkedin.com/in/hasiniuppaluri](https://www.linkedin.com/in/hasini-uppaluri-387a592a2/)
    """, unsafe_allow_html=True)
