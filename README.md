# AI_Health_Assistant

An intelligent, context-aware health assistant that answers medical queries using Retrieval-Augmented Generation (RAG).  
Built with LangChain, Groq LLM, Hugging Face Embeddings, and Pinecone vector DB.

---

## 💡 What It Does

- Parses a medical textbook (PDF)
- Splits content into text chunks and embeds them
- Stores embeddings in **Pinecone**
- On user input, retrieves relevant context
- Passes context and query to **Groq LLM**
- Displays safe, structured medical advice — with fallbacks for low-confidence cases

---

## 🧠 Technologies & Tools

| Stack              | Purpose                        |
|-------------------|--------------------------------|
| Streamlit         | UI                             |
| LangChain         | RAG pipeline + LLM interface   |
| Groq              | LLM used for generating responses |
| Pinecone          | Vector DB for document retrieval |
| HuggingFace       | Embedding model                |
| dotenv            | Environment management         |

---

## ▶️ How to Run

1. Clone the repo:
   ```bash
   git clone https://github.com/your-username/AI_Health_Assistant.git
   cd AI_Health_Assistant
Set up environment:

bash
pip install -r requirements.txt
Add your .env:

env
PINECONE_API_KEY=your_key
GROQ_API_KEY=your_key
Create the vector index:

bash
python store_index.py
Launch the app:

bash
streamlit run app_stream.py
🎯 Highlights
✅ Real-world RAG pipeline

✅ Modular and production-ready

✅ Custom prompt ensures safety in medical domain

✅ Demonstrates prompt engineering and retrieval control
