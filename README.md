# 📚 Personal RAG Chatbot

A fully functional Retrieval-Augmented Generation (RAG) chatbot that allows you to chat with your own PDF documents. Upload a PDF, and the app will index it locally and answer your questions accurately based on the document's content.

🔗 **Live Demo:** https://personal-rag-chatbot.streamlit.app/

## 🚀 Features

- **Document Ingestion:** Upload one or multiple PDF files directly through the web interface.
- **Smart Chunking:** Automatically splits documents into overlapping chunks for optimal retrieval.
- **Local Vector Embeddings:** Uses `sentence-transformers` to generate embeddings locally on the machine, completely avoiding third-party API limits and costs.
- **Vector Search:** Uses ChromaDB to store embeddings and retrieve the most relevant context for your question.
- **LLM Integration:** Powered by Groq (OpenAI GPT-OSS 20B) for blazing-fast, context-aware answers.
- **Source-Bounded:** The prompt is engineered to answer *only* based on the provided context, minimizing hallucinations.
- **Streamlit UI:** A clean, chat-based interface with streaming responses and document management.

## 🛠️ Tech Stack

- **Frontend:** Streamlit
- **Orchestration:** LangChain
- **Vector Database:** ChromaDB
- **Embeddings:** HuggingFace (Local via `sentence-transformers/all-MiniLM-L6-v2`)
- **LLM:** Groq API (`openai/gpt-oss-20b`)
- **Document Parsing:** PyPDF

## 📂 Project Structure

```text
personal-rag-chatbot/
│
├── app.py                  # Main Streamlit application and RAG logic
├── requirements.txt        # Python dependencies
├── .gitignore              # Files ignored by Git (e.g., secrets, temp files)
├── README.md               # Project documentation
└── .streamlit/
    └── secrets.toml        # (Local only) API key for Groq