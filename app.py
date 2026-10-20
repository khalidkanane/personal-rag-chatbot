import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# --- Setup Streamlit UI ---
st.set_page_config(page_title="Personal RAG Chatbot", page_icon="📚")
st.title("📚 Personal RAG Chatbot")

# Load API key from secrets
# Note: We no longer need the HuggingFace token because embeddings run locally!
groq_api_key = st.secrets["GROQ_API_KEY"]

# --- Sidebar: Document Upload & Indexing ---
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing and indexing documents..."):
            docs = []
            # Save uploaded files temporarily and load them
            for uploaded_file in uploaded_files:
                temp_file_path = f"./temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_file_path)
                docs.extend(loader.load())
                os.remove(temp_file_path) # Clean up temp file
            
            # Split text into chunks
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            # Create LOCAL embeddings (No API calls, no billing issues)
            embeddings = HuggingFaceEmbeddings(
                model_name="sentence-transformers/all-MiniLM-L6-v2"
            )
            
            # Delete old DB if it exists to avoid mixing documents
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
            
            # Save to session state so it persists across reruns
            st.session_state.vectorstore = vectorstore
            st.success(f"Processed {len(splits)} chunks from {len(uploaded_files)} files!")

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Check if documents are indexed
    if "vectorstore" not in st.session_state:
        with st.chat_message("assistant"):
            st.error("Please upload and process documents in the sidebar first!")
    else:
        # Generate response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # 1. Setup Retriever
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                
                # 2. Setup LLM (Updated to the current working Groq model)
                llm = ChatGroq(
                    temperature=0, 
                    groq_api_key=groq_api_key, 
                    model_name="openai/gpt-oss-20b"
                )
                
                # 3. Prompt Template
                template = """Answer the question based only on the following context:
                {context}

                Question: {question}
                
                If you don't know the answer based on the context, just say "I don't know based on the provided documents."
                """
                prompt_template = ChatPromptTemplate.from_template(template)
                
                # 4. RAG Chain
                rag_chain = (
                    {"context": retriever, "question": RunnablePassthrough()}
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                # 5. Stream the answer
                response = st.write_stream(rag_chain.stream(prompt))
        
        # Save assistant response to history
        st.session_state.messages.append({"role": "assistant", "content": response})