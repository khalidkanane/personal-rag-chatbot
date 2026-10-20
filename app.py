import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage

# --- Setup Streamlit UI ---
st.set_page_config(page_title="Personal RAG Chatbot", page_icon="📚")
st.title("📚 Personal RAG Chatbot")

groq_api_key = st.secrets["GROQ_API_KEY"]

# --- Sidebar: Document Management ---
with st.sidebar:
    st.header("Document Management")
    uploaded_files = st.file_uploader("Upload PDF documents", type="pdf", accept_multiple_files=True)
    
    if st.button("Process Documents") and uploaded_files:
        with st.spinner("Processing and indexing documents..."):
            docs = []
            for uploaded_file in uploaded_files:
                temp_file_path = f"./temp_{uploaded_file.name}"
                with open(temp_file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                loader = PyPDFLoader(temp_file_path)
                docs.extend(loader.load())
                os.remove(temp_file_path)
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            splits = text_splitter.split_documents(docs)
            
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            
            if os.path.exists("./chroma_db"):
                shutil.rmtree("./chroma_db")
                
            vectorstore = Chroma.from_documents(documents=splits, embedding=embeddings, persist_directory="./chroma_db")
            st.session_state.vectorstore = vectorstore
            st.success(f"Processed {len(splits)} chunks!")

    # NEW: Clear Chat Button
    if st.button("Clear Chat History"):
        st.session_state.messages = []
        st.rerun()

# --- Main Chat Interface ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history (UI)
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    if "vectorstore" not in st.session_state:
        with st.chat_message("assistant"):
            st.error("Please upload and process documents in the sidebar first!")
    else:
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                retriever = st.session_state.vectorstore.as_retriever(search_kwargs={"k": 4})
                llm = ChatGroq(temperature=0, groq_api_key=groq_api_key, model_name="openai/gpt-oss-20b")
                
                # NEW: Prompt Template with Chat History (Memory)
                prompt_template = ChatPromptTemplate.from_messages([
                    ("system", "Answer the question based only on the following context:\n\n{context}\n\nIf you don't know the answer, just say you don't know."),
                    MessagesPlaceholder(variable_name="chat_history"), # This adds the memory
                    ("human", "{question}")
                ])
                
                # NEW: Convert previous messages to LangChain Message objects
                chat_history = []
                for msg in st.session_state.messages[:-1]: # Exclude the current prompt
                    if msg["role"] == "user":
                        chat_history.append(HumanMessage(content=msg["content"]))
                    elif msg["role"] == "assistant":
                        chat_history.append(AIMessage(content=msg["content"]))
                
                # NEW: The RAG Chain with Memory and Source Extraction
                rag_chain = (
                    {
                        "context": retriever, 
                        "question": RunnablePassthrough(),
                        "chat_history": lambda _: chat_history
                    }
                    | prompt_template
                    | llm
                    | StrOutputParser()
                )
                
                response = st.write_stream(rag_chain.stream(prompt))
                
                # NEW: Show Source Citations (UI)
                docs = retriever.invoke(prompt)
                sources = list(set([f"Page {doc.metadata.get('page', '?') + 1}" for doc in docs if 'page' in doc.metadata]))
                if sources:
                    st.caption(f"📖 Sources: {', '.join(sources)}")
        
        st.session_state.messages.append({"role": "assistant", "content": response})

# NEW: Footer (UI)
st.markdown("---")
st.caption("Built with Streamlit, LangChain, ChromaDB, and Groq.")