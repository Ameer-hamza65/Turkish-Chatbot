import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
import time

# Load environment variables
load_dotenv()

# Set page configuration
st.set_page_config(
    page_title="Belge ile Sohbet - LLAMA Modeli",
    page_icon="📄",
    layout="centered"
)

# Sidebar for loading PDF documents
with st.sidebar:
    st.title("📄 Belge ile Sohbet - LLAMA 3.1")
    st.header("PDF belgelerinizi buraya yükleyin:")
    
    # File uploader widget
    uploaded_files = st.file_uploader("Bir PDF dosyası seçin", type="pdf", accept_multiple_files=True)

# Initialize chat history
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Load and process uploaded documents
def load_documents(uploaded_files):
    documents = []
    for uploaded_file in uploaded_files:
        try:
            # Use a temporary file to store the uploaded PDF
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
                temp_file.write(uploaded_file.read())
                temp_file_name = temp_file.name  # Save the temp file name for later use

            # Load the PDF from the temporary file
            loader = PyPDFLoader(temp_file_name)  # Use the temporary file's name
            documents.extend(loader.load())
            # Optionally, remove the temporary file after loading
            os.remove(temp_file_name)

        except Exception as e:
            st.error(f"{uploaded_file.name} yüklenirken hata oluştu: {e}")
    
    if not documents:
        st.error("Yüklenen belgeler okunamadı veya yüklenmedi.")
        raise ValueError("Belge bulunamadı.")
    
    return documents

# Create FAISS vector store
def setup_vectorstore(documents):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=512, chunk_overlap=80)
    doc_chunks = text_splitter.split_documents(documents)
    vectorstore = FAISS.from_documents(doc_chunks, embeddings)
    return vectorstore

# Create the conversation chain for document responses
def create_doc_chain(vectorstore):
    # Use ChatGroq with an instruction for professional Turkish responses
    llm_doc = ChatGroq(model_name="llama-3.1-8b-instant", system_message="Tüm yanıtlarınızı profesyonel, resmi Türkçe dilinde verin.")
    retriever = vectorstore.as_retriever()
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm_doc,
        retriever=retriever,
        memory=memory,
        verbose=True
    )
    return chain

# Create the conversation chain for general knowledge responses
def create_general_chain():
    # Use ChatGroq with a Turkish instruction for general responses
    llm_general = ChatGroq(model_name="general-knowledge-model", system_message="Lütfen tüm sorulara profesyonel, resmi Türkçe dilinde yanıt verin.")  
    return llm_general

# Load and process the documents from the uploaded files
if uploaded_files:
    if "vectorstore" not in st.session_state:
        documents = load_documents(uploaded_files)
        st.session_state.vectorstore = setup_vectorstore(documents)

    if "doc_chain" not in st.session_state:
        st.session_state.doc_chain = create_doc_chain(st.session_state.vectorstore)

# Always create the general knowledge chain
if "general_chain" not in st.session_state:
    st.session_state.general_chain = create_general_chain()

# Display chat history
for message in st.session_state.chat_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# User input for the chat
user_input = st.chat_input("LLAMA'ya sor...")

if user_input:
    st.session_state.chat_history.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        start = time.process_time()

        # Attempt to get a response from the document conversation chain
        if "vectorstore" in st.session_state and st.session_state.vectorstore:
            docs = st.session_state.vectorstore.similarity_search(user_input)

            if docs:
                # If documents are found, use the document chain to respond
                response = st.session_state.doc_chain({"question": user_input})  # Correct input format
                assistant_response = response.get('answer', "Belge üzerinden yanıt alınamadı.")
            else:
                # If no relevant documents are found, fallback to the general knowledge model
                fallback_response = st.session_state.general_chain({"question": user_input})  # Correct input format
                assistant_response = fallback_response.get('answer', "Bu konuda bilgi bulamadım.")
        else:
            # If no vector store is available (no documents uploaded), fall back to the general model
            fallback_response = st.session_state.general_chain({"question": user_input})  # Correct input format
            assistant_response = fallback_response.get('answer', "Bu konuda bilgi bulamadım.")

        end = time.process_time()

        st.markdown(assistant_response)
        st.session_state.chat_history.append({"role": "assistant", "content": assistant_response})
        st.write(f"Cevap süresi: {end - start:.2f} saniye")
