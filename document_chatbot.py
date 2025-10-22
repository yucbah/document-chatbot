# chatbot_app.py
import streamlit as st
from openai import OpenAI
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.chat_models import ChatOpenAI

st.title("📄 Document QA Chatbot")

# Upload document
uploaded_file = st.file_uploader("Upload your document", type=["pdf"])
if uploaded_file:
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Split text into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    chunks = text_splitter.split_text(text)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_texts(chunks, embeddings)

    # Chat interface
    user_question = st.text_input("Ask a question about your document:")
    if user_question:
        llm = ChatOpenAI(model_name="gpt-4o-mini")
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm, knowledge_base.as_retriever()
        )
        response = qa_chain.run(user_question)
        st.write(response)
