import streamlit as st
import os
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

load_dotenv()

client = InferenceClient(
    model="mistralai/Mistral-7B-Instruct-v0.2",
    token=os.getenv("HUGGINGFACEHUB_API_TOKEN")
)

st.title("AI Interview Assistant (LangChain RAG) 🚀")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

docs = []

if uploaded_file:
    # save temp file (LangChain needs file path)
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    docs = splitter.split_documents(docs)

    st.success("PDF loaded & split with LangChain!")

user_question = st.text_input("Ask a question based on the PDF:")

if user_question and docs:

    # lightweight retrieval (keyword scoring)
    relevant_chunks = [
        d.page_content for d in docs
        if any(word in d.page_content.lower()
               for word in user_question.lower().split())
    ]

    context = " ".join(relevant_chunks[:5])

    prompt = f"""
    Use this context to answer:

    Context:
    {context}

    Question:
    {user_question}
    """

    response = client.chat_completion(
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000
    )

    st.write(response.choices[0].message.content)