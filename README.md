# AI Interview Assistant 

An LLM-powered AI Interview Assistant built using Streamlit, LangChain, and HuggingFace.

## Demo Screenshot

![AI Interview Assistant Demo](app_demo.jpg)

## Working (How it works)

1. User uploads a PDF (resume/notes).
2. LangChain loads and splits the document into chunks.
3. Relevant chunks are retrieved using lightweight keyword-based retrieval.
4. Retrieved context is combined with the user question.
5. The prompt is sent to the HuggingFace LLM endpoint (Mistral-7B).
6. AI generates a context-aware answer and displays it in Streamlit.

## Features
- Upload PDF (resume/notes)
- Ask questions based on document
- Lightweight RAG pipeline
- LLM-powered answers

## Tech Stack
- Python
- Streamlit
- LangChain
- HuggingFace (Mistral-7B)
- PDF processing

## How to Run
```bash
venv\Scripts\activate
streamlit run app.py

