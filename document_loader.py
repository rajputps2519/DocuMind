"""
document_loader.py
Handles loading and splitting text from PDFs and URLs into chunks.
"""
import os
os.environ["USER_AGENT"] = "DocuMind/1.0"

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
import tempfile


def _split(docs):
    """Split documents into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
        separators=["\n\n", "\n", ".", " ", ""],
    )
    return splitter.split_documents(docs)


def load_pdf(uploaded_file) -> list:
    """
    Load a Streamlit UploadedFile (PDF) and return a list of text chunks.
    PyPDFLoader needs a real file path, so we write to a temp file first.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    try:
        loader = PyPDFLoader(tmp_path, extract_images=False)
        docs = loader.load()

        # Fix encoding issues
        for doc in docs:
            doc.page_content = doc.page_content.encode(
                "utf-8", errors="ignore"
            ).decode("utf-8")
            doc.page_content = " ".join(doc.page_content.split())

    finally:
        os.unlink(tmp_path)

    return _split(docs)


def load_url(url: str) -> list:
    """
    Load a webpage and return a list of text chunks.
    Requires: pip install beautifulsoup4
    """
    loader = WebBaseLoader(url)
    docs = loader.load()
    return _split(docs)