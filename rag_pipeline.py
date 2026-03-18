"""
rag_pipeline.py
Core RAG logic:
  1. Embed document chunks with a free Hugging Face model
  2. Store vectors in FAISS
  3. At query time: embed the question, retrieve top-k chunks,
     pass them as context to Groq LLM, return the answer.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.prompts import PromptTemplate
import requests
import os


# ── Model configuration ───────────────────────────────────────────────────────
# Free embedding model (~90 MB, runs locally, no API key needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Groq model — fast, free, reliable
# Alternatives: "mixtral-8x7b-32768", "gemma2-9b-it", "llama-3.3-70b-versatile"
LLM_MODEL = "llama-3.1-8b-instant"

TOP_K = 4   # number of chunks to retrieve per query

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


# ── Prompt template ───────────────────────────────────────────────────────────
RAG_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful assistant. Answer the question using ONLY the
provided context. If the answer is not in the context, say
"I couldn't find that in the documents."

Context:
{context}

Question: {question}

Answer:""",
)


def build_vector_store(chunks: list) -> FAISS:
    """
    Embed all chunks and build a FAISS vector store.
    Returns the store object (kept in Streamlit session_state).
    """
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store


def get_answer(question: str, vector_store: FAISS) -> tuple[str, list[str]]:
    """
    Retrieve relevant chunks and generate an answer using Groq.
    Returns (answer_text, list_of_source_snippets).
    """
    # ── Check API key ─────────────────────────────────────────────────────────
    groq_key = os.getenv("GROQ_API_KEY1")
    if not groq_key:
        return ("⚠️ GROQ_API_KEY not set. Add it to your .env file.", [])

    # ── Retrieve relevant chunks from FAISS ───────────────────────────────────
    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])

    # ── Build prompt ──────────────────────────────────────────────────────────
    prompt = RAG_PROMPT.format(context=context, question=question)

    # ── Call Groq API ─────────────────────────────────────────────────────────
    headers = {
        "Authorization": f"Bearer {groq_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3
    }

    try:
        response = requests.post(
            GROQ_API_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        result = response.json()

        if "choices" in result:
            answer = result["choices"][0]["message"]["content"]
        elif "error" in result:
            answer = f"❌ Groq error: {result['error']['message']}"
        else:
            answer = f"❌ Unexpected response: {str(result)}"

    except requests.exceptions.Timeout:
        answer = "❌ Request timed out. Please try again."
    except requests.exceptions.ConnectionError:
        answer = "❌ Connection error. Check your internet connection."
    except Exception as e:
        answer = f"❌ Unexpected error: {str(e)}"

    sources = [doc.page_content for doc in docs]
    return answer, sources