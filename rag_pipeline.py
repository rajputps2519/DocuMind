"""
rag_pipeline.py
Core RAG logic:
  1. Embed document chunks with a free Hugging Face model
  2. Store vectors in FAISS
  3. At query time: embed the question, retrieve top-k chunks,
     pass them as context to a Hugging Face LLM, return the answer.
"""

from dotenv import load_dotenv
load_dotenv()

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

from langchain_core.prompts import PromptTemplate
import os


# ── Model configuration ───────────────────────────────────────────────────────
# Free embedding model (~90 MB, runs locally, no API key needed)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Free HF Inference API model (requires HF_TOKEN env var — free tier is fine)
# You can swap this for any text-generation model on Hugging Face Hub
LLM_MODEL = "google/gemma-2-2b-it"

TOP_K = 4   # number of chunks to retrieve per query


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
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        return ("⚠️ HF_TOKEN not set.", [])

    retriever = vector_store.as_retriever(
        search_type="similarity", search_kwargs={"k": TOP_K}
    )
    docs = retriever.invoke(question)
    context = "\n\n".join([doc.page_content for doc in docs])
    prompt = RAG_PROMPT.format(context=context, question=question)

    import requests
    headers = {
        "Authorization": f"Bearer {hf_token}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": "google/gemma-2-2b-it",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 512,
        "temperature": 0.3
    }

    response = requests.post(
        "https://router.huggingface.co/hf-inference/models/google/gemma-2-2b-it/v1/chat/completions",
        headers=headers,
        json=payload
    )

    result = response.json()

    if "choices" in result:
        answer = result["choices"][0]["message"]["content"]
    elif "error" in result:
        answer = f"❌ Model error: {result['error']}"
    else:
        answer = str(result)

    sources = [doc.page_content for doc in docs]
    return answer, sources