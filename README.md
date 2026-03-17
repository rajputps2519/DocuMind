# 🧠 DocuMind — Chat with Your Documents

> A Retrieval-Augmented Generation (RAG) chatbot that lets you upload PDFs and ask questions about them — powered by Hugging Face embeddings, FAISS vector search, and Groq/Ollama LLMs.

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?style=flat-square&logo=streamlit)
![LangChain](https://img.shields.io/badge/LangChain-0.2+-green?style=flat-square)
![FAISS](https://img.shields.io/badge/FAISS-CPU-orange?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

---

## 📌 What is DocuMind?

DocuMind is a **RAG-powered document chatbot** that allows users to:
- Upload one or more PDF documents
- Ask natural language questions about the content
- Get accurate, grounded answers with source citations
- Works completely free using open-source models

---

## 🖼️ Demo

```
1. Upload a PDF (research paper, resume, notes, book)
2. Click "Build Knowledge Base"
3. Ask questions in the chat
4. Get answers with source citations
```

---

## 🏗️ Architecture

```
User Question
      │
      ▼
Query Embedding          PDF / URL
(all-MiniLM-L6-v2)         │
      │                     ▼
      │              Text Chunking
      │            (500 chars, 50 overlap)
      │                     │
      │                     ▼
      │              Chunk Embedding
      │            (all-MiniLM-L6-v2)
      │                     │
      │                     ▼
      └──────► FAISS Vector Store
                     │
                     ▼
              Top-K Similar Chunks
                     │
                     ▼
         LLM (Groq / Ollama / HF)
                     │
                     ▼
            Grounded Answer + Sources
```

---

## 🚀 Quick Start

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/documind.git
cd documind
```

### 2. Create virtual environment
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Set up environment variables

Create a `.env` file in the root folder:

```env
HF_TOKEN=hf_your_huggingface_token
GROQ_API_KEY=gsk_your_groq_api_key
USER_AGENT=DocuMind/1.0
```

- Get your free **HF token** → https://huggingface.co/settings/tokens
- Get your free **Groq key** → https://console.groq.com

### 5. Run the app
```bash
streamlit run app.py
```

Open your browser at `http://localhost:8501` 🎉

---

## 🛠️ Tech Stack

| Component | Technology |
|---|---|
| UI | Streamlit |
| Embeddings | `sentence-transformers/all-MiniLM-L6-v2` (HuggingFace) |
| Vector Store | FAISS (CPU) |
| LLM | Groq (Llama 3.1) / Ollama (local) |
| PDF Loader | PyPDF |
| URL Loader | LangChain WebBaseLoader |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |
| Orchestration | LangChain + Pure Python |

---

## 📁 Project Structure

```
documind/
├── app.py                  ← Streamlit UI
├── rag_pipeline.py         ← Embeddings, FAISS, LLM call
├── document_loader.py      ← PDF & URL ingestion + chunking
├── requirements.txt        ← All dependencies
├── .env                    ← API keys (never commit this!)
├── .gitignore              ← Ignores .env and venv
└── README.md               ← You are here
```

---

## ⚙️ Configuration

You can tweak these settings in `rag_pipeline.py`:

```python
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # embedding model
LLM_MODEL       = "llama-3.1-8b-instant"                     # LLM model (Groq)
TOP_K           = 4                                           # chunks retrieved per query
```

And in `document_loader.py`:

```python
chunk_size    = 500   # characters per chunk
chunk_overlap = 50    # overlap between chunks
```

---

## 🔄 Switching LLM Providers

### Option 1 — Groq (Recommended, free & fast)
```python
# In rag_pipeline.py
LLM_MODEL = "llama-3.1-8b-instant"
# Uses GROQ_API_KEY from .env
```

### Option 2 — Ollama (100% local, no API key)
```bash
# Install Ollama from https://ollama.com
ollama pull llama3.1
```
```python
# In rag_pipeline.py
import ollama
response = ollama.chat(model="llama3.1", messages=[...])
```

### Option 3 — Hugging Face Inference API (free tier)
```python
LLM_MODEL = "google/gemma-2-2b-it"
# Uses HF_TOKEN from .env
```

---

## 📊 How RAG Works

1. **Ingestion** — PDFs are loaded and split into 500-character chunks
2. **Embedding** — Each chunk is converted to a vector using `all-MiniLM-L6-v2`
3. **Storage** — Vectors are stored in a FAISS index in memory
4. **Retrieval** — User question is embedded and top-4 similar chunks are retrieved
5. **Generation** — Retrieved chunks + question are sent to the LLM as context
6. **Answer** — LLM responds using only the provided context (no hallucination)

---

## 🚀 Deployment

### Deploy to Hugging Face Spaces (Free)

1. Create a new Space at https://huggingface.co/spaces
2. Select **Streamlit** as the SDK
3. Upload all project files
4. Add your API keys in **Settings → Repository secrets**
5. Your app will be live at `https://huggingface.co/spaces/yourusername/documind`

---

## 🗺️ Roadmap

- [x] PDF upload and chunking
- [x] FAISS vector store
- [x] Streamlit chat UI
- [x] Source citations
- [ ] Groq LLM integration
- [ ] Ollama local LLM support
- [ ] Conversation memory
- [ ] Multi-language support
- [ ] Deploy to HF Spaces

---

## 🤝 Contributing

Pull requests are welcome! For major changes, please open an issue first.

---

## 📄 License

MIT License — feel free to use this project for learning and portfolio purposes.

---

## 👨‍💻 Author

Built with ❤️ as a Gen AI portfolio project.

> ⭐ If you found this useful, give it a star on GitHub!