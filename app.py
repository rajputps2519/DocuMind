import os
os.environ.setdefault("USER_AGENT", "DocuMind/1.0")

import streamlit as st
from rag_pipeline import build_vector_store, get_answer
from document_loader import load_pdf, load_url

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocuMind",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 1.5rem 0 0.5rem 0;
    }
    .main-header h1 {
        font-size: 2.8rem;
        font-weight: 700;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .main-header p {
        color: #888;
        font-size: 1rem;
        margin-top: -0.5rem;
    }
    .status-ready {
        background: linear-gradient(135deg, #11998e, #38ef7d);
        color: white;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        font-weight: 600;
        text-align: center;
        margin-top: 0.5rem;
    }
    .status-empty {
        background: #2a2a2a;
        color: #aaa;
        padding: 0.6rem 1rem;
        border-radius: 10px;
        text-align: center;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
    .welcome-box {
        background: linear-gradient(135deg, #1a1a2e, #16213e);
        border: 1px solid #333;
        border-radius: 16px;
        padding: 2.5rem;
        text-align: center;
        margin: 4rem auto;
        max-width: 600px;
    }
    .welcome-box h2 { color: #fff; margin-bottom: 0.5rem; }
    .welcome-box p  { color: #aaa; font-size: 0.95rem; }
    .stat-box {
        background: #1e1e1e;
        border: 1px solid #333;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        margin-top: 0.5rem;
    }
    .stat-box .num  { font-size: 1.4rem; font-weight: 700; color: #667eea; }
    .stat-box .lbl  { font-size: 0.75rem; color: #888; }
    div[data-testid="stSidebar"] { background: #111; }
</style>
""", unsafe_allow_html=True)

# ── Session state init ────────────────────────────────────────────────────────
if "db_ready"      not in st.session_state: st.session_state.db_ready      = False
if "messages"      not in st.session_state: st.session_state.messages      = []
if "chunk_count"   not in st.session_state: st.session_state.chunk_count   = 0
if "doc_names"     not in st.session_state: st.session_state.doc_names     = []

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🧠 DocuMind")
    st.markdown("---")

    st.markdown("### 📂 Load Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs", type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF files"
    )
    url_input = st.text_input(
        "Or paste a URL",
        placeholder="https://example.com/article",
        help="Paste any webpage URL"
    )

    load_btn = st.button("🚀 Build Knowledge Base", type="primary", use_container_width=True)

    if load_btn:
        all_chunks = []
        doc_names  = []

        for f in (uploaded_files or []):
            if f is not None:
                with st.spinner(f"Loading {f.name}..."):
                    try:
                        chunks = load_pdf(f)
                        all_chunks.extend(chunks)
                        doc_names.append(f.name)
                        st.success(f"✅ {f.name}")
                    except Exception as e:
                        st.error(f"❌ Failed to load {f.name}: {e}")

        if url_input.strip():
            with st.spinner("Loading URL..."):
                try:
                    chunks = load_url(url_input.strip())
                    all_chunks.extend(chunks)
                    doc_names.append(url_input.strip()[:40])
                    st.success("✅ URL loaded")
                except Exception as e:
                    st.error(f"❌ Failed to load URL: {e}")

        if all_chunks:
            with st.spinner("Building FAISS index..."):
                st.session_state.vector_store = build_vector_store(all_chunks)
                st.session_state.db_ready     = True
                st.session_state.chunk_count  = len(all_chunks)
                st.session_state.doc_names    = doc_names
                st.session_state.messages     = []
            st.balloons()
        else:
            st.warning("⚠️ Please upload a PDF or enter a URL first.")

    # Status panel
    st.markdown("---")
    if st.session_state.db_ready:
        st.markdown('<div class="status-ready">✅ Knowledge base ready</div>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="stat-box"><div class="num">{st.session_state.chunk_count}</div><div class="lbl">Chunks</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="stat-box"><div class="num">{len(st.session_state.doc_names)}</div><div class="lbl">Docs</div></div>', unsafe_allow_html=True)
        if st.session_state.doc_names:
            st.markdown("**Loaded documents:**")
            for name in st.session_state.doc_names:
                st.markdown(f"- 📄 `{name}`")
    else:
        st.markdown('<div class="status-empty">No documents loaded yet</div>', unsafe_allow_html=True)

    st.markdown("---")
    if st.session_state.messages:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

# ── Main area ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="main-header">
    <h1>🧠 DocuMind</h1>
    <p>Chat with your documents — powered by Hugging Face + FAISS</p>
</div>
""", unsafe_allow_html=True)

# Welcome screen
if not st.session_state.db_ready and not st.session_state.messages:
    st.markdown("""
    <div class="welcome-box">
        <h2>👋 Welcome to DocuMind</h2>
        <p>Upload a PDF or paste a URL in the sidebar,<br>
        then click <strong>Build Knowledge Base</strong> to get started.</p>
        <br>
        <p>💡 You can upload multiple PDFs at once<br>
        and ask questions across all of them.</p>
    </div>
    """, unsafe_allow_html=True)

# Chat messages
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 View sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Chunk {i}:**")
                    st.caption(src[:400] + "...")
                    st.divider()

# Chat input
if prompt := st.chat_input("Ask a question about your documents..."):
    if not st.session_state.db_ready:
        st.warning("⚠️ Please load documents first using the sidebar.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    answer, sources = get_answer(prompt, st.session_state.vector_store)
                except Exception as e:
                    answer  = f"❌ Error: {str(e)}"
                    sources = []
            st.markdown(answer)
            if sources:
                with st.expander("📚 View sources"):
                    for i, src in enumerate(sources, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.caption(src[:400] + "...")
                        st.divider()

        st.session_state.messages.append({
            "role":    "assistant",
            "content": answer,
            "sources": sources,
        })