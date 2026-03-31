"""
app.py  –  GenAI Banking Policy & Compliance Copilot v3
Run with:  streamlit run app.py
"""
from __future__ import annotations
import logging, sys, time, traceback
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    APP_TITLE, CHUNK_OVERLAP, CHUNK_SIZE,
    MAX_NEW_TOKENS, PHI2_MODEL_NAME, TEMPERATURE, TOP_K_RESULTS,
    UPLOAD_DIR,
)

logging.basicConfig(level=logging.INFO)

st.set_page_config(page_title=APP_TITLE, page_icon="🏦",
                   layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.main-header {
    background: linear-gradient(135deg, #1a2744 0%, #0d3b6e 50%, #1565c0 100%);
    padding: 1.5rem 2rem; border-radius: 12px; margin-bottom: 1rem; color: white;
}
.main-header h1 { margin: 0; font-size: 1.8rem; }
.main-header p  { margin: 0.3rem 0 0; opacity: 0.8; font-size: 0.9rem; }
.badge-green  { color: #2e7d32; font-weight: 600; }
.badge-orange { color: #e65100; font-weight: 600; }
.ocr-on  { background:#e8f5e9; border-left:4px solid #2e7d32; padding:6px 10px; border-radius:4px; font-size:0.85rem; margin-bottom:6px; }
.ocr-off { background:#fff8e1; border-left:4px solid #f9a825; padding:6px 10px; border-radius:4px; font-size:0.85rem; margin-bottom:6px; }
#MainMenu { visibility: hidden; }
footer    { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Session state ─────────────────────────────────────────────────────────────
for k, v in {"messages": [], "indexed_files": []}.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ── Pipeline ──────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_pipeline():
    from src.rag_pipeline import ComplianceRAGPipeline
    return ComplianceRAGPipeline(
        top_k=TOP_K_RESULTS, model_name=PHI2_MODEL_NAME,
        max_new_tokens=MAX_NEW_TOKENS, temperature=TEMPERATURE,
    )

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🏦 Compliance Copilot")
    st.markdown("*v3 — FAISS + EasyOCR*")
    st.markdown("---")

    from src.document_processor import ocr_status
    ocr = ocr_status()
    css = "ocr-on" if ocr["available"] else "ocr-off"
    st.markdown(f'<div class="{css}">{ocr["message"]}</div>', unsafe_allow_html=True)

    if not ocr["available"]:
        st.markdown("**To enable:** `pip install easyocr`  then restart the app.")

    st.markdown("---")
    st.markdown("### 📄 Upload Policy Documents")

    uploaded_files = st.file_uploader(
        "Upload PDF documents", type=["pdf"], accept_multiple_files=True,
    )

    if uploaded_files:
        pipeline = get_pipeline()
        for uf in uploaded_files:
            if uf.name not in st.session_state["indexed_files"]:
                dest = UPLOAD_DIR / uf.name
                with open(dest, "wb") as f:
                    f.write(uf.getbuffer())
                with st.spinner(f"Indexing {uf.name}…"):
                    try:
                        from src.document_processor import load_and_chunk_pdf
                        chunks = load_and_chunk_pdf(dest, CHUNK_SIZE, CHUNK_OVERLAP)
                        text_chunks = sum(1 for c in chunks if c.chunk_type == "text")
                        img_chunks  = sum(1 for c in chunks if c.chunk_type == "image_ocr")
                        pipeline.vectorstore.add_documents(chunks)
                        st.session_state["indexed_files"].append(uf.name)
                        msg = f"✅ {uf.name} — {len(chunks)} chunks"
                        if img_chunks > 0:
                            msg += f"\n   ({text_chunks} text + {img_chunks} image OCR)"
                        st.success(msg)
                    except Exception as exc:
                        st.error(f"❌ {uf.name}:\n{exc}\n\n{traceback.format_exc()}")

    st.markdown("---")
    st.markdown("### 📊 Index Status")
    try:
        pipeline  = get_pipeline()
        doc_count = pipeline.vectorstore.document_count()
        if doc_count > 0:
            st.markdown(
                f'<span class="badge-green">● FAISS — {doc_count} vectors</span>',
                unsafe_allow_html=True,
            )
            for fname in st.session_state["indexed_files"]:
                st.markdown(f"  • {fname}")
        else:
            st.markdown(
                '<span class="badge-orange">● No documents yet</span>',
                unsafe_allow_html=True,
            )
    except Exception as exc:
        st.error(str(exc))

    st.markdown("---")
    with st.expander("⚙️ Settings"):
        st.markdown(f"**Phi-2:** `{PHI2_MODEL_NAME}`")
        st.markdown(f"**Vector DB:** FAISS (TF-IDF + SVD, offline)")
        st.markdown(f"**Image OCR:** {'EasyOCR ✅' if ocr['available'] else 'Disabled ⚠️'}")
        st.markdown(f"**Chunk size:** {CHUNK_SIZE} | **Top-K:** {TOP_K_RESULTS}")
        st.markdown(f"**Max tokens:** {MAX_NEW_TOKENS}")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("🗑️ Clear Chat"):
            st.session_state["messages"] = []
            st.rerun()
    with col2:
        if st.button("🔄 Reset Index"):
            try:
                get_pipeline().vectorstore.reset()
                st.session_state["indexed_files"] = []
                st.success("Index cleared.")
                st.rerun()
            except Exception as exc:
                st.error(str(exc))

# ── Main panel ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="main-header">
    <h1>🏦 {APP_TITLE}</h1>
    <p>Phi-2 (local) · FAISS vector search · PDF text + EasyOCR image extraction</p>
</div>
""", unsafe_allow_html=True)

# ── Render existing messages ──────────────────────────────────────────────────
# Sources stored in session state are plain dicts — use s["key"] syntax here
for msg in st.session_state["messages"]:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg.get("error"):
            st.warning(f"LLM Error: {msg['error']}")
        if msg.get("sources"):
            with st.expander(f"📚 Sources ({len(msg['sources'])} cited)"):
                for s in msg["sources"]:
                    icon  = "🖼️" if s.get("chunk_type") == "image_ocr" else "📄"
                    label = " *(from image)*" if s.get("chunk_type") == "image_ocr" else ""
                    st.markdown(
                        f"{icon} **{s['document_name']}** — Page {s['page_number']}{label}"
                    )
                    st.caption(f"…{s['excerpt']}…")

# ── Empty state ───────────────────────────────────────────────────────────────
if not st.session_state["messages"]:
    st.info("👆 Upload a policy PDF in the sidebar, then type your question below and press Enter.")

# ── Chat input ────────────────────────────────────────────────────────────────
question = st.chat_input("Ask a compliance question…")

if question and question.strip():
    question = question.strip()

    st.session_state["messages"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching FAISS index and generating answer…"):
            try:
                t0       = time.time()
                pipeline = get_pipeline()
                response = pipeline.query(question)
                elapsed  = time.time() - t0

                answer = response.answer.strip() if response.answer else ""
                if not answer:
                    answer = "The model returned an empty response. Please rephrase your question."
                else:
                    answer += f"\n\n*⏱ {elapsed:.1f}s*"

                st.write(answer)

                # Build sources as plain dicts for safe session state storage
                sources_data = []
                if response.sources:
                    with st.expander(f"📚 Sources ({len(response.sources)} cited)"):
                        for s in response.sources:
                            # s is a CitedSource dataclass — access via s.field (not s["field"])
                            # chunk_type lives in the rag_pipeline metadata dict
                            chunk_type = "text"
                            if hasattr(s, "metadata") and isinstance(s.metadata, dict):
                                chunk_type = s.metadata.get("chunk_type", "text")

                            icon  = "🖼️" if chunk_type == "image_ocr" else "📄"
                            label = " *(from image)*" if chunk_type == "image_ocr" else ""

                            # No match % — just document name and page
                            st.markdown(
                                f"{icon} **{s.document_name}** — Page {s.page_number}{label}"
                            )
                            st.caption(f"…{s.excerpt}…")

                            # Store as plain dict (no CitedSource objects in session state)
                            sources_data.append({
                                "document_name": s.document_name,
                                "page_number":   s.page_number,
                                "excerpt":       s.excerpt,
                                "chunk_type":    chunk_type,
                            })

                if response.error:
                    st.warning(f"LLM Error: {response.error}")

                st.session_state["messages"].append({
                    "role":    "assistant",
                    "content": answer,
                    "error":   response.error,
                    "sources": sources_data,
                })

            except Exception as exc:
                err_msg = traceback.format_exc()
                st.error(f"{type(exc).__name__}: {exc}")
                st.session_state["messages"].append({
                    "role":    "assistant",
                    "content": f"⚠️ {type(exc).__name__}: {exc}",
                    "error":   err_msg,
                    "sources": [],
                })
