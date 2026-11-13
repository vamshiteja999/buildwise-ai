"""BuildWise AI: chat, admin upload and evaluation pages (run with: streamlit run app.py)."""
import html
import re
import time
from pathlib import Path

import streamlit as st
from qdrant_client import models

from config import (
    ADMIN_PASSWORD, ANSWER_MODEL, COLLECTION, DATA_DIR, EMBED_MODEL,
    EXTRACT_V2, PROMPT_V2, QUERY_UNDERSTANDING, SUPPORTED_EXTENSIONS, VERIFY,
)
from buildwise.core import LAST_PLAN, ask, get_client, health_check, ingest_all

st.set_page_config(page_title="BuildWise AI", page_icon="🏗️", layout="wide")
st.markdown(
    """
    <style>
    #MainMenu, footer {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 62rem;}
    .bw-hero h1 {margin-bottom: 0.2rem;}
    .bw-hero p {color: #5b6670; font-size: 1.05rem; margin-top: 0;}
    div[data-testid="stChatMessage"] {border-radius: 12px;}
    div[data-testid="stColumn"] button {min-height: 2.6rem; padding: 0.25rem 0.7rem; text-align: left;
        justify-content: flex-start; font-size: 0.85rem; font-weight: 400; color: #4a5560;
        background: transparent; border: 1px solid #E3DED6;}
    div[data-testid="stColumn"] button:hover {border-color: #D9631E; color: #D9631E; background: #FBF6F1;}
    div[data-testid="stChatInput"] {border: 2px solid #D9631E; border-radius: 16px;
        box-shadow: 0 6px 22px rgba(217, 99, 30, 0.22); background: #FFFFFF;}
    div[data-testid="stChatInput"] textarea {font-size: 1.05rem;}
    .bw-src {border-left: 3px solid #D9631E; padding: 0.3rem 0.7rem; margin: 0.35rem 0;
             background: #F6F3EE; border-radius: 0 8px 8px 0; font-size: 0.9rem; color: #1F2933;}
    .bw-src b {color: #1F2933;}
    .bw-src span {color: #5b6670;}
    </style>
    """,
    unsafe_allow_html=True,
)

EXAMPLES = [  # (what it shows off, question): an even number so the grid has no odd one out
    ("📏", "What top rail height is required for a guardrail system?"),
    ("📋", "Which sections and attachments does a Boca Raton permit for a metal roof require?"),
    ("📄", "What is OSHA3146 about?"),
    ("📚", "What documents do you have?"),
    ("♿", "What are the seat height, seat width and lifting capacity requirements for pool lifts?"),
    ("🚪", "What is the fire rating of door 101?"),
]


# ---------- helpers ----------
def indexed_documents():
    """Return {source file: {"type": ..., "chunks": n}} from the vector store."""
    client = get_client()
    docs, offset = {}, None
    while True:
        points, offset = client.scroll(COLLECTION, limit=256, offset=offset,
                                       with_payload=True, with_vectors=False)
        for point in points:
            payload = point.payload or {}
            entry = docs.setdefault(payload.get("source", "unknown"),
                                    {"type": payload.get("document_type", ""), "chunks": 0})
            entry["chunks"] += 1
        if offset is None:
            break
    return docs


def delete_document(name):
    """Remove a document's chunks from the index and its file from the data folder."""
    get_client().delete(
        collection_name=COLLECTION,
        points_selector=models.FilterSelector(filter=models.Filter(must=[
            models.FieldCondition(key="source", match=models.MatchValue(value=name))
        ])),
    )
    for path in DATA_DIR.rglob(name):
        if path.is_file():
            path.unlink()


def safe_markdown(text):
    """Dollar signs would otherwise be rendered as math by Streamlit."""
    return text.replace("$", "\\$")


SOURCE_RE = re.compile(r"^\[([\d.]+)\]\s*Source:\s*(.+?),\s*page\s*([^,]+),\s*section\s*(.*)$")


def show_sources(sources):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for src in sources:
            m = SOURCE_RE.match(src.strip())
            if m:
                num, name, page, section = m.groups()
                where = f"page {page}" + ("" if section.strip() in ("", "not specified") else f" · {section.strip()}")
                st.markdown(f'<div class="bw-src"><b>[{num}] {html.escape(name)}</b><br><span>{html.escape(where)}</span></div>',
                            unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="bw-src">{html.escape(src)}</div>', unsafe_allow_html=True)


# ---------- sidebar ----------
st.sidebar.title("🏗️ BuildWise AI")
st.sidebar.caption("Ask questions about your construction documents and get answers with sources.")
PAGE_LABELS = {"Chat": "💬  Ask", "Admin": "📁  Documents", "Evaluation": "📊  Results"}
page = st.sidebar.radio("Page", list(PAGE_LABELS), format_func=PAGE_LABELS.get, label_visibility="collapsed")

try:
    health_check()
except Exception as exc:  # Ollama not running, models missing, etc.
    st.error(f"BuildWise cannot start: {exc}")
    st.stop()

try:
    docs = indexed_documents()
except Exception as exc:
    st.error(f"Could not read the document index: {exc}\n\nClose any other BuildWise "
             "process (ingest, eval, chat) that is using the same database, then reload.")
    st.stop()

st.sidebar.divider()
st.sidebar.markdown(f"**{len(docs)}** documents indexed")
st.sidebar.caption("🔒 Runs on this computer. Your documents never leave it.")

# ---------- chat ----------
if page == "Chat":
    st.markdown('<div class="bw-hero"><h1>Ask your construction documents</h1>'
                '<p>Answers come only from your files, with the source file and page for every answer.</p></div>',
                unsafe_allow_html=True)
    if not docs:
        st.info("No documents are indexed yet. Open **Admin**, upload files, and click "
                "*Save and index*.")
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages and "queued" not in st.session_state and docs:
        st.caption("Type a question in the box at the bottom, or try one of these:")
        cols = st.columns(2)
        for i, (icon, example) in enumerate(EXAMPLES):
            if cols[i % 2].button(f"{icon}  {example}", key=f"ex{i}", use_container_width=True):
                st.session_state.queued = example
                st.rerun()
        st.caption("I can also summarize a document, answer several questions in one message, and follow up on "
                   "my last answer, for example: \"what about midrails?\"")
        with st.expander(f"Documents I can search ({len(docs)})"):
            for name in sorted(docs):
                st.markdown(f"- {safe_markdown(name)}")
    if st.session_state.messages and st.sidebar.button("Start a new conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(safe_markdown(message["content"]))
            if message["role"] == "assistant":
                show_sources(message.get("sources"))
                if message.get("searched"):
                    st.caption("Searched for: " + " | ".join(message["searched"]))
                if message.get("seconds") is not None:
                    st.caption(f"Answered in {message['seconds']:.1f} s")

    typed = st.chat_input("Type your question here: rules, permits, drawings, specifications...")
    question = typed or st.session_state.pop("queued", None)
    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(safe_markdown(question))
        with st.chat_message("assistant"):
            started = time.perf_counter()
            history = [(m["role"], m["content"]) for m in st.session_state.messages[:-1]]
            try:
                with st.spinner("Searching the documents..."):
                    answer, sources = ask(question, history)
            except Exception as exc:
                answer, sources = f"Sorry, something went wrong: {exc}", []
            seconds = time.perf_counter() - started
            searched = []
            if LAST_PLAN.get("how") in ("model", "rules", "fallback") and LAST_PLAN.get("queries") != [question]:
                searched = list(LAST_PLAN.get("queries") or [])
            st.markdown(safe_markdown(answer))
            show_sources(sources)
            if searched:
                st.caption("Searched for: " + " | ".join(searched))
            st.caption(f"Answered in {seconds:.1f} s")
        st.session_state.messages.append(
            {"role": "assistant", "content": answer, "sources": sources, "seconds": seconds,
             "searched": searched})

# ---------- admin ----------
elif page == "Admin":
    st.header("Admin: documents")
    if not ADMIN_PASSWORD:
        st.warning("Admin is locked because no password is set. Add this line to your `.env` "
                   "file and restart the app:\n\n`ADMIN_PASSWORD=choose-a-password`")
        st.stop()
    if not st.session_state.get("admin_ok"):
        entered = st.text_input("Admin password", type="password")
        if st.button("Unlock"):
            if entered == ADMIN_PASSWORD:
                st.session_state.admin_ok = True
                st.rerun()
            else:
                st.error("Wrong password.")
        st.stop()

    with st.expander("System details"):
        st.markdown(
            f"**Answer model:** `{ANSWER_MODEL}`  \n"
            f"**Embeddings:** `{EMBED_MODEL}`  \n"
            f"**Collection:** `{COLLECTION}`  \n"
            f"**Refusal gate:** {'on' if PROMPT_V2 else 'off'} · "
            f"**Answer check:** {'on' if VERIFY else 'off'} · "
            f"**Follow-ups and small talk:** {'on' if QUERY_UNDERSTANDING else 'off'} · "
            f"**Table-aware extraction:** {'on' if EXTRACT_V2 else 'off'}")

    st.subheader("Upload and index")
    types = sorted(ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS)
    uploads = st.file_uploader("PDF, DOCX, CSV, TXT or Markdown files", type=types,
                               accept_multiple_files=True)
    if st.button("Save and index", type="primary", disabled=not uploads):
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        for upload in uploads:
            name = Path(upload.name).name  # never trust a path from the browser
            if name in docs:  # same file name: replace the old version
                delete_document(name)
            (DATA_DIR / name).write_bytes(upload.getbuffer())
        with st.spinner("Reading, chunking and embedding. This can take a few minutes..."):
            try:
                summary = ingest_all()
                st.success(f"Indexed {summary['documents_indexed']} new document(s), "
                           f"{summary['chunks_indexed']} chunks; "
                           f"{summary['documents_skipped']} already indexed.")
                if summary["errors"]:
                    st.error(summary["errors"])
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
        docs = indexed_documents()

    pending = sorted(p.name for p in DATA_DIR.rglob("*")
                     if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS) if DATA_DIR.exists() else []
    if st.button("Index files already in the folder (no upload needed)"):
        with st.spinner("Reading, chunking and embedding. This can take a few minutes..."):
            try:
                summary = ingest_all()
                st.success(f"Indexed {summary['documents_indexed']} new document(s), "
                           f"{summary['chunks_indexed']} chunks; {summary['documents_skipped']} already indexed.")
                if summary["errors"]:
                    st.error(summary["errors"])
            except Exception as exc:
                st.error(f"Indexing failed: {exc}")
        docs = indexed_documents()
    not_indexed = [n for n in pending if n not in docs]
    if not_indexed:
        st.warning("Saved but not indexed yet: " + ", ".join(not_indexed))

    st.subheader("Indexed documents")
    if docs:
        st.dataframe(
            [{"Document": name, "Type": info["type"], "Chunks": info["chunks"]}
             for name, info in sorted(docs.items())],
            use_container_width=True, hide_index=True)
        with st.expander("Remove a document"):
            target = st.selectbox("Document", sorted(docs))
            confirm = st.checkbox("Yes, remove it from the index and delete the file")
            if st.button("Remove", disabled=not confirm):
                delete_document(target)
                st.success(f"Removed {target}.")
                st.rerun()
    else:
        st.info("Nothing indexed yet.")
    if st.button("Lock admin"):
        st.session_state.admin_ok = False
        st.rerun()

# ---------- evaluation ----------
else:
    st.header("How well does it work?")
    st.caption("Measured on 71 hand-checked questions (31 easy, 35 multi-part, 5 that must be "
               "refused). Full history of every change and run: `EXPERIMENT_LOG.md`.")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Right document in top 5", "100%")
    c2.metric("Answers with citations", "97%")
    c3.metric("Needed text retrieved", "92.7%")
    c4.metric("Median time per answer", "15.7 s")
    c5, c6, c7, c8 = st.columns(4)
    c5.metric("False refusals", "3.0%")
    c6.metric("Correct refusals", "4 of 5")
    c7.metric("Fully correct answers", "53 of 66")
    c8.metric("Ranking quality (MRR)", "0.852")
    st.subheader("What improved the system")
    st.markdown(
        "- Clause-aware chunking for regulations and a context prefix on every chunk.\n"
        "- Qwen3 thinking switched off: about 10 s per answer instead of 25 to 40 s.\n"
        "- Answer-first prompt with a citation on every sentence and an answerable/not-answerable gate.\n"
        "- Table- and drawing-aware extraction: table rows stay together; drawing sheets split into "
        "keynote legend, title block and annotations.\n"
        "- BM25 keyword search plus repeated header/footer removal: right document in the top 5 for every question.\n"
        "- Answer verification: a second check refuses answers the cited text does not support."
    )
    st.subheader("Known limits")
    st.markdown(
        "- A small local model sometimes answers from a related rule instead of refusing "
        "(for example scaffold guardrails from Subpart L).\n"
        "- Scanned pages and drawing annotations are read less reliably than text.\n"
        "- Numbers shown are for the 16 GB setup (qwen3:8b). An 8 GB laptop uses a smaller model and answers 46 of 66 fully correctly."
    )
