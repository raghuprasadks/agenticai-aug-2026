from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploads"
COLLECTION_NAME = "multi_pdf_docs"
DEFAULT_TOP_K = 5


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_helpers() -> tuple[Any, Any]:
    embedding_module = load_module(BASE_DIR / "1-rag-embedding.py", "rag_embedding")
    chat_module = load_module(BASE_DIR / "2-rag-chat.py", "rag_chat")
    return embedding_module, chat_module


def ensure_state() -> None:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if "indexed_files" not in st.session_state:
        st.session_state.indexed_files = {}

    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K


def save_uploaded_pdf(uploaded_file: Any) -> tuple[Path, str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha1(file_bytes).hexdigest()
    target_path = UPLOAD_DIR / f"{file_hash}-{uploaded_file.name}"
    target_path.write_bytes(file_bytes)
    return target_path, file_hash


def get_collection() -> Any:
    _, chat_module = load_helpers()
    return chat_module.load_collection(DB_PATH, COLLECTION_NAME)


def reset_collection() -> None:
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    st.session_state.messages = []
    st.session_state.indexed_files = {}


def ingest_uploaded_pdfs(uploaded_files: list[Any]) -> dict[str, Any]:
    embedding_module, chat_module = load_helpers()

    all_records: list[dict[str, Any]] = []
    added_files: list[str] = []
    skipped_files: list[str] = []
    page_count = 0

    for uploaded_file in uploaded_files:
        pdf_path, file_hash = save_uploaded_pdf(uploaded_file)
        if file_hash in st.session_state.indexed_files:
            skipped_files.append(uploaded_file.name)
            continue

        pages = embedding_module.read_pdf_pages(pdf_path)
        records = embedding_module.build_chunks(
            pages=pages,
            source_name=uploaded_file.name,
            chunk_size=900,
            chunk_overlap=150,
        )
        all_records.extend(records)
        page_count += len(pages)
        added_files.append(uploaded_file.name)
        st.session_state.indexed_files[file_hash] = uploaded_file.name

    total = embedding_module.store_in_chroma(
        records=all_records,
        db_path=DB_PATH,
        collection_name=COLLECTION_NAME,
        reset_collection=False,
    )
    collection = chat_module.load_collection(DB_PATH, COLLECTION_NAME)

    return {
        "collection": collection,
        "added_files": added_files,
        "skipped_files": skipped_files,
        "pages": page_count,
        "chunks": len(all_records),
        "total": total,
    }


def answer_from_documents(question: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    _, chat_module = load_helpers()
    client = chat_module.get_cohere_client()
    collection = get_collection()

    chunks = chat_module.retrieve_context(collection=collection, question=question, top_k=top_k)
    context = chat_module.build_context(chunks)

    if not context:
        return "I could not find relevant content in the uploaded documents.", []

    answer = chat_module.answer_question(client=client, question=question, context=context)
    return answer, chunks


def render_sources(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    with st.expander("Retrieved document chunks", expanded=False):
        for index, chunk in enumerate(chunks, start=1):
            metadata = chunk["metadata"]
            source = metadata.get("source", "unknown")
            page = metadata.get("page", "?")
            chunk_index = metadata.get("chunk_index", "?")
            distance = chunk.get("distance")
            st.markdown(
                f"{index}. source={source}, page={page}, chunk={chunk_index}, distance={distance}"
            )
            st.caption(chunk["document"][:400])


def main() -> None:
    st.set_page_config(page_title="Multi-Document RAG Chat", page_icon="📚", layout="centered")
    st.title("RAG Chat Across Multiple PDFs")
    st.caption("Upload several PDFs, index them into one Chroma collection, then ask questions across all uploaded documents.")

    ensure_state()

    with st.sidebar:
        st.subheader("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=["pdf"],
            accept_multiple_files=True,
        )
        top_k = st.slider("Top chunks to retrieve", min_value=1, max_value=10, value=5)
        st.session_state.top_k = top_k

        index_clicked = st.button("Index Uploaded PDFs", use_container_width=True)
        clear_clicked = st.button("Clear Indexed Documents", use_container_width=True)

        if clear_clicked:
            reset_collection()
            st.success("Cleared indexed documents and chat history.")

        if uploaded_files:
            if index_clicked:
                with st.spinner("Reading PDFs and storing chunks in ChromaDB..."):
                    info = ingest_uploaded_pdfs(list(uploaded_files))

                if info["added_files"]:
                    st.success(
                        f"Indexed {len(info['added_files'])} file(s), {info['pages']} pages, and {info['chunks']} chunks."
                    )
                if info["skipped_files"]:
                    st.info(
                        "Skipped already indexed files: " + ", ".join(info["skipped_files"])
                    )
                st.write(f"Collection: {COLLECTION_NAME}")
                st.write(f"Total records: {info['total']}")
        else:
            st.info("Upload one or more PDFs to start chatting.")

        st.subheader("Indexed Documents")
        if st.session_state.indexed_files:
            for file_name in st.session_state.indexed_files.values():
                st.write(f"- {file_name}")
        else:
            st.write("No documents indexed yet.")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("Ask a question about the uploaded PDFs...")
    if prompt:
        if not st.session_state.indexed_files:
            st.warning("Upload and index at least one PDF before asking questions.")
            return

        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context from uploaded PDFs..."):
                try:
                    answer, chunks = answer_from_documents(prompt, st.session_state.top_k)
                    st.markdown(answer)
                    render_sources(chunks)
                    st.session_state.messages.append(
                        {"role": "assistant", "text": answer, "sources": chunks}
                    )
                except Exception as error:
                    st.error(f"Assistant error: {error}")


if __name__ == "__main__":
    main()
