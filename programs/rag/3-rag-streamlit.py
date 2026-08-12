from __future__ import annotations

import hashlib
import importlib.util
from pathlib import Path
from typing import Any

import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "chroma_db"
UPLOAD_DIR = BASE_DIR / "uploads"
DEFAULT_TOP_K = 4


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

    if "collection_name" not in st.session_state:
        st.session_state.collection_name = None

    if "source_name" not in st.session_state:
        st.session_state.source_name = None

    if "file_hash" not in st.session_state:
        st.session_state.file_hash = None

    if "top_k" not in st.session_state:
        st.session_state.top_k = DEFAULT_TOP_K


def save_uploaded_pdf(uploaded_file: Any) -> tuple[Path, str]:
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    file_bytes = uploaded_file.getvalue()
    file_hash = hashlib.sha1(file_bytes).hexdigest()
    target_path = UPLOAD_DIR / f"{file_hash}-{uploaded_file.name}"
    target_path.write_bytes(file_bytes)
    return target_path, file_hash


def build_collection_name(file_hash: str) -> str:
    return f"pdf_{file_hash[:12]}"


def ingest_uploaded_pdf(uploaded_file: Any) -> dict[str, Any]:
    embedding_module, chat_module = load_helpers()
    pdf_path, file_hash = save_uploaded_pdf(uploaded_file)
    collection_name = build_collection_name(file_hash)

    pages = embedding_module.read_pdf_pages(pdf_path)
    chunks = embedding_module.build_chunks(
        pages=pages,
        source_name=pdf_path.name,
        chunk_size=900,
        chunk_overlap=150,
    )
    total = embedding_module.store_in_chroma(
        records=chunks,
        db_path=DB_PATH,
        collection_name=collection_name,
        reset_collection=True,
    )
    collection = chat_module.load_collection(DB_PATH, collection_name)

    st.session_state.collection_name = collection_name
    st.session_state.source_name = uploaded_file.name
    st.session_state.file_hash = file_hash
    st.session_state.messages = []

    return {
        "collection": collection,
        "collection_name": collection_name,
        "pages": len(pages),
        "chunks": len(chunks),
        "total": total,
        "source": uploaded_file.name,
    }


def get_active_collection() -> Any | None:
    if not st.session_state.collection_name:
        return None

    _, chat_module = load_helpers()
    return chat_module.load_collection(DB_PATH, st.session_state.collection_name)


def answer_from_uploaded_pdf(question: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    _, chat_module = load_helpers()
    collection = get_active_collection()
    if collection is None:
        raise RuntimeError("No indexed PDF is currently active.")

    client = chat_module.get_cohere_client()
    chunks = chat_module.retrieve_context(collection=collection, question=question, top_k=top_k)
    context = chat_module.build_context(chunks)

    if not context:
        return "I could not find relevant content in the uploaded PDF.", []

    answer = chat_module.answer_question(client=client, question=question, context=context)
    return answer, chunks


def render_sources(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        return

    with st.expander("Retrieved PDF chunks", expanded=False):
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
    st.set_page_config(page_title="RAG PDF Chat", page_icon="📄", layout="centered")
    st.title("RAG Chat With Uploaded PDF")
    st.caption("Upload a PDF, index it into ChromaDB, then ask questions about its content.")

    ensure_state()

    with st.sidebar:
        st.subheader("Upload PDF")
        uploaded_file = st.file_uploader("Choose a PDF file", type=["pdf"])
        top_k = st.slider("Top chunks to retrieve", min_value=1, max_value=8, value=4)
        st.session_state.top_k = top_k

        if uploaded_file is not None:
            uploaded_bytes = uploaded_file.getvalue()
            current_hash = hashlib.sha1(uploaded_bytes).hexdigest()
            already_indexed = current_hash == st.session_state.file_hash

            if st.button("Index PDF", use_container_width=True) or already_indexed:
                if already_indexed and st.session_state.collection_name:
                    st.success(f"PDF already indexed: {st.session_state.source_name}")
                else:
                    with st.spinner("Reading PDF and storing chunks in ChromaDB..."):
                        info = ingest_uploaded_pdf(uploaded_file)
                    st.success(
                        f"Indexed {info['source']} with {info['pages']} pages and {info['chunks']} chunks."
                    )
                    st.write(f"Collection: {info['collection_name']}")
                    st.write(f"Total records: {info['total']}")
        else:
            st.info("Upload a PDF to start chatting.")

        if st.session_state.source_name:
            st.subheader("Active Document")
            st.write(st.session_state.source_name)
            st.write(f"Collection: {st.session_state.collection_name}")

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["text"])
            if message.get("sources"):
                render_sources(message["sources"])

    prompt = st.chat_input("Ask a question about the uploaded PDF...")
    if prompt:
        if not st.session_state.collection_name:
            st.warning("Upload and index a PDF before asking questions.")
            return

        st.session_state.messages.append({"role": "user", "text": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Retrieving context and generating answer..."):
                try:
                    answer, chunks = answer_from_uploaded_pdf(prompt, st.session_state.top_k)
                    st.markdown(answer)
                    render_sources(chunks)
                    st.session_state.messages.append(
                        {"role": "assistant", "text": answer, "sources": chunks}
                    )
                except Exception as error:
                    st.error(f"Assistant error: {error}")


if __name__ == "__main__":
    main()
