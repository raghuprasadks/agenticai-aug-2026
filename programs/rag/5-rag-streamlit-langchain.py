from __future__ import annotations

import hashlib
import importlib.util
import os
from pathlib import Path
from typing import Any

import chromadb
import streamlit as st

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "langchain_chroma_db"
UPLOAD_DIR = BASE_DIR / "uploads"
COLLECTION_NAME = "langchain_multi_pdf_docs"
DEFAULT_TOP_K = 5
MODEL = "command-a-plus-05-2026"
EMBED_MODEL = "embed-english-v3.0"


def get_api_key() -> str:
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if not api_key:
        api_key = "your api key"
    return api_key


def load_module(module_path: Path, module_name: str) -> Any:
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load module from {module_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_embedding_helpers() -> Any:
    return load_module(BASE_DIR / "1-rag-embedding.py", "rag_embedding")


def ensure_langchain_dependencies() -> None:
    try:
        import langchain_chroma  # noqa: F401
        import langchain_cohere  # noqa: F401
        import langchain_core  # noqa: F401
    except ModuleNotFoundError:
        st.error("Missing LangChain RAG dependencies.")
        st.code(
            "./myenv/Scripts/python.exe -m pip install langchain langchain-core langchain-cohere langchain-chroma chromadb cohere streamlit pypdf"
        )
        st.stop()


def get_langchain_objects() -> tuple[Any, Any, Any, Any]:
    from langchain_chroma import Chroma
    from langchain_cohere import ChatCohere, CohereEmbeddings
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate

    return Chroma, ChatCohere, CohereEmbeddings, (ChatPromptTemplate, StrOutputParser)


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


def get_vector_store() -> Any:
    Chroma, _, CohereEmbeddings, _ = get_langchain_objects()
    embeddings = CohereEmbeddings(model=EMBED_MODEL, cohere_api_key=get_api_key())
    return Chroma(
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH),
        embedding_function=embeddings,
    )


def get_collection_count() -> int:
    client = chromadb.PersistentClient(path=str(DB_PATH))
    try:
        collection = client.get_collection(COLLECTION_NAME)
    except Exception:
        return 0
    return collection.count()


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
    embedding_module = load_embedding_helpers()
    vector_store = get_vector_store()

    texts: list[str] = []
    metadatas: list[dict[str, Any]] = []
    ids: list[str] = []
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

        for record in records:
            texts.append(record["document"])
            metadatas.append(record["metadata"])
            ids.append(record["id"])

        page_count += len(pages)
        added_files.append(uploaded_file.name)
        st.session_state.indexed_files[file_hash] = uploaded_file.name

    if texts:
        vector_store.add_texts(texts=texts, metadatas=metadatas, ids=ids)

    return {
        "added_files": added_files,
        "skipped_files": skipped_files,
        "pages": page_count,
        "chunks": len(texts),
        "total": get_collection_count(),
    }


def retrieve_chunks(question: str, top_k: int) -> list[dict[str, Any]]:
    vector_store = get_vector_store()
    results = vector_store.similarity_search_with_score(question, k=top_k)

    chunks: list[dict[str, Any]] = []
    for doc, score in results:
        chunks.append(
            {
                "document": doc.page_content,
                "metadata": doc.metadata or {},
                "distance": score,
            }
        )

    return chunks


def build_context(chunks: list[dict[str, Any]]) -> str:
    if not chunks:
        return ""

    lines = []
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        chunk_index = metadata.get("chunk_index", "?")

        lines.append(f"Source {index}: {source}, page {page}, chunk {chunk_index}")
        lines.append(chunk["document"])
        lines.append("")

    return "\n".join(lines).strip()


def answer_from_documents(question: str, top_k: int) -> tuple[str, list[dict[str, Any]]]:
    _, ChatCohere, _, prompt_objects = get_langchain_objects()
    ChatPromptTemplate, StrOutputParser = prompt_objects

    chunks = retrieve_chunks(question=question, top_k=top_k)
    context = build_context(chunks)
    if not context:
        return "I could not find relevant content in the uploaded documents.", []

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a RAG assistant. Answer only from the provided PDF context. "
                "If the answer is not present in the context, say that the information is not available in the indexed documents. "
                "Mention the relevant source and page when possible.",
            ),
            (
                "human",
                "Question:\n{question}\n\nRetrieved context:\n{context}",
            ),
        ]
    )

    llm = ChatCohere(
        model=MODEL,
        temperature=0,
        cohere_api_key=get_api_key(),
    )
    chain = prompt | llm | StrOutputParser()
    answer = chain.invoke({"question": question, "context": context})
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
                f"{index}. source={source}, page={page}, chunk={chunk_index}, score={distance}"
            )
            st.caption(chunk["document"][:400])


def main() -> None:
    st.set_page_config(page_title="LangChain Multi-Document RAG", page_icon="📚", layout="centered")
    st.title("LangChain RAG Chat Across Multiple PDFs")
    st.caption(
        "Upload several PDFs, index them with LangChain + Chroma, then ask questions across all uploaded documents."
    )

    ensure_langchain_dependencies()
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
                with st.spinner("Reading PDFs and indexing with LangChain Chroma..."):
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
            with st.spinner("Retrieving context with LangChain and generating answer..."):
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
