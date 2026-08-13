import os
import re
import uuid
from typing import List, Dict, Any

import chromadb
import cohere
import streamlit as st
from dotenv import load_dotenv
from docx import Document
from pypdf import PdfReader

load_dotenv()

COLLECTION_NAME = "crop_management_docs"
PERSIST_DIR = os.path.join(os.getcwd(), "chroma_db")


@st.cache_resource
def get_chroma_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=PERSIST_DIR)


@st.cache_resource
def get_cohere_client():
    api_key = os.getenv("COHERE_API_KEY")
    if not api_key:
        st.warning("Missing COHERE_API_KEY. Add it to a .env file or your environment.")
        return None

    try:
        return cohere.ClientV2(api_key=api_key)
    except Exception:
        return cohere.Client(api_key=api_key)


def extract_text_from_file(file_obj) -> str:
    filename = (file_obj.name or "").lower()

    try:
        if filename.endswith(".pdf"):
            reader = PdfReader(file_obj)
            pages = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                pages.append(page_text)
            return "\n\n".join(pages)

        if filename.endswith(".docx"):
            doc = Document(file_obj)
            return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

        if filename.endswith((".txt", ".md")):
            return file_obj.read().decode("utf-8", errors="ignore")

    except Exception as exc:
        st.warning(f"Could not read {file_obj.name}: {exc}")

    return ""


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 120) -> List[str]:
    cleaned = re.sub(r"\s+", " ", text).strip()
    if not cleaned:
        return []

    words = cleaned.split()
    chunks: List[str] = []
    start = 0

    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)

        if end == len(words):
            break

        start += max(1, chunk_size - overlap)

    return chunks


def extract_chat_text(response: Any) -> str:
    if hasattr(response, "message"):
        content = getattr(response.message, "content", [])
        if isinstance(content, list) and content:
            candidate = content[0]
            if hasattr(candidate, "text"):
                return candidate.text
            if isinstance(candidate, dict) and "text" in candidate:
                return candidate["text"]
    if hasattr(response, "text"):
        return response.text
    if isinstance(response, dict):
        if "text" in response:
            return response["text"]
        if "message" in response and isinstance(response["message"], dict):
            content = response["message"].get("content", [])
            if isinstance(content, list) and content:
                first = content[0]
                if isinstance(first, dict) and "text" in first:
                    return first["text"]
    return str(response)


def embed_texts(client: Any, texts: List[str]) -> List[List[float]]:
    if not texts or client is None:
        return []

    def unwrap_embeddings(raw: Any) -> List[List[float]]:
        if raw is None:
            return []
        if isinstance(raw, list):
            return raw
        if hasattr(raw, "float_") and raw.float_ is not None:
            return raw.float_
        if hasattr(raw, "model_dump"):
            payload = raw.model_dump(exclude_none=True)
            if isinstance(payload, dict):
                for key in ("float", "float_"):
                    value = payload.get(key)
                    if value is not None:
                        return value
                for value in payload.values():
                    if isinstance(value, list):
                        return value
        if isinstance(raw, dict):
            for key in ("float", "float_"):
                value = raw.get(key)
                if value is not None:
                    return value
            if "embeddings" in raw:
                return unwrap_embeddings(raw["embeddings"])
        if hasattr(raw, "tolist"):
            try:
                return raw.tolist()
            except Exception:
                pass
        try:
            return list(raw)
        except Exception:
            return []

    try:
        response = client.embed(
            model=os.getenv("COHERE_EMBED_MODEL", "embed-v4.0"),
            texts=texts,
            input_type="search_document",
        )
        embeddings = getattr(response, "embeddings", None)
        if embeddings is None and isinstance(response, dict):
            embeddings = response.get("embeddings")
        if embeddings is not None:
            unwrapped = unwrap_embeddings(embeddings)
            if unwrapped:
                return unwrapped
    except TypeError:
        try:
            response = client.embed(
                model=os.getenv("COHERE_EMBED_MODEL", "embed-v4.0"),
                texts=texts,
            )
            embeddings = getattr(response, "embeddings", None)
            if embeddings is None and isinstance(response, dict):
                embeddings = response.get("embeddings")
            if embeddings is not None:
                unwrapped = unwrap_embeddings(embeddings)
                if unwrapped:
                    return unwrapped
        except Exception:
            pass
    except Exception:
        pass

    return []


def generate_answer(client: Any, question: str, relevant_docs: List[Dict[str, str]]) -> str:
    if not relevant_docs:
        return "I could not find relevant information in the uploaded crop-management documents. Please upload relevant documents or ask a more specific question."

    context = "\n\n".join(
        f"Source: {doc.get('source', 'Unknown')}\n{doc.get('text', '')}"
        for doc in relevant_docs
    )

    prompt = (
        "You are an expert agricultural advisor helping farmers in crop management. "
        "Use only the context provided below. If the answer is not present in the context, say that clearly and avoid guessing.\n\n"
        f"Context:\n{context}\n\nUser question: {question}"
    )

    try:
        if hasattr(client, "chat"):
            try:
                response = client.chat(
                    model=os.getenv("COHERE_CHAT_MODEL", "command-r-plus"),
                    messages=[{"role": "user", "content": prompt}],
                )
                return extract_chat_text(response)
            except TypeError:
                response = client.chat(
                    model=os.getenv("COHERE_CHAT_MODEL", "command-r-plus"),
                    message=prompt,
                )
                return extract_chat_text(response)
    except Exception as exc:
        return f"I could not generate a reply because the Cohere chat call failed: {exc}"

    return "The answer could not be generated because there is no working Cohere chat client configured."


def get_collection():
    client = get_chroma_client()
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


def index_uploaded_files(uploaded_files: List[Any], client: Any) -> int:
    if not uploaded_files:
        return 0

    collection = get_collection()
    all_chunks: List[str] = []
    all_metadata: List[Dict[str, Any]] = []
    ids: List[str] = []

    for file in uploaded_files:
        text = extract_text_from_file(file)
        if not text.strip():
            continue

        chunks = chunk_text(text)
        for chunk_index, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            ids.append(f"{uuid.uuid4().hex}")
            all_metadata.append({
                "source": file.name,
                "chunk_index": chunk_index,
            })

    if not all_chunks:
        st.warning("No readable text was found in the uploaded files.")
        return 0

    embeddings = embed_texts(client, all_chunks)
    if not embeddings or len(embeddings) != len(all_chunks):
        st.error("Embedding generation failed. Please check your Cohere API key or model name.")
        return 0

    collection.add(
        ids=ids,
        documents=all_chunks,
        metadatas=all_metadata,
        embeddings=embeddings,
    )
    return len(all_chunks)


def retrieve_relevant_docs(client: Any, question: str, top_k: int = 5) -> List[Dict[str, str]]:
    collection = get_collection()
    embeddings = embed_texts(client, [question])
    if not embeddings:
        return []

    results = collection.query(
        query_embeddings=[embeddings[0]],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]

    relevant = []
    for document, metadata in zip(documents, metadatas):
        relevant.append({
            "text": document,
            "source": metadata.get("source", "Uploaded document") if isinstance(metadata, dict) else "Uploaded document",
        })
    return relevant


st.set_page_config(page_title="Crop Management Assistant", page_icon="🌾", layout="wide")
st.title("🌾 Crop Management RAG Assistant")
st.caption("Upload crop manuals, agronomy notes, or extension documents and ask questions in natural language.")

with st.sidebar:
    st.header("Document library")
    uploaded_files = st.file_uploader(
        "Upload crop management documents",
        type=["pdf", "docx", "txt", "md"],
        accept_multiple_files=True,
    )

    if st.button("Index uploaded documents"):
        client = get_cohere_client()
        if client is None:
            st.stop()
        if uploaded_files:
            chunk_count = index_uploaded_files(uploaded_files, client)
            if chunk_count:
                st.success(f"Successfully indexed {chunk_count} document chunks.")
        else:
            st.warning("Please upload at least one document.")

    if st.button("Clear indexed documents"):
        try:
            chroma_client = get_chroma_client()
            try:
                chroma_client.delete_collection(name=COLLECTION_NAME)
            except Exception:
                collection = get_collection()
                collection.delete(where={})
            st.success("Indexed documents cleared.")
        except Exception as exc:
            st.error(f"Could not clear the collection: {exc}")

st.subheader("Ask a question")
question = st.text_area(
    "Question",
    value="",
    placeholder="Example: What is the best irrigation schedule for paddy during flowering stage?",
    height=120,
)

if st.button("Ask assistant"):
    if not question.strip():
        st.warning("Please enter a question before submitting.")
    else:
        client = get_cohere_client()
        if client is None:
            st.stop()

        relevant_docs = retrieve_relevant_docs(client, question)
        answer = generate_answer(client, question, relevant_docs)

        st.subheader("Answer")
        st.write(answer)

        if relevant_docs:
            st.subheader("Relevant sources")
            for index, item in enumerate(relevant_docs, start=1):
                st.markdown(f"**{index}. {item['source']}**")
                st.write(item["text"][:800])

if "__main__" == __name__:
    pass
