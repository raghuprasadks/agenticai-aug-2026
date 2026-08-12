import hashlib
import os
import uuid
from typing import List

import chromadb
import cohere
import gradio as gr
from dotenv import load_dotenv
from pypdf import PdfReader

try:
    import spaces
except ImportError:
    class _SpacesShim:
        @staticmethod
        def GPU(fn=None, **_kwargs):
            if fn is not None:
                return fn

            def decorator(func):
                return func

            return decorator

    spaces = _SpacesShim()


load_dotenv()

COHERE_API_KEY = os.getenv("COHERE_API_KEY", "")
CHAT_MODEL = os.getenv("COHERE_CHAT_MODEL", "command-a-plus-05-2026")
EMBED_MODEL = os.getenv("COHERE_EMBED_MODEL", "embed-v4.0")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "pdf_rag_docs")

co = cohere.ClientV2(api_key=COHERE_API_KEY) if COHERE_API_KEY else None
chroma_client = chromadb.PersistentClient(path=CHROMA_PATH)
collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)


def extract_text_from_pdf(file_path: str) -> str:
    reader = PdfReader(file_path)
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    return "\n".join(pages)


def chunk_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    chunks: List[str] = []
    start = 0
    text = " ".join(text.split())
    while start < len(text):
        end = min(len(text), start + chunk_size)
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


def embed_texts(texts: List[str], input_type: str) -> List[List[float]]:
    if co is None:
        raise RuntimeError(
            "COHERE_API_KEY was not found. Set it in .env for local runs or in Hugging Face Space Secrets."
        )

    response = co.embed(
        model=EMBED_MODEL,
        input_type=input_type,
        embedding_types=["float"],
        texts=texts,
    )
    return response.embeddings.float


def index_pdf(uploaded_file_path: str) -> str:
    if not uploaded_file_path:
        return "Please upload a PDF first."

    with open(uploaded_file_path, "rb") as f:
        file_bytes = f.read()

    file_hash = hashlib.sha256(file_bytes).hexdigest()

    existing = collection.get(where={"file_hash": file_hash}, include=[])
    if existing.get("ids"):
        return "This PDF is already indexed."

    full_text = extract_text_from_pdf(uploaded_file_path)
    if not full_text.strip():
        return "No readable text found in the PDF."

    chunks = chunk_text(full_text)
    if not chunks:
        return "Could not create text chunks from the PDF."

    embeddings = embed_texts(chunks, input_type="search_document")

    ids = [str(uuid.uuid4()) for _ in chunks]
    metadatas = [
        {
            "file_name": os.path.basename(uploaded_file_path),
            "file_hash": file_hash,
            "chunk_index": i,
        }
        for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        documents=chunks,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return f"Indexed {len(chunks)} chunks from {os.path.basename(uploaded_file_path)}."


def retrieve_context(question: str, k: int = 4) -> List[str]:
    q_embedding = embed_texts([question], input_type="search_query")[0]
    result = collection.query(query_embeddings=[q_embedding], n_results=k)
    documents = result.get("documents", [[]])[0]
    return documents


@spaces.GPU
def answer_question(question: str) -> str:
    if not question or not question.strip():
        return "Please enter a question."

    if co is None:
        return "COHERE_API_KEY was not found. Set it in .env for local runs or in Hugging Face Space Secrets."

    contexts = retrieve_context(question)

    if not contexts:
        return "No relevant content found. Upload and index a PDF first."

    context_block = "\n\n---\n\n".join(contexts)

    system_prompt = (
        "You are a helpful assistant. Answer the user question using only the provided context. "
        "If the answer is not in the context, clearly say you do not know based on the document."
    )

    response = co.chat(
        model=CHAT_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": (
                    f"Context:\n{context_block}\n\n"
                    f"Question: {question}\n\n"
                    "Return a concise answer grounded in the context."
                ),
            },
        ],
    )

    content = response.message.content
    if not content:
        return "The model returned an empty response."

    for block in content:
        text = getattr(block, "text", None)
        if text:
            return text

    return "The model response did not include text output."


def clear_collection() -> str:
    global collection
    try:
        chroma_client.delete_collection(COLLECTION_NAME)
    except Exception:
        pass

    collection = chroma_client.get_or_create_collection(name=COLLECTION_NAME)
    return "Collection cleared."


with gr.Blocks(title="PDF RAG with Cohere + Chroma") as demo:
    gr.Markdown("# PDF RAG: Ask Questions From Your Uploaded PDF")

    if not COHERE_API_KEY:
        gr.Markdown(
            "**Warning:** COHERE_API_KEY was not found. Set it in `.env` for local runs or in Hugging Face Space Secrets."
        )

    with gr.Row():
        uploaded_pdf = gr.File(label="Upload a PDF", file_types=[".pdf"], type="filepath")
        index_btn = gr.Button("Index PDF", variant="primary")
        clear_btn = gr.Button("Clear Chroma Collection", variant="stop")

    index_status = gr.Textbox(label="Index Status", lines=3, interactive=False)

    question = gr.Textbox(label="Question", placeholder="Enter your question about the indexed PDF")
    ask_btn = gr.Button("Get Answer")
    answer = gr.Textbox(label="Answer", lines=8, interactive=False)

    index_btn.click(fn=index_pdf, inputs=[uploaded_pdf], outputs=[index_status])
    clear_btn.click(fn=clear_collection, outputs=[index_status])
    ask_btn.click(fn=answer_question, inputs=[question], outputs=[answer])


if __name__ == "__main__":
    demo.launch()
