from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any

import chromadb
import cohere
from chromadb.utils import embedding_functions

MODEL = "command-a-plus-05-2026"
DEFAULT_DB_PATH = "rag/chroma_db"
DEFAULT_COLLECTION = "pdf_docs"
DEFAULT_TOP_K = 4


def get_cohere_client() -> cohere.ClientV2:
    api_key = os.getenv("COHERE_API_KEY", "").strip()
    if not api_key:
        api_key = "your api key"
    return cohere.ClientV2(api_key=api_key)


def get_answer_text(response: Any) -> str:
    content_items = getattr(response.message, "content", []) or []
    text_parts = []

    for item in content_items:
        if getattr(item, "type", None) == "text" and hasattr(item, "text"):
            text_parts.append(item.text)

    if text_parts:
        return "\n".join(text_parts).strip()

    return "No answer returned by the model."


def load_collection(db_path: Path, collection_name: str):
    client = chromadb.PersistentClient(path=str(db_path))
    embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return client.get_collection(name=collection_name, embedding_function=embedding_fn)


def retrieve_context(collection: Any, question: str, top_k: int) -> list[dict[str, Any]]:
    result = collection.query(query_texts=[question], n_results=top_k)

    documents = result.get("documents", [[]])
    metadatas = result.get("metadatas", [[]])
    distances = result.get("distances", [[]])

    rows: list[dict[str, Any]] = []
    for document, metadata, distance in zip(documents[0], metadatas[0], distances[0]):
        rows.append(
            {
                "document": document,
                "metadata": metadata or {},
                "distance": distance,
            }
        )

    return rows


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


def answer_question(client: cohere.ClientV2, question: str, context: str) -> str:
    messages = [
        {
            "role": "system",
            "content": (
                "You are a RAG assistant. Answer only from the provided PDF context. "
                "If the answer is not present in the context, say that the information "
                "is not available in the indexed PDF. Keep answers clear and concise."
            ),
        },
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": (
                        f"Question:\n{question}\n\n"
                        f"PDF context:\n{context}\n\n"
                        "Answer the question and mention the relevant source/page when possible."
                    ),
                }
            ],
        },
    ]

    response = client.chat(
        model=MODEL,
        temperature=0,
        messages=messages,
    )
    return get_answer_text(response)


def print_sources(chunks: list[dict[str, Any]]) -> None:
    if not chunks:
        print("No matching chunks found.")
        return

    print("Sources used:")
    for index, chunk in enumerate(chunks, start=1):
        metadata = chunk["metadata"]
        source = metadata.get("source", "unknown")
        page = metadata.get("page", "?")
        chunk_index = metadata.get("chunk_index", "?")
        distance = chunk.get("distance")
        print(
            f"{index}. source={source}, page={page}, chunk={chunk_index}, distance={distance}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Command-line RAG chat over PDF content stored in ChromaDB."
    )
    parser.add_argument(
        "--db-path",
        default=DEFAULT_DB_PATH,
        help="Directory where ChromaDB persistent data is stored",
    )
    parser.add_argument(
        "--collection",
        default=DEFAULT_COLLECTION,
        help="Chroma collection name",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of retrieved chunks to send to the model",
    )
    parser.add_argument(
        "--show-sources",
        action="store_true",
        help="Print retrieved chunk metadata after every answer",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    db_path = Path(args.db_path)

    if not db_path.exists():
        raise FileNotFoundError(
            f"ChromaDB path not found: {db_path}. Run 1-rag-embedding.py first."
        )

    try:
        collection = load_collection(db_path=db_path, collection_name=args.collection)
    except Exception as error:
        raise RuntimeError(
            "Could not open Chroma collection. Check --db-path and --collection."
        ) from error

    client = get_cohere_client()

    print("RAG PDF Chat")
    print(f"Collection: {args.collection}")
    print(f"DB path: {db_path}")
    print("Ask questions about the indexed PDF content.")
    print("Type 'exit' or 'quit' to stop.\n")

    while True:
        try:
            question = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye!")
            break

        if not question:
            continue

        if question.lower() in {"exit", "quit"}:
            print("Goodbye!")
            break

        chunks = retrieve_context(collection=collection, question=question, top_k=args.top_k)
        context = build_context(chunks)

        if not context:
            print("Assistant: I could not find relevant content in the indexed PDF.\n")
            continue

        try:
            answer = answer_question(client=client, question=question, context=context)
            print(f"Assistant: {answer}\n")
            if args.show_sources:
                print_sources(chunks)
                print()
        except Exception as error:
            print(f"Assistant error: {error}\n")


if __name__ == "__main__":
    main()
