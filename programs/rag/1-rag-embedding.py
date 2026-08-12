from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any
from uuid import uuid4

import chromadb
from chromadb.utils import embedding_functions
from pypdf import PdfReader


def read_pdf_pages(pdf_path: Path) -> list[dict[str, Any]]:
	reader = PdfReader(str(pdf_path))
	pages: list[dict[str, Any]] = []

	for index, page in enumerate(reader.pages, start=1):
		text = (page.extract_text() or "").strip()
		if text:
			pages.append({"page": index, "text": text})

	return pages


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
	if chunk_size <= 0:
		raise ValueError("chunk_size must be greater than 0")

	if chunk_overlap < 0:
		raise ValueError("chunk_overlap cannot be negative")

	if chunk_overlap >= chunk_size:
		raise ValueError("chunk_overlap must be smaller than chunk_size")

	chunks: list[str] = []
	start = 0
	step = chunk_size - chunk_overlap

	while start < len(text):
		end = start + chunk_size
		chunk = text[start:end].strip()
		if chunk:
			chunks.append(chunk)
		start += step

	return chunks


def build_chunks(
	pages: list[dict[str, Any]],
	source_name: str,
	chunk_size: int,
	chunk_overlap: int,
) -> list[dict[str, Any]]:
	records: list[dict[str, Any]] = []

	for page_info in pages:
		page_number = int(page_info["page"])
		page_text = str(page_info["text"])
		chunks = chunk_text(page_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

		for idx, chunk in enumerate(chunks):
			records.append(
				{
					"id": f"{source_name}-p{page_number}-c{idx}-{uuid4().hex[:8]}",
					"document": chunk,
					"metadata": {
						"source": source_name,
						"page": page_number,
						"chunk_index": idx,
					},
				}
			)

	return records


def store_in_chroma(
	records: list[dict[str, Any]],
	db_path: Path,
	collection_name: str,
	reset_collection: bool,
) -> int:
	db_path.mkdir(parents=True, exist_ok=True)

	client = chromadb.PersistentClient(path=str(db_path))
	embedding_fn = embedding_functions.DefaultEmbeddingFunction()

	if reset_collection:
		try:
			client.delete_collection(collection_name)
		except Exception:
			pass

	collection = client.get_or_create_collection(
		name=collection_name,
		embedding_function=embedding_fn,
	)

	if not records:
		return collection.count()

	ids = [item["id"] for item in records]
	documents = [item["document"] for item in records]
	metadatas = [item["metadata"] for item in records]

	collection.add(ids=ids, documents=documents, metadatas=metadatas)
	return collection.count()


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Read a PDF, chunk the text, and store it in ChromaDB."
	)
	parser.add_argument("--pdf", required=True, help="Path to input PDF file")
	parser.add_argument(
		"--db-path",
		default="rag/chroma_db",
		help="Directory where ChromaDB persistent data is stored",
	)
	parser.add_argument(
		"--collection",
		default="pdf_docs",
		help="Chroma collection name",
	)
	parser.add_argument(
		"--chunk-size",
		type=int,
		default=900,
		help="Approximate characters per chunk",
	)
	parser.add_argument(
		"--chunk-overlap",
		type=int,
		default=150,
		help="Character overlap between chunks",
	)
	parser.add_argument(
		"--reset",
		action="store_true",
		help="Delete and recreate collection before inserting",
	)
	return parser.parse_args()


def main() -> None:
	args = parse_args()

	pdf_path = Path(args.pdf)
	if not pdf_path.exists():
		raise FileNotFoundError(f"PDF not found: {pdf_path}")

	pages = read_pdf_pages(pdf_path)
	chunks = build_chunks(
		pages=pages,
		source_name=pdf_path.name,
		chunk_size=args.chunk_size,
		chunk_overlap=args.chunk_overlap,
	)

	total = store_in_chroma(
		records=chunks,
		db_path=Path(args.db_path),
		collection_name=args.collection,
		reset_collection=args.reset,
	)

	print("RAG ingestion complete")
	print(f"PDF: {pdf_path}")
	print(f"Pages with text: {len(pages)}")
	print(f"Chunks added: {len(chunks)}")
	print(f"Collection: {args.collection}")
	print(f"DB path: {args.db_path}")
	print(f"Total records in collection: {total}")


if __name__ == "__main__":
	main()
