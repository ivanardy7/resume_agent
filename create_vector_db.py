from __future__ import annotations
import re
from pathlib import Path
from typing import Iterable, Iterator, List
import pandas as pd
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from config import (
    BATCH_SIZE,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    MAX_ROWS,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    RESET_COLLECTION,
    validate_required_settings,
)

DATA_PATH = Path("data/Resume.csv")


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        if end >= text_length:
            break

        start = max(0, end - overlap)

    return chunks


def batch_items(items: Iterable[Document], batch_size: int) -> Iterator[List[Document]]:
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []

    if batch:
        yield batch


def build_documents() -> Iterator[Document]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH}. Put Resume.csv inside the data folder."
        )

    usecols = ["ID", "Resume_str", "Category"]
    df = pd.read_csv(DATA_PATH, usecols=usecols)

    if MAX_ROWS and MAX_ROWS > 0:
        df = df.head(MAX_ROWS)

    print(f"Loaded {len(df):,} resume rows from {DATA_PATH}")

    for row_index, row in df.iterrows():
        resume_id = str(row.get("ID", row_index))
        category = str(row.get("Category", "UNKNOWN"))
        resume_text = clean_text(row.get("Resume_str", ""))

        if not resume_text:
            continue

        chunks = chunk_text(resume_text)

        for chunk_index, chunk in enumerate(chunks):
            page_content = (
                f"Resume ID: {resume_id}\n"
                f"Category: {category}\n"
                f"Chunk: {chunk_index}\n\n"
                f"Resume Content:\n{chunk}"
            )

            yield Document(
                page_content=page_content,
                metadata={
                    "resume_id": resume_id,
                    "category": category,
                    "chunk_index": chunk_index,
                    "row_index": int(row_index),
                },
            )


def main() -> None:
    validate_required_settings()

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        api_key=OPENAI_API_KEY,
    )

    vector_store = None
    total_docs = 0

    for batch_number, docs_batch in enumerate(batch_items(build_documents(), BATCH_SIZE), start=1):
        if vector_store is None:
            print(
                f"Creating Qdrant collection '{COLLECTION_NAME}' "
                f"with first batch of {len(docs_batch)} chunks..."
            )
            vector_store = QdrantVectorStore.from_documents(
                documents=docs_batch,
                embedding=embeddings,
                collection_name=COLLECTION_NAME,
                url=QDRANT_URL,
                api_key=QDRANT_API_KEY,
                force_recreate=RESET_COLLECTION,
            )
        else:
            vector_store.add_documents(docs_batch)

        total_docs += len(docs_batch)
        print(f"Uploaded batch {batch_number} | total chunks: {total_docs:,}")

    print("\nDone.")
    print(f"Collection name: {COLLECTION_NAME}")
    print(f"Total uploaded chunks: {total_docs:,}")


if __name__ == "__main__":
    main()
