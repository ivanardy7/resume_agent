from __future__ import annotations
from typing import Any, Dict, List
from langchain.tools import tool
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from config import (
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    QDRANT_API_KEY,
    QDRANT_URL,
    TOP_K,
    validate_required_settings,
)

_vector_store: QdrantVectorStore | None = None

def get_vector_store() -> QdrantVectorStore:
    global _vector_store

    if _vector_store is None:
        validate_required_settings()
        
        embeddings = OpenAIEmbeddings(
            model=EMBEDDING_MODEL,
            api_key=OPENAI_API_KEY,
        )

        _vector_store = QdrantVectorStore.from_existing_collection(
            embedding=embeddings,
            collection_name=COLLECTION_NAME,
            url=QDRANT_URL,
            api_key=QDRANT_API_KEY,
        )

    return _vector_store


def _format_doc_result(doc: Any, score: float) -> Dict[str, Any]:
    metadata = doc.metadata or {}
    content = doc.page_content or ""

    return {
        "resume_id": metadata.get("resume_id", "UNKNOWN"),
        "category": metadata.get("category", "UNKNOWN"),
        "chunk_index": metadata.get("chunk_index", "UNKNOWN"),
        "score": float(score),
        "content": content[:1500],
    }


@tool
def search_resume_tool(query: str, k: int = TOP_K) -> List[Dict[str, Any]]:
    """
    Search for resume snippets from the Qdrant vector database based on a semantic search query.
    Use this tool to find candidates relevant to specific skills, experience, or job requirements.
    """
    vector_store = get_vector_store()

    results = vector_store.similarity_search_with_score(
        query=query,
        k=k,
    )

    return [_format_doc_result(doc, score) for doc, score in results]
