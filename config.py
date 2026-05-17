import os
from dotenv import load_dotenv
load_dotenv()

def get_secret(name: str, default: str | None = None) -> str | None:
    value = os.getenv(name)
    if value not in (None, ""):
        return value

    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        # Safe fallback for non-Streamlit scripts.
        pass

    return default

def get_int(name: str, default: int) -> int:
    value = get_secret(name, str(default))
    try:
        return int(value)
    except Exception:
        return default


def get_bool(name: str, default: bool) -> bool:
   
    value = str(get_secret(name, str(default))).lower().strip()
    return value in {"true", "1", "yes", "y"}


OPENAI_API_KEY = get_secret("OPENAI_API_KEY")
QDRANT_URL = get_secret("QDRANT_URL")
QDRANT_API_KEY = get_secret("QDRANT_API_KEY")

COLLECTION_NAME = get_secret("QDRANT_COLLECTION_NAME", "resume_collection")
EMBEDDING_MODEL = get_secret("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = get_secret("OPENAI_CHAT_MODEL", "gpt-4o-mini")

TOP_K = get_int("TOP_K", 5)
IDR_PER_USD = get_int("IDR_PER_USD", 17_000)

MAX_ROWS = get_int("MAX_ROWS", 0)
CHUNK_SIZE = get_int("CHUNK_SIZE", 1200)
CHUNK_OVERLAP = get_int("CHUNK_OVERLAP", 150)
BATCH_SIZE = get_int("BATCH_SIZE", 64)
RESET_COLLECTION = get_bool("RESET_COLLECTION", True)


def validate_required_settings() -> None:
    missing = []
    if not OPENAI_API_KEY:
        missing.append("OPENAI_API_KEY")
    if not QDRANT_URL:
        missing.append("QDRANT_URL")
    if not QDRANT_API_KEY:
        missing.append("QDRANT_API_KEY")

    if missing:
        raise ValueError(
            "Missing settings: "
            + ", ".join(missing)
            + ". Fill .env locally or Streamlit secrets on deployment."
        )
