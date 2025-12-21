import os
from pathlib import Path
from dataclasses import dataclass


def _load_dotenv() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    root = Path(__file__).resolve().parents[1]
    load_dotenv(root / ".env")


def _get_env(name, default, cast=str):
    value = os.environ.get(name, default)
    if cast is str:
        return value
    try:
        return cast(value)
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    papers_dir: str
    data_dir: str
    corpus_path: str
    chroma_dir: str
    collection_name: str
    embedding_model: str
    chat_model: str
    top_k: int
    max_context_chars: int
    min_chunk_chars: int

    @classmethod
    def from_env(cls) -> "Settings":
        _load_dotenv()
        papers_dir = _get_env("PAPERS_DIR", "papers")
        data_dir = _get_env("DATA_DIR", "data")
        corpus_path = _get_env("CORPUS_PATH", os.path.join(data_dir, "papers.json"))
        chroma_dir = _get_env("CHROMA_DIR", os.path.join(data_dir, "chroma"))
        collection_name = _get_env("CHROMA_COLLECTION", "paper_chunks")
        embedding_model = _get_env("EMBEDDING_MODEL", "text-embedding-3-small")
        chat_model = _get_env("CHAT_MODEL", "gpt-4o-mini")
        top_k = _get_env("TOP_K", 5, cast=int)
        max_context_chars = _get_env("MAX_CONTEXT_CHARS", 12000, cast=int)
        min_chunk_chars = _get_env("MIN_CHUNK_CHARS", 5, cast=int)
        return cls(
            papers_dir=papers_dir,
            data_dir=data_dir,
            corpus_path=corpus_path,
            chroma_dir=chroma_dir,
            collection_name=collection_name,
            embedding_model=embedding_model,
            chat_model=chat_model,
            top_k=top_k,
            max_context_chars=max_context_chars,
            min_chunk_chars=min_chunk_chars,
        )
