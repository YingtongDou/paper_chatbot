import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import chromadb
from openai import OpenAI

from .corpus import load_corpus


def iter_chunks(papers: Iterable[Dict]) -> Iterable[Tuple[str, str, Dict]]:
    for paper in papers:
        paper_id = paper.get("paper_id")
        title = paper.get("title")
        for chunk in paper.get("chunks", []):
            chunk_id = chunk.get("chunk_id")
            text = chunk.get("text", "")
            metadata = {
                "paper_id": paper_id,
                "paper_title": title,
                "page_idx": chunk.get("page_idx"),
                "chunk_index": chunk.get("chunk_index"),
                "chunk_type": chunk.get("type"),
                "source_path": paper.get("source_path"),
                "pdf_path": paper.get("pdf_path"),
            }
            yield chunk_id, text, metadata


def create_openai_client() -> OpenAI:
    return OpenAI()


def embed_texts(
    client: OpenAI,
    texts: Sequence[str],
    model: str,
    max_retries: int = 5,
    backoff_s: float = 1.5,
) -> List[List[float]]:
    last_error: Optional[Exception] = None
    for attempt in range(max_retries):
        try:
            response = client.embeddings.create(model=model, input=list(texts))
            return [item.embedding for item in response.data]
        except Exception as exc:  # noqa: BLE001
            last_error = exc
            sleep_s = backoff_s * (2 ** attempt)
            time.sleep(sleep_s)
    if last_error:
        raise last_error
    raise RuntimeError("Failed to generate embeddings")


def get_collection(
    chroma_dir: str,
    name: str,
    reset: bool = False,
) -> chromadb.api.models.Collection.Collection:
    client = chromadb.PersistentClient(path=chroma_dir)
    if reset:
        try:
            client.delete_collection(name)
        except Exception:  # noqa: BLE001
            pass
    return client.get_or_create_collection(name=name, metadata={"hnsw:space": "cosine"})


def load_papers(corpus_path: str) -> List[Dict]:
    return load_corpus(Path(corpus_path))
