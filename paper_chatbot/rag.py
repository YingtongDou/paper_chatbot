from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import chromadb
from openai import OpenAI

from .config import Settings

SYSTEM_PROMPT = (
    "You are a helpful research assistant for a collection of papers. "
    "Use the provided context to answer the question. "
    "If the context does not contain the answer, say you do not know."
)


@dataclass
class RAGChatbot:
    openai_client: OpenAI
    collection: chromadb.api.models.Collection.Collection
    embedding_model: str
    chat_model: str
    top_k: int
    max_context_chars: int

    @classmethod
    def from_settings(cls, settings: Settings) -> "RAGChatbot":
        openai_client = OpenAI()
        chroma_client = chromadb.PersistentClient(path=settings.chroma_dir)
        collection = chroma_client.get_or_create_collection(
            name=settings.collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return cls(
            openai_client=openai_client,
            collection=collection,
            embedding_model=settings.embedding_model,
            chat_model=settings.chat_model,
            top_k=settings.top_k,
            max_context_chars=settings.max_context_chars,
        )

    def _embed_query(self, query: str) -> List[float]:
        response = self.openai_client.embeddings.create(
            model=self.embedding_model,
            input=[query],
        )
        return response.data[0].embedding

    def _retrieve(self, query: str, top_k: int):
        query_embedding = self._embed_query(query)
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=top_k,
            include=["documents", "metadatas", "distances"],
        )

    def _build_context(self, results, max_chars: int) -> Tuple[str, str]:
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        ids = results.get("ids", [[]])[0]
        distances = results.get("distances", [[]])[0]

        context_blocks = []
        source_lines = []
        total_chars = 0

        if not ids:
            ids = [None] * len(documents)

        for idx, (doc, meta, chunk_id, distance) in enumerate(
            zip(documents, metadatas, ids, distances), start=1
        ):
            title = (meta or {}).get("paper_title") or (meta or {}).get("paper_id") or "Unknown"
            page_idx = (meta or {}).get("page_idx")
            location = title
            if isinstance(page_idx, int):
                location = f"{location}, page {page_idx + 1}"

            block = f"[Source {idx}] {location}\n{doc.strip()}"
            if context_blocks and total_chars + len(block) > max_chars:
                break
            context_blocks.append(block)
            total_chars += len(block)
            if isinstance(distance, (int, float)):
                score = f"{distance:.3f}"
            else:
                score = "n/a"
            chunk_label = chunk_id or f"chunk-{idx}"
            source_lines.append(f"{idx}. {location} (chunk {chunk_label}, score {score})")

        context = "\n\n".join(context_blocks)
        sources = "\n".join(source_lines) if source_lines else "No sources retrieved."
        return context, sources

    def answer(self, question: str, history: List[Tuple[str, str]], top_k: int | None = None):
        k = top_k or self.top_k
        results = self._retrieve(question, k)
        context, sources = self._build_context(results, self.max_context_chars)

        messages = [{"role": "system", "content": SYSTEM_PROMPT}]
        for user_msg, assistant_msg in history:
            if user_msg:
                messages.append({"role": "user", "content": user_msg})
            if assistant_msg:
                messages.append({"role": "assistant", "content": assistant_msg})

        user_prompt = f"Context:\n{context}\n\nQuestion: {question}"
        messages.append({"role": "user", "content": user_prompt})

        response = self.openai_client.chat.completions.create(
            model=self.chat_model,
            messages=messages,
            temperature=0.2,
        )
        answer = response.choices[0].message.content.strip()
        return answer, sources
