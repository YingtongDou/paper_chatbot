import argparse
import sys
import time
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from paper_chatbot.chroma_index import (
    create_openai_client,
    embed_texts,
    get_collection,
)
from paper_chatbot.config import Settings
from paper_chatbot.corpus import load_corpus
from paper_chatbot.chroma_index import iter_chunks


def chunked(iterable, size):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Create a ChromaDB index from the corpus.")
    parser.add_argument("--corpus", default=settings.corpus_path)
    parser.add_argument("--chroma-dir", default=settings.chroma_dir)
    parser.add_argument("--collection", default=settings.collection_name)
    parser.add_argument("--embedding-model", default=settings.embedding_model)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    corpus_path = Path(args.corpus)
    papers = load_corpus(corpus_path)

    collection = get_collection(args.chroma_dir, args.collection, reset=args.reset)

    existing_ids = set()
    if args.skip_existing and not args.reset:
        try:
            existing = collection.get(include=[])
        except TypeError:
            existing = collection.get()
        existing_ids = set(existing.get("ids", []))
        print(f"Found {len(existing_ids)} existing embeddings")

    openai_client = create_openai_client()

    all_chunks = iter_chunks(papers)
    if args.limit:
        all_chunks = (chunk for idx, chunk in enumerate(all_chunks) if idx < args.limit)

    total_indexed = 0
    total_skipped = 0

    for batch in chunked(all_chunks, args.batch_size):
        ids = []
        texts = []
        metas = []
        for chunk_id, text, meta in batch:
            if args.skip_existing and chunk_id in existing_ids:
                total_skipped += 1
                continue
            ids.append(chunk_id)
            texts.append(text)
            metas.append(meta)

        if not ids:
            continue

        embeddings = embed_texts(openai_client, texts, args.embedding_model)
        collection.add(ids=ids, embeddings=embeddings, documents=texts, metadatas=metas)
        total_indexed += len(ids)
        print(f"Indexed {total_indexed} chunks")
        if args.sleep:
            time.sleep(args.sleep)

    print(
        "Done. Indexed {indexed} chunks (skipped {skipped}).".format(
            indexed=total_indexed, skipped=total_skipped
        )
    )


if __name__ == "__main__":
    main()
