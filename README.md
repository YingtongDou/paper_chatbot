# Paper Chatbot

This project builds a local RAG chatbot over parsed research paper chunks. It:

- Combines per-paper `*_content_list.json` files into a single corpus file.
- Generates OpenAI embeddings for chunks and stores them in a local ChromaDB.
- Serves a Gradio web UI to chat with the papers.

## Setup

> Note: ChromaDB depends on `onnxruntime`, which currently has no wheels for Python 3.13.
> Use Python 3.11 or 3.12 when creating your virtual environment.

1. Create a virtual environment and install dependencies.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Set your OpenAI API key (and optional overrides).

```bash
export OPENAI_API_KEY="..."
```

Optional overrides are listed in `.env.example`. If you create a `.env` file in the
project root, it will be loaded automatically.

## Build the corpus

```bash
python scripts/build_corpus.py --papers-dir papers --output data/papers.json
```

This writes a single JSON file containing one document per paper.

## Build the Chroma index

```bash
python scripts/build_chroma.py --corpus data/papers.json --chroma-dir data/chroma
```

Useful flags:

- `--reset` to rebuild the collection from scratch.
- `--skip-existing` to avoid re-embedding chunks already in Chroma.
- `--batch-size 100` to tune embedding batches.

## Run the chatbot

```bash
python app.py
```

Open the local Gradio URL shown in your terminal.

## Notes

- Chunk metadata includes paper title, page index, and chunk id for citation.
- If the UI says the index is missing, run the corpus + Chroma build steps.
