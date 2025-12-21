import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from paper_chatbot.config import Settings
from paper_chatbot.corpus import build_document, find_content_lists


def main() -> None:
    settings = Settings.from_env()
    parser = argparse.ArgumentParser(description="Build a combined paper corpus JSON.")
    parser.add_argument("--papers-dir", default=settings.papers_dir)
    parser.add_argument("--output", default=settings.corpus_path)
    parser.add_argument("--min-chars", type=int, default=settings.min_chunk_chars)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    papers_dir = Path(args.papers_dir)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    content_files = find_content_lists(papers_dir)
    if args.limit:
        content_files = content_files[: args.limit]

    papers = []
    for content_path in content_files:
        doc = build_document(content_path, papers_dir, args.min_chars)
        if doc:
            papers.append(doc)

    payload = {
        "schema_version": 1,
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "num_papers": len(papers),
        "papers": papers,
    }

    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True))
    print(f"Wrote {len(papers)} papers to {output_path}")


if __name__ == "__main__":
    main()
