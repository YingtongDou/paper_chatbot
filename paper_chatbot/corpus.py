import json
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


def find_content_lists(papers_dir: Path) -> List[Path]:
    return sorted(papers_dir.rglob("*_content_list.json"))


def load_content_list(path: Path) -> List[Dict]:
    return json.loads(path.read_text())


def _normalize_whitespace(text: str) -> str:
    lines = []
    for line in text.splitlines():
        cleaned = re.sub(r"[ \t]+", " ", line).strip()
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def _extract_text(item: Dict) -> Tuple[Optional[str], str]:
    item_type = item.get("type", "unknown")
    raw = ""
    if item_type == "text":
        raw = item.get("text", "")
    elif item_type == "list":
        list_items = item.get("list_items") or []
        lines = []
        for entry in list_items:
            entry = entry.strip()
            if not entry:
                continue
            if entry.lstrip().startswith(("-", "*")):
                lines.append(entry)
            else:
                lines.append(f"- {entry}")
        raw = "\n".join(lines)
    elif item_type == "equation":
        raw = item.get("text", "")
    elif item_type == "code":
        parts = []
        captions = [c.strip() for c in (item.get("code_caption") or []) if c.strip()]
        if captions:
            parts.append(" ".join(captions))
        body = item.get("code_body", "")
        if body and body.strip():
            parts.append(body)
        raw = "\n".join(parts)
    elif item_type == "image":
        parts = []
        for entry in (item.get("image_caption") or []) + (item.get("image_footnote") or []):
            entry = entry.strip()
            if entry:
                parts.append(entry)
        raw = "\n".join(parts)
    else:
        raw = item.get("text", "") or ""

    text = _normalize_whitespace(raw)
    if not text:
        return None, item_type
    return text, item_type


def _infer_title(items: Iterable[Dict], fallback: str) -> str:
    for item in items:
        if item.get("type") == "text" and item.get("text_level") == 1:
            text = _normalize_whitespace(item.get("text", ""))
            if text:
                return text
    for item in items:
        text, _ = _extract_text(item)
        if text and len(text) >= 12:
            return text
    return fallback


def _infer_pdf_path(folder: Path) -> Optional[Path]:
    pdfs = sorted(folder.glob("*.pdf"))
    if not pdfs:
        return None
    for pdf in pdfs:
        if "origin" in pdf.name:
            return pdf
    return pdfs[0]


def _paper_label(folder_name: str) -> str:
    if ".pdf-" in folder_name:
        base = folder_name.split(".pdf-", 1)[0]
    else:
        base = folder_name
    return base.replace("_", " ")


def build_document(
    content_path: Path,
    papers_dir: Path,
    min_chars: int,
) -> Optional[Dict]:
    items = load_content_list(content_path)
    if not isinstance(items, list):
        return None

    folder = content_path.parent
    paper_id = folder.name
    fallback_title = _paper_label(paper_id)
    title = _infer_title(items, fallback_title)

    pdf_path = _infer_pdf_path(folder)
    pdf_rel = None
    if pdf_path is not None:
        try:
            pdf_rel = str(pdf_path.relative_to(papers_dir.parent))
        except ValueError:
            pdf_rel = str(pdf_path)

    chunks = []
    max_page = None
    for item_idx, item in enumerate(items):
        text, item_type = _extract_text(item)
        if not text or len(text) < min_chars:
            continue
        page_idx = item.get("page_idx")
        if isinstance(page_idx, int):
            max_page = page_idx if max_page is None else max(max_page, page_idx)
        chunks.append(
            {
                "chunk_id": f"{paper_id}-{item_idx:06d}",
                "chunk_index": item_idx,
                "text": text,
                "type": item_type,
                "page_idx": page_idx,
                "text_level": item.get("text_level"),
            }
        )

    if not chunks:
        return None

    source_rel = str(content_path)
    try:
        source_rel = str(content_path.relative_to(papers_dir.parent))
    except ValueError:
        pass

    num_pages = max_page + 1 if isinstance(max_page, int) else None

    return {
        "paper_id": paper_id,
        "title": title,
        "source_path": source_rel,
        "pdf_path": pdf_rel,
        "num_pages": num_pages,
        "num_chunks": len(chunks),
        "chunks": chunks,
    }


def load_corpus(path: Path) -> List[Dict]:
    data = json.loads(path.read_text())
    if isinstance(data, list):
        return data
    if isinstance(data, dict) and "papers" in data:
        return data["papers"]
    raise ValueError("Unsupported corpus format")
