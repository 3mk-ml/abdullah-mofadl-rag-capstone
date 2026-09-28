from __future__ import annotations

from pathlib import Path
from typing import Iterable

import pymupdf  # PyMuPDF

from .models import Chunk
from .text_utils import normalize_text


def iter_pdf_pages(path: Path) -> Iterable[tuple[int, str]]:
    doc = pymupdf.open(path)
    try:
        for i, page in enumerate(doc):
            text = normalize_text(page.get_text("text") or "")
            if text:
                yield i + 1, text
    finally:
        doc.close()


def iter_text_file(path: Path) -> Iterable[tuple[int | None, str]]:
    text = normalize_text(path.read_text(encoding="utf-8", errors="ignore"))
    if text:
        yield None, text


def load_documents(raw_dir: Path) -> list[dict]:
    docs: list[dict] = []
    for path in sorted(raw_dir.rglob("*")):
        if not path.is_file() or path.name.startswith("."):
            continue
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            for page, text in iter_pdf_pages(path):
                docs.append({"source": path.name, "path": str(path), "page": page, "text": text})
        elif suffix in {".txt", ".md"}:
            for page, text in iter_text_file(path):
                docs.append({"source": path.name, "path": str(path), "page": page, "text": text})
    return docs
