from __future__ import annotations

import hashlib
import re

from .config import CHUNK_OVERLAP, CHUNK_SIZE
from .models import Chunk

SEPARATORS = ["\n\n", "\n", ". ", "؟ ", "! ", "؛ ", "; ", "، ", ", ", " "]


def _split_recursive(text: str, max_chars: int, separators: list[str]) -> list[str]:
    if len(text) <= max_chars:
        return [text.strip()] if text.strip() else []
    if not separators:
        return [text[i : i + max_chars].strip() for i in range(0, len(text), max_chars)]

    sep = separators[0]
    parts = text.split(sep)
    if len(parts) == 1:
        return _split_recursive(text, max_chars, separators[1:])

    out: list[str] = []
    current = ""
    for part in parts:
        candidate = (current + sep + part).strip() if current else part.strip()
        if len(candidate) <= max_chars:
            current = candidate
        else:
            if current:
                out.extend(_split_recursive(current, max_chars, separators[1:]))
            current = part.strip()
    if current:
        out.extend(_split_recursive(current, max_chars, separators[1:]))
    return out


def recursive_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    # E5 maxes out at 512 wordpiece tokens. Character sizing is a practical,
    # deterministic proxy for this capstone and is later tuned against Recall@5.
    max_chars = chunk_size * 4
    overlap_chars = overlap * 4
    base = _split_recursive(text, max_chars, SEPARATORS)
    if not base or overlap_chars <= 0:
        return base

    chunks: list[str] = []
    prev_tail = ""
    for piece in base:
        merged = (prev_tail + "\n" + piece).strip() if prev_tail else piece
        chunks.append(merged)
        prev_tail = piece[-overlap_chars:]
    return chunks


def build_chunks(docs: list[dict]) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc in docs:
        for local_idx, text in enumerate(recursive_chunks(doc["text"])):
            if len(text) < 80:
                continue
            raw_id = f'{doc["source"]}|{doc.get("page")}|{local_idx}|{text[:80]}'
            chunk_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:20]
            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=text,
                    source=doc["source"],
                    page=doc.get("page"),
                    metadata={"path": doc.get("path", "")},
                )
            )
    return chunks
