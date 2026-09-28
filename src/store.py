from __future__ import annotations

import json
from pathlib import Path

import chromadb

from .config import CHROMA_DIR, CHUNKS_PATH, COLLECTION_NAME
from .models import Chunk


def client() -> chromadb.PersistentClient:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def collection():
    return client().get_or_create_collection(name=COLLECTION_NAME, metadata={"hnsw:space": "cosine"})


def save_chunks(chunks: list[Chunk]) -> None:
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CHUNKS_PATH.open("w", encoding="utf-8") as f:
        for ch in chunks:
            f.write(json.dumps(ch.to_dict(), ensure_ascii=False) + "\n")


def load_chunks() -> list[dict]:
    if not CHUNKS_PATH.exists():
        return []
    with CHUNKS_PATH.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def reset_collection() -> None:
    c = client()
    try:
        c.delete_collection(COLLECTION_NAME)
    except Exception:
        pass
