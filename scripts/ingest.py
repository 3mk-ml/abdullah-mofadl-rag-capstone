from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse

from src.bm25_index import build as build_bm25
from src.chunking import build_chunks
from src.config import RAW_DIR
from src.embeddings import embed_passages
from src.loaders import load_documents
from src.store import collection, reset_collection, save_chunks


def main(reset: bool = True) -> None:
    docs = load_documents(RAW_DIR)
    if not docs:
        raise SystemExit(f"No PDF/TXT/MD files found in {RAW_DIR}")

    chunks = build_chunks(docs)
    if not chunks:
        raise SystemExit("No chunks were produced. Check extraction.")

    print(f"Loaded {len(docs)} document pages/units")
    print(f"Built {len(chunks)} chunks")

    texts = [c.text for c in chunks]
    vectors = embed_passages(texts)

    if reset:
        reset_collection()
    col = collection()
    col.add(
        ids=[c.id for c in chunks],
        documents=texts,
        embeddings=[v.tolist() for v in vectors],
        metadatas=[
            {"source": c.source, "page": c.page or -1}
            for c in chunks
        ],
    )
    save_chunks(chunks)
    build_bm25([c.id for c in chunks], texts)

    print(f"Indexed {len(chunks)} chunks into Chroma + BM25")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-reset", action="store_true", help="Do not delete the existing collection first")
    args = parser.parse_args()
    main(reset=not args.no_reset)
