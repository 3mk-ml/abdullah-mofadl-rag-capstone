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

# Keep comfortably below Chroma's backend batch limit. The limit can vary by
# installed Chroma/SQLite build, so ingestion must never assume the entire
# corpus can be inserted in one call.
CHROMA_ADD_BATCH_SIZE = 4000


def add_chunks_in_batches(col, chunks, texts, vectors) -> None:
    total = len(chunks)
    for start in range(0, total, CHROMA_ADD_BATCH_SIZE):
        end = min(start + CHROMA_ADD_BATCH_SIZE, total)
        batch_chunks = chunks[start:end]
        batch_texts = texts[start:end]
        batch_vectors = vectors[start:end]

        col.add(
            ids=[c.id for c in batch_chunks],
            documents=batch_texts,
            embeddings=[v.tolist() for v in batch_vectors],
            metadatas=[
                {
                    "source": c.source,
                    "page": c.page if c.page is not None else -1,
                }
                for c in batch_chunks
            ],
        )
        print(f"Chroma batch indexed: {end}/{total}")


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
    add_chunks_in_batches(col, chunks, texts, vectors)

    save_chunks(chunks)
    build_bm25([c.id for c in chunks], texts)

    print(f"Indexed {len(chunks)} chunks into Chroma + BM25")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--no-reset",
        action="store_true",
        help="Do not delete the existing collection first",
    )
    args = parser.parse_args()
    main(reset=not args.no_reset)
