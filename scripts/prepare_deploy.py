from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.download_corpus import main as download_corpus_main  # noqa: E402
from scripts.ingest import main as ingest_main  # noqa: E402
from src.config import BM25_PATH, CHROMA_DIR, CHUNKS_PATH, RAW_DIR  # noqa: E402


def _index_ready() -> bool:
    if not CHUNKS_PATH.exists() or not BM25_PATH.exists() or not CHROMA_DIR.exists():
        return False
    try:
        count = sum(
            1
            for line in CHUNKS_PATH.open(encoding="utf-8")
            if line.strip()
        )
    except OSError:
        return False
    return count >= 100


def main() -> None:
    if _index_ready():
        print("Deployment index already present; skipping rebuild.")
        return

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    CHUNKS_PATH.parent.mkdir(parents=True, exist_ok=True)

    print("Preparing AgriRAG deployment corpus...")
    original_argv = sys.argv[:]
    try:
        # download_corpus.py parses sys.argv itself.
        sys.argv = ["download_corpus.py"]
        download_corpus_main()
    finally:
        sys.argv = original_argv

    print("Building AgriRAG deployment index...")
    ingest_main(reset=True)

    if not _index_ready():
        raise SystemExit("Deployment preparation finished without a valid index.")

    print("Deployment corpus/index ready.")


if __name__ == "__main__":
    main()
