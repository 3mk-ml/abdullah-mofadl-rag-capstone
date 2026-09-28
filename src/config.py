from __future__ import annotations

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
INDEX_DIR = ROOT / "data" / "index"
EVAL_DIR = ROOT / "data" / "eval"
CHROMA_DIR = INDEX_DIR / "chroma"
BM25_PATH = INDEX_DIR / "bm25.pkl"
CHUNKS_PATH = INDEX_DIR / "chunks.jsonl"

COLLECTION_NAME = os.getenv("CHROMA_COLLECTION", "agri_rag")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "intfloat/multilingual-e5-small")
RERANKER_MODEL = os.getenv(
    "RERANKER_MODEL", "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
)
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-5.6-luna")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
APP_PASSWORD = os.getenv("APP_PASSWORD", "")

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "350"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "60"))
VECTOR_K = int(os.getenv("VECTOR_K", "20"))
BM25_K = int(os.getenv("BM25_K", "20"))
RERANK_TOP_N = int(os.getenv("RERANK_TOP_N", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
