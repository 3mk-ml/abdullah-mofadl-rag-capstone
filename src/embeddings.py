from __future__ import annotations

from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL


@lru_cache(maxsize=1)
def model() -> SentenceTransformer:
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_passages(texts: list[str]) -> np.ndarray:
    prepared = [f"passage: {t}" for t in texts]
    return model().encode(prepared, normalize_embeddings=True, show_progress_bar=False)


def embed_query(text: str) -> np.ndarray:
    return model().encode([f"query: {text}"], normalize_embeddings=True, show_progress_bar=False)[0]
