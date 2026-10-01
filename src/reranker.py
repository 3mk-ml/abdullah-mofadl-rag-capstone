from __future__ import annotations

import time
from functools import lru_cache

import cohere
from sentence_transformers import CrossEncoder

from .config import (
    COHERE_API_KEY,
    COHERE_RERANK_MODEL,
    RERANKER_MODEL,
    RERANKER_PROVIDER,
)


@lru_cache(maxsize=1)
def model() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def _cohere_rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    if not COHERE_API_KEY:
        raise RuntimeError("COHERE_API_KEY is required when RERANKER_PROVIDER=cohere")

    client = cohere.ClientV2(api_key=COHERE_API_KEY)
    documents = [c["text"] for c in candidates]

    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = client.rerank(
                model=COHERE_RERANK_MODEL,
                query=query,
                documents=documents,
                top_n=min(top_n, len(documents)),
            )
            out: list[dict] = []
            for result in response.results:
                row = dict(candidates[result.index])
                row["rerank_score"] = float(result.relevance_score)
                row["reranker"] = COHERE_RERANK_MODEL
                out.append(row)
            return out
        except Exception as exc:
            last_error = exc
            if getattr(exc, "status_code", None) != 429 or attempt == 2:
                raise
            time.sleep(4 * (attempt + 1))

    raise RuntimeError("Cohere reranking failed") from last_error


def _local_rerank(query: str, candidates: list[dict], top_n: int) -> list[dict]:
    pairs = [(query, c["text"]) for c in candidates]
    scores = model().predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)

    out: list[dict] = []
    for item, score in ranked[:top_n]:
        row = dict(item)
        row["rerank_score"] = float(score)
        row["reranker"] = RERANKER_MODEL
        out.append(row)
    return out


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    if not candidates:
        return []

    if RERANKER_PROVIDER == "cohere":
        return _cohere_rerank(query, candidates, top_n)

    if RERANKER_PROVIDER != "local":
        raise RuntimeError(
            f"Unsupported RERANKER_PROVIDER={RERANKER_PROVIDER!r}; use 'local' or 'cohere'."
        )

    return _local_rerank(query, candidates, top_n)
