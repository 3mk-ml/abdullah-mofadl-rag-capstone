from __future__ import annotations

from functools import lru_cache

from sentence_transformers import CrossEncoder

from .config import RERANKER_MODEL


@lru_cache(maxsize=1)
def model() -> CrossEncoder:
    return CrossEncoder(RERANKER_MODEL)


def rerank(query: str, candidates: list[dict], top_n: int = 5) -> list[dict]:
    if not candidates:
        return []
    pairs = [(query, c["text"]) for c in candidates]
    scores = model().predict(pairs, show_progress_bar=False)
    ranked = sorted(zip(candidates, scores), key=lambda x: float(x[1]), reverse=True)
    out: list[dict] = []
    for item, score in ranked[:top_n]:
        row = dict(item)
        row["rerank_score"] = float(score)
        out.append(row)
    return out
