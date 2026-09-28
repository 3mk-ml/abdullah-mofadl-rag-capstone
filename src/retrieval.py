from __future__ import annotations

from .bm25_index import load as load_bm25
from .config import BM25_K, RERANK_TOP_N, RRF_K, VECTOR_K
from .embeddings import embed_query
from .reranker import rerank
from .store import collection, load_chunks


def _rrf(vector_ids: list[str], bm25_ids: list[str], rrf_k: int = RRF_K) -> list[str]:
    scores: dict[str, float] = {}
    for rank, cid in enumerate(vector_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    for rank, cid in enumerate(bm25_ids, start=1):
        scores[cid] = scores.get(cid, 0.0) + 1.0 / (rrf_k + rank)
    return sorted(scores, key=scores.get, reverse=True)


def hybrid_candidates(query: str, vector_k: int = VECTOR_K, bm25_k: int = BM25_K) -> list[dict]:
    chunks = load_chunks()
    by_id = {c["id"]: c for c in chunks}

    qvec = embed_query(query)
    vres = collection().query(query_embeddings=[qvec.tolist()], n_results=vector_k)
    vector_ids = vres.get("ids", [[]])[0]

    bm25 = load_bm25()
    bm25_ids = [cid for cid, _ in bm25.search(query, bm25_k)]

    fused = _rrf(vector_ids, bm25_ids)
    return [by_id[cid] for cid in fused if cid in by_id]


def retrieve(query: str, top_n: int = RERANK_TOP_N) -> list[dict]:
    candidates = hybrid_candidates(query)
    return rerank(query, candidates[: max(VECTOR_K, BM25_K) * 2], top_n=top_n)
