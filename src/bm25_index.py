from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path

from rank_bm25 import BM25Okapi

from .config import BM25_PATH
from .text_utils import tokenize


@dataclass
class BM25Index:
    ids: list[str]
    texts: list[str]
    bm25: BM25Okapi

    def search(self, query: str, k: int) -> list[tuple[str, float]]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: float(scores[i]), reverse=True)[:k]
        return [(self.ids[i], float(scores[i])) for i in ranked]


def build(ids: list[str], texts: list[str], out_path: Path = BM25_PATH) -> BM25Index:
    idx = BM25Index(ids=ids, texts=texts, bm25=BM25Okapi([tokenize(t) for t in texts]))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(idx, f)
    return idx


def load(path: Path = BM25_PATH) -> BM25Index:
    with path.open("rb") as f:
        return pickle.load(f)
