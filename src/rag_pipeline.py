from __future__ import annotations

import time

from .llm import answer
from .retrieval import retrieve


def ask(question: str, top_n: int = 5) -> dict:
    started = time.perf_counter()
    contexts = retrieve(question, top_n=top_n)
    text = answer(question, contexts)
    elapsed = time.perf_counter() - started
    return {
        "answer": text,
        "sources": contexts,
        "latency_seconds": elapsed,
    }
