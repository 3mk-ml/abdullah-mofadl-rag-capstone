from __future__ import annotations

from openai import OpenAI

from .config import OPENAI_API_KEY, OPENAI_MODEL

SYSTEM = """You are AgriRAG, an evidence-grounded agriculture assistant.
Answer ONLY from the supplied sources. If the sources do not contain enough evidence, say so clearly.
Do not invent crop thresholds, fertilizer doses, irrigation quantities, or scientific claims.
Cite every factual paragraph using [1], [2], etc. Keep units exactly as stated in the sources.
If the user asks in Arabic, answer in Arabic. If the user asks in English, answer in English."""


def answer(question: str, contexts: list[dict]) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")
    client = OpenAI(api_key=OPENAI_API_KEY)
    source_block = "\n\n".join(
        f"[{i}] Source: {c['source']}, page {c.get('page') or '?'}\n{c['text']}"
        for i, c in enumerate(contexts, start=1)
    )
    prompt = f"Sources:\n{source_block}\n\nQuestion: {question}"
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text
