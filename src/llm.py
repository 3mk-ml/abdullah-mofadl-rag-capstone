from __future__ import annotations

from openai import OpenAI

from .config import (
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)

SYSTEM = """You are AgriRAG, an evidence-grounded agriculture assistant.
Answer ONLY from the supplied sources. If the sources do not contain enough evidence, say so clearly.
Do not invent crop thresholds, fertilizer doses, irrigation quantities, or scientific claims.
Cite every factual paragraph using [1], [2], etc. Keep units exactly as stated in the sources.
If the user asks in Arabic, answer in Arabic. If the user asks in English, answer in English."""


def _source_prompt(question: str, contexts: list[dict]) -> str:
    source_block = "\n\n".join(
        f"[{i}] Source: {c['source']}, page {c.get('page') or '?'}\n{c['text']}"
        for i, c in enumerate(contexts, start=1)
    )
    return f"Sources:\n{source_block}\n\nQuestion: {question}"


def answer(question: str, contexts: list[dict]) -> str:
    prompt = _source_prompt(question, contexts)

    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise RuntimeError("GROQ_API_KEY is not set")
        client = OpenAI(
            api_key=GROQ_API_KEY,
            base_url=GROQ_BASE_URL,
            max_retries=10,
        )
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            max_tokens=160,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    if LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise RuntimeError("GEMINI_API_KEY is not set")

        # Gemini exposes an OpenAI-compatible endpoint. Chat Completions is
        # used because it is explicitly supported by Google's compatibility API.
        client = OpenAI(
            api_key=GEMINI_API_KEY,
            base_url=GEMINI_BASE_URL,
            max_retries=8,
        )
        resp = client.chat.completions.create(
            model=GEMINI_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": prompt},
            ],
        )
        return (resp.choices[0].message.content or "").strip()

    if LLM_PROVIDER != "openai":
        raise RuntimeError(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; use 'openai', 'gemini', or 'groq'."
        )

    if not OPENAI_API_KEY:
        raise RuntimeError("OPENAI_API_KEY is not set")

    client = OpenAI(api_key=OPENAI_API_KEY, max_retries=8)
    resp = client.responses.create(
        model=OPENAI_MODEL,
        input=[
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": prompt},
        ],
    )
    return resp.output_text
