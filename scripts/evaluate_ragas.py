from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import os
import re

import pandas as pd
from openai import AsyncOpenAI, RateLimitError

from src.config import (
    COHERE_API_KEY,
    COHERE_BASE_URL,
    COHERE_MODEL,
    GEMINI_API_KEY,
    GEMINI_BASE_URL,
    GEMINI_EMBEDDING_MODEL,
    GEMINI_MODEL,
    GROQ_API_KEY,
    GROQ_BASE_URL,
    GROQ_MODEL,
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
)
from src.rag_pipeline import ask

from ragas.llms.base import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.context_precision import ContextPrecision
from ragas.metrics.collections.context_recall import ContextRecall


def _retry_delay_seconds(exc: Exception) -> float:
    """Parse a provider's 'try again in XmYs' hint from a 429."""
    message = str(exc)
    match = re.search(
        r"try again in\s+(?:(\d+)m)?([0-9.]+)s",
        message,
        flags=re.IGNORECASE,
    )
    if match:
        minutes = int(match.group(1) or 0)
        seconds = float(match.group(2))
        return minutes * 60 + seconds
    return 60.0


async def _with_quota_wait(label: str, factory, max_waits: int = 8):
    """Wait for a provider rate-limit window instead of failing the workflow."""
    waits = 0
    while True:
        try:
            return await factory()
        except RateLimitError as exc:
            waits += 1
            if waits > max_waits:
                raise
            delay = _retry_delay_seconds(exc) + 5.0
            print(
                f"Provider quota reached during {label}; "
                f"waiting {delay:.1f}s then retrying automatically "
                f"(wait {waits}/{max_waits}).",
                flush=True,
            )
            await asyncio.sleep(delay)


async def _ask_with_quota_wait(question: str, top_n: int = 5) -> dict:
    waits = 0
    while True:
        try:
            return await asyncio.to_thread(ask, question, top_n=top_n)
        except RateLimitError as exc:
            waits += 1
            if waits > 8:
                raise
            delay = _retry_delay_seconds(exc) + 5.0
            print(
                f"Provider quota reached during answer generation; "
                f"waiting {delay:.1f}s then retrying automatically "
                f"(wait {waits}/8).",
                flush=True,
            )
            await asyncio.sleep(delay)


async def score_item(item: dict, result: dict, scorers: dict) -> dict:
    question = item["question"]
    reference = item["ground_truth"]
    response = result["answer"]
    contexts = [s["text"] for s in result["sources"]]

    faith = await _with_quota_wait(
        "faithfulness",
        lambda: scorers["faithfulness"].ascore(
            user_input=question,
            response=response,
            retrieved_contexts=contexts,
        ),
    )
    relevance = await _with_quota_wait(
        "answer relevancy",
        lambda: scorers["answer_relevancy"].ascore(
            user_input=question,
            response=response,
        ),
    )
    precision = await _with_quota_wait(
        "context precision",
        lambda: scorers["context_precision"].ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        ),
    )
    recall = await _with_quota_wait(
        "context recall",
        lambda: scorers["context_recall"].ascore(
            user_input=question,
            reference=reference,
            retrieved_contexts=contexts,
        ),
    )

    return {
        "id": item.get("id", ""),
        "question": question,
        "reference": reference,
        "response": response,
        "faithfulness": float(faith.value),
        "answer_relevancy": float(relevance.value),
        "context_precision": float(precision.value),
        "context_recall": float(recall.value),
        "latency_seconds": float(result.get("latency_seconds", 0.0)),
    }


def save_checkpoint(
    path: Path,
    provider: str,
    evaluator_model: str,
    generator_model: str,
    rows: list[dict],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "llm_provider": provider,
        "evaluator_model": evaluator_model,
        "generator_model": generator_model,
        "completed": len(rows),
        "rows": rows,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_checkpoint(
    path: Path,
    provider: str,
    evaluator_model: str,
    generator_model: str,
) -> list[dict]:
    if not path.exists():
        return []
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []
    if (
        payload.get("llm_provider") != provider
        or payload.get("evaluator_model") != evaluator_model
        or payload.get("generator_model") != generator_model
    ):
        print("Ignoring checkpoint from a different provider/model.", flush=True)
        return []
    rows = payload.get("rows") or []
    print(f"Resuming from checkpoint: {len(rows)} completed question(s).", flush=True)
    return rows


async def run(
    golden_path: Path,
    out_csv: Path,
    evaluator_model: str,
    checkpoint_path: Path,
) -> None:
    if LLM_PROVIDER == "cohere":
        if not COHERE_API_KEY:
            raise SystemExit("COHERE_API_KEY is not set")
        api_key = COHERE_API_KEY
        base_url = COHERE_BASE_URL
        embedding_model = "intfloat/multilingual-e5-small"
        # Cohere exposes an OpenAI-compatible endpoint with structured outputs.
        # Use RAGAS' stable OpenAI/Instructor adapter against that endpoint.
        ragas_provider = "openai"
    elif LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise SystemExit("GROQ_API_KEY is not set")
        api_key = GROQ_API_KEY
        base_url = GROQ_BASE_URL
        embedding_model = "intfloat/multilingual-e5-small"
        # RAGAS 0.4.3's native "groq" Instructor adapter is currently
        # incompatible with the installed Instructor enum (GENAI error).
        # Groq exposes an OpenAI-compatible endpoint, so use the stable
        # OpenAI adapter while keeping the Groq base URL/client.
        ragas_provider = "openai"
    elif LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise SystemExit("GEMINI_API_KEY is not set")
        api_key = GEMINI_API_KEY
        base_url = GEMINI_BASE_URL
        embedding_model = GEMINI_EMBEDDING_MODEL
        ragas_provider = "openai"
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise SystemExit("OPENAI_API_KEY is not set")
        api_key = OPENAI_API_KEY
        base_url = None
        embedding_model = "text-embedding-3-small"
        ragas_provider = "openai"
    else:
        raise SystemExit(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; "
            "use 'openai', 'gemini', 'groq', or 'cohere'."
        )

    import ragas
    print(f"RAGAS version: {getattr(ragas, '__version__', 'unknown')}", flush=True)
    print(f"LLM provider: {LLM_PROVIDER}", flush=True)
    print(f"Evaluator model: {evaluator_model}", flush=True)
    print(f"Evaluator embedding model: {embedding_model}", flush=True)

    raw = json.loads(golden_path.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) else raw
    items = [x for x in items if x.get("ground_truth")][:20]
    if len(items) < 20:
        raise SystemExit("Need 20 manually verified questions with ground_truth for RAGAS.")

    client_kwargs = {"api_key": api_key, "max_retries": 10}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)

    evaluator_max_tokens = int(os.getenv("RAGAS_MAX_TOKENS", "4096"))
    print(
        f"RAGAS evaluator max tokens/request: {evaluator_max_tokens}",
        flush=True,
    )

    evaluator_kwargs = {
        "provider": ragas_provider,
        "client": client,
        "max_tokens": evaluator_max_tokens,
    }
    if LLM_PROVIDER == "groq" and evaluator_model.startswith("openai/gpt-oss-"):
        evaluator_kwargs["reasoning_effort"] = "low"

    evaluator_llm = llm_factory(
        evaluator_model,
        **evaluator_kwargs,
    )

    if LLM_PROVIDER in {"groq", "cohere"}:
        evaluator_embeddings = embedding_factory(
            "huggingface",
            model=embedding_model,
            interface="modern",
        )
    else:
        evaluator_embeddings = embedding_factory(
            "openai",
            model=embedding_model,
            client=client,
        )

    scorers = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm,
            embeddings=evaluator_embeddings,
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }

    rows = load_checkpoint(
        checkpoint_path,
        provider=LLM_PROVIDER,
        evaluator_model=evaluator_model,
        generator_model=GROQ_MODEL if LLM_PROVIDER == "groq" else evaluator_model,
    )
    done_ids = {r.get("id") for r in rows}

    for i, item in enumerate(items, start=1):
        if item.get("id") in done_ids:
            print(f"[{i:02d}/20] checkpoint HIT {item['id']} - skipping", flush=True)
            continue

        print(f"[{i:02d}/20] Generating answer: {item['id']}", flush=True)
        rag_result = await _ask_with_quota_wait(item["question"], top_n=5)
        row = await score_item(item, rag_result, scorers)
        rows.append(row)
        done_ids.add(item.get("id"))

        # Save immediately so a quota/rate-limit failure never loses completed work.
        save_checkpoint(
            checkpoint_path,
            provider=LLM_PROVIDER,
            evaluator_model=evaluator_model,
            generator_model=GROQ_MODEL if LLM_PROVIDER == "groq" else evaluator_model,
            rows=rows,
        )

        print(
            f"[{i:02d}/20] Scored {item['id']}: "
            f"faith={row['faithfulness']:.3f}, "
            f"relevancy={row['answer_relevancy']:.3f}, "
            f"precision={row['context_precision']:.3f}, "
            f"recall={row['context_recall']:.3f}",
            flush=True,
        )

    order = {item["id"]: idx for idx, item in enumerate(items)}
    rows.sort(key=lambda r: order.get(r.get("id", ""), 999))
    df = pd.DataFrame(rows)

    if len(df) != 20:
        raise SystemExit(f"Expected 20 completed RAGAS rows, found {len(df)}.")

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)

    metric_cols = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
    ]
    summary = {
        "questions": len(df),
        **{k: float(v) for k, v in df[metric_cols].mean().to_dict().items()},
        "mean_latency_seconds": float(df["latency_seconds"].mean()),
        "evaluator_model": evaluator_model,
        "generator_model": GROQ_MODEL if LLM_PROVIDER == "groq" else evaluator_model,
        "evaluator_max_tokens": evaluator_max_tokens,
        "llm_provider": LLM_PROVIDER,
    }

    summary_path = out_csv.with_name("ragas_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nRAGAS summary:", flush=True)
    print(json.dumps(summary, indent=2), flush=True)
    print(f"Saved: {out_csv}", flush=True)
    print(f"Saved: {summary_path}", flush=True)
    print(f"Checkpoint: {checkpoint_path}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("data/eval/golden_questions.json"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/eval/ragas_report.csv"),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("data/eval/ragas_checkpoint.json"),
    )

    if LLM_PROVIDER == "cohere":
        default_model = COHERE_MODEL
    elif LLM_PROVIDER == "groq":
        default_model = GROQ_MODEL
    elif LLM_PROVIDER == "gemini":
        default_model = GEMINI_MODEL
    else:
        default_model = OPENAI_MODEL

    parser.add_argument("--evaluator-model", default=default_model)
    args = parser.parse_args()
    asyncio.run(
        run(
            args.golden,
            args.out,
            args.evaluator_model,
            args.checkpoint,
        )
    )


if __name__ == "__main__":
    main()
