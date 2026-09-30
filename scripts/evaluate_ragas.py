from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json
import os

import pandas as pd
from openai import AsyncOpenAI

from src.config import (
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


# RAGAS 0.4.3 modern component API. Import from the concrete modules so that
# CI shows the real failing import instead of hiding it behind a generic error.
from ragas.llms.base import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections.faithfulness import Faithfulness
from ragas.metrics.collections.answer_relevancy import AnswerRelevancy
from ragas.metrics.collections.context_precision import ContextPrecision
from ragas.metrics.collections.context_recall import ContextRecall


async def score_item(item: dict, result: dict, scorers: dict) -> dict:
    question = item["question"]
    reference = item["ground_truth"]
    response = result["answer"]
    contexts = [s["text"] for s in result["sources"]]

    faith = await scorers["faithfulness"].ascore(
        user_input=question,
        response=response,
        retrieved_contexts=contexts,
    )
    relevance = await scorers["answer_relevancy"].ascore(
        user_input=question,
        response=response,
    )
    precision = await scorers["context_precision"].ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=contexts,
    )
    recall = await scorers["context_recall"].ascore(
        user_input=question,
        reference=reference,
        retrieved_contexts=contexts,
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


async def run(golden_path: Path, out_csv: Path, evaluator_model: str) -> None:
    if LLM_PROVIDER == "groq":
        if not GROQ_API_KEY:
            raise SystemExit("GROQ_API_KEY is not set")
        api_key = GROQ_API_KEY
        base_url = GROQ_BASE_URL
        # RAGAS answer relevancy uses the same local E5 family as the retriever.
        # This avoids a paid embedding API and the HF model is public.
        embedding_model = "intfloat/multilingual-e5-small"
    elif LLM_PROVIDER == "gemini":
        if not GEMINI_API_KEY:
            raise SystemExit("GEMINI_API_KEY is not set")
        api_key = GEMINI_API_KEY
        base_url = GEMINI_BASE_URL
        embedding_model = GEMINI_EMBEDDING_MODEL
    elif LLM_PROVIDER == "openai":
        if not OPENAI_API_KEY:
            raise SystemExit("OPENAI_API_KEY is not set")
        api_key = OPENAI_API_KEY
        base_url = None
        embedding_model = "text-embedding-3-small"
    else:
        raise SystemExit(
            f"Unsupported LLM_PROVIDER={LLM_PROVIDER!r}; "
            "use 'openai', 'gemini', or 'groq'."
        )

    import ragas
    print(f"RAGAS version: {getattr(ragas, '__version__', 'unknown')}")
    print(f"LLM provider: {LLM_PROVIDER}")
    print(f"Evaluator model: {evaluator_model}")
    print(f"Evaluator embedding model: {embedding_model}")

    raw = json.loads(golden_path.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) else raw
    items = [x for x in items if x.get("ground_truth")][:20]
    if len(items) < 20:
        raise SystemExit("Need 20 manually verified questions with ground_truth for RAGAS.")

    client_kwargs = {"api_key": api_key, "max_retries": 10}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = AsyncOpenAI(**client_kwargs)

    # RAGAS uses an OpenAI-compatible client adapter. For Groq and Gemini,
    # the client points at the provider's OpenAI-compatible base URL.
    evaluator_max_tokens = int(os.getenv("RAGAS_MAX_TOKENS", "512"))
    print(f"RAGAS evaluator max tokens/request: {evaluator_max_tokens}")

    evaluator_llm = llm_factory(
        evaluator_model,
        provider="openai",
        client=client,
        max_tokens=evaluator_max_tokens,
        reasoning_effort="low" if LLM_PROVIDER == "groq" else None,
    )

    if LLM_PROVIDER == "groq":
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

    rows: list[dict] = []
    for i, item in enumerate(items, start=1):
        print(f"Generating answer {i}/20: {item['id']}")
        rag_result = ask(item["question"], top_n=5)
        row = await score_item(item, rag_result, scorers)
        rows.append(row)
        print(
            f"Scored {i}/20 {item['id']}: "
            f"faith={row['faithfulness']:.3f}, "
            f"relevancy={row['answer_relevancy']:.3f}, "
            f"precision={row['context_precision']:.3f}, "
            f"recall={row['context_recall']:.3f}"
        )

    df = pd.DataFrame(rows)
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
        "evaluator_max_tokens": evaluator_max_tokens,
        "llm_provider": LLM_PROVIDER,
    }

    summary_path = out_csv.with_name("ragas_summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print("\nRAGAS summary:")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_csv}")
    print(f"Saved: {summary_path}")


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
    if LLM_PROVIDER == "groq":
        default_model = GROQ_MODEL
    elif LLM_PROVIDER == "gemini":
        default_model = GEMINI_MODEL
    else:
        default_model = OPENAI_MODEL
    parser.add_argument("--evaluator-model", default=default_model)
    args = parser.parse_args()
    asyncio.run(run(args.golden, args.out, args.evaluator_model))


if __name__ == "__main__":
    main()
