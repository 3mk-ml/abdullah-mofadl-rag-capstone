from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import asyncio
import json

import pandas as pd
from openai import AsyncOpenAI

from src.config import OPENAI_API_KEY, OPENAI_MODEL
from src.rag_pipeline import ask


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
    }


async def run(golden_path: Path, out_csv: Path, evaluator_model: str) -> None:
    if not OPENAI_API_KEY:
        raise SystemExit("OPENAI_API_KEY is not set")

    try:
        from ragas.llms import llm_factory
        from ragas.embeddings.base import embedding_factory
        from ragas.metrics.collections import (
            Faithfulness,
            AnswerRelevancy,
            ContextPrecision,
            ContextRecall,
        )
    except Exception as exc:
        raise SystemExit(
            "RAGAS collections API is unavailable. Install requirements.txt and retry."
        ) from exc

    raw = json.loads(golden_path.read_text(encoding="utf-8"))
    items = raw["questions"] if isinstance(raw, dict) else raw
    items = [x for x in items if x.get("ground_truth")][:20]
    if len(items) < 20:
        raise SystemExit("Need 20 manually verified questions with ground_truth for RAGAS.")

    client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    evaluator_llm = llm_factory(evaluator_model, client=client)
    evaluator_embeddings = embedding_factory(
        "openai", model="text-embedding-3-small", client=client
    )
    scorers = {
        "faithfulness": Faithfulness(llm=evaluator_llm),
        "answer_relevancy": AnswerRelevancy(
            llm=evaluator_llm, embeddings=evaluator_embeddings
        ),
        "context_precision": ContextPrecision(llm=evaluator_llm),
        "context_recall": ContextRecall(llm=evaluator_llm),
    }

    rows: list[dict] = []
    for i, item in enumerate(items, start=1):
        rag_result = ask(item["question"], top_n=5)
        row = await score_item(item, rag_result, scorers)
        rows.append(row)
        print(f"Scored {i}/20: {item['id']}")

    df = pd.DataFrame(rows)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    summary = df[[
        "faithfulness", "answer_relevancy", "context_precision", "context_recall"
    ]].mean().to_dict()
    summary_path = out_csv.with_name("ragas_summary.json")
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"Saved: {out_csv}")
    print(f"Saved: {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_questions.json"))
    parser.add_argument("--out", type=Path, default=Path("data/eval/ragas_report.csv"))
    parser.add_argument("--evaluator-model", default=OPENAI_MODEL)
    args = parser.parse_args()
    asyncio.run(run(args.golden, args.out, args.evaluator_model))


if __name__ == "__main__":
    main()
