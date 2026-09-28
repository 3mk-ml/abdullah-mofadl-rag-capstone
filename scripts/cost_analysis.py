from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse


def estimate(
    input_tokens: int,
    output_tokens: int,
    input_per_million: float,
    output_per_million: float,
    rerank_per_query: float = 0.0,
) -> float:
    return (
        input_tokens / 1_000_000 * input_per_million
        + output_tokens / 1_000_000 * output_per_million
        + rerank_per_query
    )


def main() -> None:
    p = argparse.ArgumentParser(description="RAG cost calculator")
    p.add_argument("--input-tokens", type=int, default=4500)
    p.add_argument("--output-tokens", type=int, default=450)
    p.add_argument("--input-price", type=float, default=0.20, help="USD per 1M input tokens")
    p.add_argument("--output-price", type=float, default=1.20, help="USD per 1M output tokens")
    p.add_argument("--rerank", type=float, default=0.0, help="USD reranking cost per query; local reranker is 0")
    p.add_argument("--hosting-monthly", type=float, default=0.0)
    p.add_argument("--queries-per-user", type=int, default=10)
    args = p.parse_args()

    q_cost = estimate(
        args.input_tokens,
        args.output_tokens,
        args.input_price,
        args.output_price,
        args.rerank,
    )
    print(f"Estimated variable cost/query: ${q_cost:.6f}")
    print("\nQuery-volume view:")
    for n in (1_000, 10_000, 100_000):
        total = q_cost * n + args.hosting_monthly
        print(f"{n:>9,} queries/month: ${total:,.2f}")

    print(f"\nUser-volume view (assumption: {args.queries_per_user} queries/user/month):")
    for users in (1_000, 10_000, 100_000):
        queries = users * args.queries_per_user
        total = q_cost * queries + args.hosting_monthly
        print(f"{users:>9,} users -> {queries:>10,} queries/month: ${total:,.2f}")


if __name__ == "__main__":
    main()
