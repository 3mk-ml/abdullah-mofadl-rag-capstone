from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json

from src.retrieval import retrieve


def load_golden(path: Path) -> list[dict]:
    data = json.loads(path.read_text(encoding="utf-8"))
    return data["questions"] if isinstance(data, dict) else data


def is_hit(item: dict, retrieved: list[dict]) -> bool:
    gold_ids = set(item.get("gold_chunk_ids") or [])
    gold_source = (item.get("gold_source") or "").strip()
    must_contain = (item.get("must_contain") or "").strip().lower()

    for r in retrieved:
        if gold_ids and r.get("id") in gold_ids:
            return True
        if gold_source and r.get("source") == gold_source:
            if not must_contain or must_contain in r.get("text", "").lower():
                return True
        if must_contain and must_contain in r.get("text", "").lower():
            return True
    return False


def main(path: Path, k: int = 5) -> None:
    golden = load_golden(path)
    if not golden:
        raise SystemExit("Golden set is empty")

    hits = 0
    rows = []
    for idx, item in enumerate(golden, start=1):
        results = retrieve(item["question"], top_n=k)
        hit = is_hit(item, results)
        hits += int(hit)
        rows.append((idx, hit, item["question"], [r.get("source") for r in results]))
        print(f"[{idx:02d}] {'HIT' if hit else 'MISS'} - {item['question']}")

    recall = hits / len(golden)
    print(f"\nRecall@{k}: {recall:.2%} ({hits}/{len(golden)})")
    if recall < 0.80:
        print("Target not met yet: tune chunking/retrieval/reranking and run again.")
    else:
        print("Target cleared: Recall@5 >= 80%.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", type=Path, default=Path("data/eval/golden_questions.json"))
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args()
    main(args.golden, args.k)
