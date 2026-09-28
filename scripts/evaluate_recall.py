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

    # When manually verified chunk IDs exist, use them as the authoritative
    # gold evidence. Do not count any arbitrary chunk from the same document.
    if gold_ids:
        return any(r.get("id") in gold_ids for r in retrieved)

    # Backward-compatible fallback for older golden sets without chunk IDs.
    for r in retrieved:
        if gold_source and r.get("source") == gold_source:
            if not must_contain or must_contain in r.get("text", "").lower():
                return True
        if must_contain and must_contain in r.get("text", "").lower():
            return True
    return False


def main(path: Path, k: int = 5, fail_below: float | None = None) -> None:
    golden = load_golden(path)
    if not golden:
        raise SystemExit("Golden set is empty")

    missing_gold = [q.get("id", "?") for q in golden if not q.get("gold_chunk_ids")]
    if missing_gold:
        raise SystemExit(
            "Golden questions are missing verified gold_chunk_ids: "
            + ", ".join(missing_gold)
        )

    hits = 0
    miss_rows: list[dict] = []

    for idx, item in enumerate(golden, start=1):
        results = retrieve(item["question"], top_n=k)
        hit = is_hit(item, results)
        hits += int(hit)

        status = "HIT" if hit else "MISS"
        print(f"[{idx:02d}] {status} {item.get('id', '')} - {item['question']}")

        if not hit:
            gold_ids = item.get("gold_chunk_ids") or []
            retrieved_ids = [r.get("id") for r in results]
            retrieved_sources = [r.get("source") for r in results]
            print(f"     gold IDs: {gold_ids}")
            print(f"     top-{k} IDs: {retrieved_ids}")
            print(f"     top-{k} sources: {retrieved_sources}")
            miss_rows.append(
                {
                    "id": item.get("id"),
                    "question": item["question"],
                    "gold_chunk_ids": gold_ids,
                    "retrieved_chunk_ids": retrieved_ids,
                    "retrieved_sources": retrieved_sources,
                }
            )

    recall = hits / len(golden)
    print(f"\nRecall@{k}: {recall:.2%} ({hits}/{len(golden)})")
    print(f"Misses: {len(miss_rows)}")

    report_path = path.parent / f"recall_at_{k}_report.json"
    report_path.write_text(
        json.dumps(
            {
                "k": k,
                "hits": hits,
                "questions": len(golden),
                "recall": recall,
                "misses": miss_rows,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved report: {report_path}")

    if fail_below is not None and recall < fail_below:
        raise SystemExit(
            f"Recall@{k} {recall:.2%} is below required threshold {fail_below:.0%}"
        )

    if recall >= 0.80:
        print("Target cleared: Recall@5 >= 80%.")
    else:
        print("Target not met yet: inspect misses and tune retrieval.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--golden",
        type=Path,
        default=Path("data/eval/golden_questions.json"),
    )
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--fail-below",
        type=float,
        default=None,
        help="Exit non-zero if recall is below this threshold, e.g. 0.80",
    )
    args = parser.parse_args()
    main(args.golden, args.k, args.fail_below)
