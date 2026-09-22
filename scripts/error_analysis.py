"""Per-question error analysis across dense/hybrid/reranked retrieval.

Tests the hypothesis that Recall@5 regressions from Week 2's hybrid/rerank
additions are concentrated in golden-set questions with multiple relevant
chunks (a rank-compression effect: RRF/reranking promotes one strong chunk
to rank 1, raising MRR/NDCG, while pushing a second relevant chunk out of
the top-5 window) rather than a general retrieval-quality regression.

Usage:
    python scripts/error_analysis.py
"""
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.eval.metrics import recall_at_k
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import reranked_search

K = 5
CONFIGS = [
    ("dense", dense_search),
    ("hybrid", hybrid_search),
    ("reranked", reranked_search),
]
OUT_CSV = ROOT / "data" / "analysis" / "error_analysis.csv"


def load_cases():
    with (config.GOLDEN_DIR / "eval_set.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rank_of_first_hit(retrieved_ids: list[str], relevant_ids) -> int | None:
    relevant = set(relevant_ids)
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant:
            return i
    return None


def analyze():
    cases = load_cases()
    rows = []

    for case in cases:
        relevant_ids = case["relevant_chunk_ids"]
        row = {
            "question": case["question"][:60],
            "n_relevant": len(relevant_ids),
        }
        hits_by_config = {}
        for name, search_fn in CONFIGS:
            hits = search_fn(case["question"], k=K)
            retrieved_ids = [h["chunk_id"] for h in hits]
            hits_by_config[name] = retrieved_ids
            row[f"recall_{name}"] = recall_at_k(retrieved_ids, relevant_ids, K)
            row[f"rank_of_first_hit_{name}"] = rank_of_first_hit(retrieved_ids, relevant_ids)

        row["delta_dense_to_hybrid"] = row["recall_hybrid"] - row["recall_dense"]
        row["_hits_by_config"] = hits_by_config  # not written to CSV, used for the dump below
        rows.append(row)

    rows.sort(key=lambda r: r["delta_dense_to_hybrid"])

    fieldnames = [
        "question", "n_relevant",
        "recall_dense", "recall_hybrid", "recall_reranked",
        "rank_of_first_hit_dense", "rank_of_first_hit_hybrid", "rank_of_first_hit_reranked",
        "delta_dense_to_hybrid",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"{'question':<62} {'n_rel':>5} {'recall_d':>9} {'recall_h':>9} {'recall_r':>9} {'delta_h':>8}")
    for row in rows:
        print(
            f"{row['question']:<62} {row['n_relevant']:>5} "
            f"{row['recall_dense']:>9.2f} {row['recall_hybrid']:>9.2f} {row['recall_reranked']:>9.2f} "
            f"{row['delta_dense_to_hybrid']:>8.2f}"
        )

    print("\n=== grouped by n_relevant (the actual hypothesis test) ===")
    single_gold = [r for r in rows if r["n_relevant"] == 1]
    multi_gold = [r for r in rows if r["n_relevant"] > 1]
    for label, group in [("single-gold (n_relevant=1)", single_gold), ("multi-gold (n_relevant>1)", multi_gold)]:
        if not group:
            print(f"{label}: no questions in this group")
            continue
        print(f"{label} (n={len(group)}):")
        if len(group) < 3:
            print(f"  n={len(group)} is too small for a mean to mean anything -- not printing one.")
            continue
        for name, _ in CONFIGS:
            print(f"  mean recall_{name}: {mean(r[f'recall_{name}'] for r in group):.3f}")

    print("\n=== 3 worst regressions (dense -> hybrid) ===")
    for row in rows[:3]:
        print(f"\nQuestion: {row['question']}")
        print(f"  delta_dense_to_hybrid: {row['delta_dense_to_hybrid']:.2f}")
        print(f"  dense top-5:  {row['_hits_by_config']['dense']}")
        print(f"  hybrid top-5: {row['_hits_by_config']['hybrid']}")

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    return rows


if __name__ == "__main__":
    analyze()
