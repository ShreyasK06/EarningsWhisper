# src/eval/run_eval.py
"""Retrieval eval harness: Recall@k, Precision@k, MRR, NDCG@k against the
golden eval set, recorded to SQLite. Works with any retriever function of
the shape search_fn(question: str, k: int) -> list[{"chunk_id": ..., ...}].

Usage:
    python scripts/evaluate.py
    python src/eval/run_eval.py   # standalone, dense-only baseline
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.eval.metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k
from src.eval.metrics_db import record_run
from src.retrieval.dense import dense_search

EVAL_SET_PATH = config.GOLDEN_DIR / "eval_set.jsonl"
TOP_K = 5


def load_eval_set():
    with EVAL_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(search_fn=dense_search, k: int = TOP_K, config_hash: str = "dense-bge-small-v1",
             notes: str = "", cases: list[dict] | None = None) -> dict:
    cases = cases if cases is not None else load_eval_set()
    print(f"Loaded {len(cases)} eval examples")

    recalls, precisions, rrs, ndcgs = [], [], [], []
    for case in cases:
        hits = search_fn(case["question"], k=k)
        retrieved_ids = [h["chunk_id"] for h in hits]
        relevant_ids = case["relevant_chunk_ids"]

        r = recall_at_k(retrieved_ids, relevant_ids, k)
        p = precision_at_k(retrieved_ids, relevant_ids, k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        n = ndcg_at_k(retrieved_ids, relevant_ids, k)

        recalls.append(r)
        precisions.append(p)
        rrs.append(rr)
        ndcgs.append(n)

        status = "HIT " if r > 0 else "MISS"
        print(f"  [{status}] recall={r:.2f} rr={rr:.2f}  {case['question'][:70]}")

    metrics = {
        "recall_at_5": mean(recalls),
        "precision_at_5": mean(precisions),
        "mrr": mean(rrs),
        "ndcg_at_5": mean(ndcgs),
    }

    print()
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{config_hash}"
    record_run(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        recall_at_5=metrics["recall_at_5"],
        precision_at_5=metrics["precision_at_5"],
        mrr=metrics["mrr"],
        ndcg_at_5=metrics["ndcg_at_5"],
        faithfulness=None,
        notes=notes,
    )
    print(f"\nRecorded run '{run_id}' to src/eval/metrics.db")
    return metrics


if __name__ == "__main__":
    evaluate(notes="dense-only baseline (bge-small-en-v1.5), week 1")
