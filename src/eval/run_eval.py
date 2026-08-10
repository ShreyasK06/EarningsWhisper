"""Dense-only retrieval eval: Recall@k, Precision@k, MRR against the golden
eval set, recorded to SQLite.

Usage:
    python scripts/evaluate.py
    python src/eval/run_eval.py   # standalone
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sentence_transformers import SentenceTransformer

import config
from src.ingestion.embed import COLLECTION_NAME, EMBEDDING_MODEL, QUERY_PREFIX, get_client
from src.eval.metrics_db import record_run

EVAL_SET_PATH = config.GOLDEN_DIR / "eval_set.jsonl"

TOP_K = 5
CONFIG_HASH = "dense-bge-small-v1"


def load_eval_set():
    examples = []
    with EVAL_SET_PATH.open(encoding="utf-8") as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def recall_at_k(retrieved: list[str], relevant: set[str]) -> float:
    hits = len(set(retrieved) & relevant)
    return hits / len(relevant) if relevant else 0.0


def precision_at_k(retrieved: list[str], relevant: set[str], k: int) -> float:
    hits = len(set(retrieved) & relevant)
    return hits / k


def reciprocal_rank(retrieved: list[str], relevant: set[str]) -> float:
    for rank, chunk_id in enumerate(retrieved, start=1):
        if chunk_id in relevant:
            return 1.0 / rank
    return 0.0


def run_eval():
    examples = load_eval_set()
    print(f"Loaded {len(examples)} eval examples")

    model = SentenceTransformer(EMBEDDING_MODEL)
    client = get_client()

    recalls, precisions, rrs = [], [], []

    for ex in examples:
        query_vector = model.encode(QUERY_PREFIX + ex["question"], normalize_embeddings=True).tolist()
        hits = client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=TOP_K,
        ).points
        retrieved_ids = [h.payload["chunk_id"] for h in hits]
        relevant_ids = set(ex["relevant_chunk_ids"])

        r = recall_at_k(retrieved_ids, relevant_ids)
        p = precision_at_k(retrieved_ids, relevant_ids, TOP_K)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)

        recalls.append(r)
        precisions.append(p)
        rrs.append(rr)

        status = "HIT " if r > 0 else "MISS"
        print(f"  [{status}] recall={r:.2f} rr={rr:.2f}  {ex['question'][:70]}")

    recall_at_5 = sum(recalls) / len(recalls)
    precision_at_5 = sum(precisions) / len(precisions)
    mrr = sum(rrs) / len(rrs)

    print()
    print(f"Recall@{TOP_K}:    {recall_at_5:.3f}")
    print(f"Precision@{TOP_K}: {precision_at_5:.3f}")
    print(f"MRR:          {mrr:.3f}")

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    record_run(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=CONFIG_HASH,
        recall_at_5=recall_at_5,
        precision_at_5=precision_at_5,
        mrr=mrr,
        faithfulness=None,
        notes="dense-only baseline (bge-small-en-v1.5), week 1, no BM25/rerank yet",
    )
    print(f"\nRecorded run '{run_id}' to src/eval/metrics.db")


if __name__ == "__main__":
    run_eval()
