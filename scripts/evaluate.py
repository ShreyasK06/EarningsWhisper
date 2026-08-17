# scripts/evaluate.py
"""CLI: run retrieval eval across dense, hybrid, and reranked configs;
write results to SQLite.

Usage:
    python scripts/evaluate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_eval import evaluate
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import reranked_search

if __name__ == "__main__":
    print("=== dense ===")
    dense_metrics = evaluate(dense_search, config_hash="dense-bge-small-v1", notes="dense-only baseline")

    print("\n=== hybrid (dense + BM25 RRF) ===")
    hybrid_metrics = evaluate(hybrid_search, config_hash="hybrid-rrf-k60", notes="added BM25 + RRF")

    print("\n=== reranked (hybrid + cross-encoder) ===")
    reranked_metrics = evaluate(reranked_search, config_hash="hybrid-rrf+ce", notes="added cross-encoder rerank")

    print("\n=== comparison ===")
    print(f"{'config':<12} {'recall@5':>10} {'precision@5':>12} {'mrr':>8} {'ndcg@5':>8}")
    for name, m in [("dense", dense_metrics), ("hybrid", hybrid_metrics), ("reranked", reranked_metrics)]:
        print(f"{name:<12} {m['recall_at_5']:>10.3f} {m['precision_at_5']:>12.3f} {m['mrr']:>8.3f} {m['ndcg_at_5']:>8.3f}")
