"""CLI: run the Recall@k sweep (k=1,3,5,10) across dense/hybrid/reranked,
logging every (config, k) run to src/eval/metrics.db.

Usage:
    python scripts/eval_sweep.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_eval import sweep
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import reranked_search

if __name__ == "__main__":
    results = sweep(
        {"dense": dense_search, "hybrid": hybrid_search, "reranked": reranked_search},
        k_values=[1, 3, 5, 10],
    )

    print("\n=== sweep summary ===")
    print(f"{'config':<10} {'k':>3} {'recall':>8} {'precision':>10} {'mrr':>8} {'ndcg':>8}")
    for (name, k), m in sorted(results.items(), key=lambda item: (item[0][0], item[0][1])):
        print(f"{name:<10} {k:>3} {m['recall_at_5']:>8.3f} {m['precision_at_5']:>10.3f} "
              f"{m['mrr']:>8.3f} {m['ndcg_at_5']:>8.3f}")
