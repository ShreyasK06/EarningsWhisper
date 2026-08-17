"""Cross-encoder reranking on top of hybrid retrieval.

Costs extra latency (loads/runs cross-encoder/ms-marco-MiniLM-L-6-v2) --
only worth keeping if src/eval/run_eval.py shows it beats hybrid_search
on Recall@5.
"""
from src.retrieval.hybrid import hybrid_search

_cross_encoder = None


def get_cross_encoder():
    global _cross_encoder
    if _cross_encoder is None:
        from sentence_transformers import CrossEncoder

        _cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
    return _cross_encoder


def reranked_search(
    query: str,
    k: int = 5,
    ticker: str | None = None,
    fiscal_quarter: str | None = None,
    pool: int = 20,
) -> list[dict]:
    candidates = hybrid_search(query, k=pool, ticker=ticker, fiscal_quarter=fiscal_quarter)
    if not candidates:
        return []
    scores = get_cross_encoder().predict([(query, c["text"]) for c in candidates])
    ranked = sorted(zip(candidates, scores), key=lambda pair: -pair[1])
    return [{**c, "score": float(s)} for c, s in ranked[:k]]
