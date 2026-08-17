"""Hybrid retrieval: dense + BM25 combined via Reciprocal Rank Fusion (RRF).

RRF combines two ranked lists using only rank position, so BM25's unbounded
scores and cosine similarity's [0,1] scores never need normalizing against
each other.
"""
from src.retrieval.bm25 import bm25_search
from src.retrieval.dense import dense_search

RRF_K = 60


def hybrid_search(
    query: str,
    k: int = 5,
    ticker: str | None = None,
    fiscal_quarter: str | None = None,
    pool: int = 50,
) -> list[dict]:
    dense_hits = dense_search(query, k=pool, ticker=ticker, fiscal_quarter=fiscal_quarter)
    sparse_hits = bm25_search(query, k=pool, ticker=ticker, fiscal_quarter=fiscal_quarter)

    scores: dict[str, float] = {}
    meta: dict[str, dict] = {}
    for hits in (dense_hits, sparse_hits):
        for rank, hit in enumerate(hits, start=1):
            chunk_id = hit["chunk_id"]
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (RRF_K + rank)
            meta[chunk_id] = hit

    top = sorted(scores.items(), key=lambda pair: -pair[1])[:k]
    return [{**meta[chunk_id], "score": score} for chunk_id, score in top]
