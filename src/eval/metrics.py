"""Pure retrieval-metric functions: Recall@k, Precision@k, MRR (reciprocal
rank), NDCG@k. Operate on lists/sets of chunk_id strings so they can be
tested without a live Qdrant index or embedding model.
"""
import math


def recall_at_k(retrieved_ids: list[str], relevant_ids, k: int) -> float:
    top = set(retrieved_ids[:k])
    relevant = set(relevant_ids)
    return len(top & relevant) / len(relevant) if relevant else 0.0


def precision_at_k(retrieved_ids: list[str], relevant_ids, k: int) -> float:
    top = retrieved_ids[:k]
    relevant = set(relevant_ids)
    return sum(1 for r in top if r in relevant) / k if k else 0.0


def reciprocal_rank(retrieved_ids: list[str], relevant_ids) -> float:
    relevant = set(relevant_ids)
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant:
            return 1.0 / i
    return 0.0


def ndcg_at_k(retrieved_ids: list[str], relevant_ids, k: int) -> float:
    relevant = set(relevant_ids)
    dcg = sum(
        1.0 / math.log2(i + 1)
        for i, r in enumerate(retrieved_ids[:k], start=1)
        if r in relevant
    )
    ideal = sum(1.0 / math.log2(i + 1) for i in range(1, min(len(relevant), k) + 1))
    return dcg / ideal if ideal else 0.0
