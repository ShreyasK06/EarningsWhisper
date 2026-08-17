from src.eval.metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k


def test_recall_at_k_counts_hits_within_top_k():
    retrieved = ["A", "B", "C", "D", "E"]
    assert recall_at_k(retrieved, ["A", "C"], k=5) == 1.0
    assert recall_at_k(retrieved, ["A", "Z"], k=5) == 0.5
    assert recall_at_k(retrieved, [], k=5) == 0.0


def test_recall_at_k_respects_k_cutoff():
    retrieved = ["A", "B", "C", "D", "E"]
    # "C" is retrieved but outside top-2
    assert recall_at_k(retrieved, ["C"], k=2) == 0.0


def test_precision_at_k():
    retrieved = ["A", "B", "C", "D", "E"]
    assert precision_at_k(retrieved, ["A", "B"], k=5) == 0.4
    assert precision_at_k(retrieved, [], k=5) == 0.0


def test_reciprocal_rank_finds_first_relevant_hit():
    assert reciprocal_rank(["A", "B", "C"], ["B"]) == 0.5
    assert reciprocal_rank(["A", "B", "C"], ["A"]) == 1.0
    assert reciprocal_rank(["A", "B", "C"], ["Z"]) == 0.0


def test_ndcg_at_k_perfect_ranking_is_one():
    # both relevant docs at the top -> ideal DCG achieved
    assert ndcg_at_k(["A", "B", "C"], ["A", "B"], k=3) == 1.0


def test_ndcg_at_k_worse_ranking_scores_lower_than_perfect():
    perfect = ndcg_at_k(["A", "B", "C"], ["A", "B"], k=3)
    worse = ndcg_at_k(["C", "A", "B"], ["A", "B"], k=3)
    assert worse < perfect
