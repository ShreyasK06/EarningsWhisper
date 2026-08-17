# tests/retrieval/test_dense.py
from src.retrieval.dense import dense_search


def test_dense_search_returns_k_results_with_expected_keys():
    results = dense_search("revenue and margin outlook", k=5)
    assert len(results) == 5
    for r in results:
        assert {"chunk_id", "text", "section", "ticker", "score"} <= r.keys()


def test_dense_search_filters_by_ticker_and_fiscal_quarter():
    results = dense_search("revenue outlook", k=5, ticker="AAPL", fiscal_quarter="2026Q2")
    assert len(results) > 0
    for r in results:
        assert r["chunk_id"].startswith("AAPL_2026Q2_")
