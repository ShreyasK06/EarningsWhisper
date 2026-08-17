import src.retrieval.hybrid as hybrid_mod


def test_hybrid_search_fuses_ranks_via_rrf(monkeypatch):
    def fake_dense(query, k=10, ticker=None, fiscal_quarter=None):
        return [
            {"chunk_id": "A", "text": "a", "section": "s", "ticker": "AAPL", "score": 0.9},
            {"chunk_id": "B", "text": "b", "section": "s", "ticker": "AAPL", "score": 0.8},
        ]

    def fake_bm25(query, k=10, ticker=None, fiscal_quarter=None):
        return [
            {"chunk_id": "B", "text": "b", "section": "s", "ticker": "AAPL", "score": 12.0},
            {"chunk_id": "C", "text": "c", "section": "s", "ticker": "AAPL", "score": 10.0},
        ]

    monkeypatch.setattr(hybrid_mod, "dense_search", fake_dense)
    monkeypatch.setattr(hybrid_mod, "bm25_search", fake_bm25)

    results = hybrid_mod.hybrid_search("query", k=3, pool=10)

    # B is rank 2 in dense AND rank 1 in bm25 -> highest combined RRF score
    assert results[0]["chunk_id"] == "B"
    assert {r["chunk_id"] for r in results} == {"A", "B", "C"}


def test_hybrid_search_passes_filters_through_to_both_retrievers(monkeypatch):
    seen = {}

    def fake_dense(query, k=10, ticker=None, fiscal_quarter=None):
        seen["dense"] = (ticker, fiscal_quarter)
        return []

    def fake_bm25(query, k=10, ticker=None, fiscal_quarter=None):
        seen["bm25"] = (ticker, fiscal_quarter)
        return []

    monkeypatch.setattr(hybrid_mod, "dense_search", fake_dense)
    monkeypatch.setattr(hybrid_mod, "bm25_search", fake_bm25)

    hybrid_mod.hybrid_search("query", ticker="AAPL", fiscal_quarter="2026Q2")

    assert seen["dense"] == ("AAPL", "2026Q2")
    assert seen["bm25"] == ("AAPL", "2026Q2")
