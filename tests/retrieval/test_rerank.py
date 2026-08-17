import src.retrieval.rerank as rerank_mod


class FakeCrossEncoder:
    def predict(self, pairs):
        return [1.0 if "target" in text else 0.0 for _, text in pairs]


def test_reranked_search_reorders_by_cross_encoder_score(monkeypatch):
    candidates = [
        {"chunk_id": "A", "text": "irrelevant filler text", "section": "s", "ticker": "AAPL", "score": 0.5},
        {"chunk_id": "B", "text": "this is the target passage", "section": "s", "ticker": "AAPL", "score": 0.1},
    ]
    monkeypatch.setattr(
        rerank_mod, "hybrid_search",
        lambda query, k=10, ticker=None, fiscal_quarter=None: candidates,
    )
    monkeypatch.setattr(rerank_mod, "get_cross_encoder", lambda: FakeCrossEncoder())

    results = rerank_mod.reranked_search("find target", k=2)

    assert results[0]["chunk_id"] == "B"


def test_reranked_search_empty_candidates_returns_empty(monkeypatch):
    monkeypatch.setattr(
        rerank_mod, "hybrid_search",
        lambda query, k=10, ticker=None, fiscal_quarter=None: [],
    )
    assert rerank_mod.reranked_search("anything") == []
