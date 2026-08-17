import src.generation.generate as generate_mod


def test_gather_context_dedups_across_queries(monkeypatch):
    def fake_hybrid_search(query, k=3, ticker=None, fiscal_quarter=None):
        return [
            {"chunk_id": "SHARED", "text": "t", "section": "s", "ticker": ticker},
            {"chunk_id": f"UNIQUE_{query[:3]}", "text": "t", "section": "s", "ticker": ticker},
        ]

    monkeypatch.setattr(generate_mod, "hybrid_search", fake_hybrid_search)

    context = generate_mod.gather_context("AAPL", k_each=2)
    ids = [c["chunk_id"] for c in context]

    assert ids.count("SHARED") == 1
    assert len(ids) == len(set(ids))


def test_gather_context_passes_ticker_and_fiscal_quarter_through(monkeypatch):
    seen = []

    def fake_hybrid_search(query, k=3, ticker=None, fiscal_quarter=None):
        seen.append((ticker, fiscal_quarter))
        return []

    monkeypatch.setattr(generate_mod, "hybrid_search", fake_hybrid_search)

    generate_mod.gather_context("AAPL", fiscal_quarter="2026Q2")

    assert all(pair == ("AAPL", "2026Q2") for pair in seen)
