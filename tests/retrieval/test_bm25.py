# tests/retrieval/test_bm25.py
import json

import src.retrieval.bm25 as bm25_mod


def _write_chunks(tmp_path, chunks):
    path = tmp_path / "chunks.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    return path


def test_bm25_search_ranks_exact_term_match_first(tmp_path, monkeypatch):
    chunks = [
        {"chunk_id": "A_1", "ticker": "AAPL", "fiscal_quarter": "2026Q2", "section": "PreparedRemarks",
         "text": "Revenue grew due to strong iPhone demand this quarter."},
        {"chunk_id": "B_1", "ticker": "MSFT", "fiscal_quarter": "2026Q2", "section": "PreparedRemarks",
         "text": "Cloud services showed continued momentum in Azure."},
    ]
    path = _write_chunks(tmp_path, chunks)
    monkeypatch.setattr(bm25_mod, "CHUNKS_PATH", path)
    bm25_mod._bm25 = None
    bm25_mod._chunks = None

    results = bm25_mod.bm25_search("iPhone demand", k=2)

    assert results[0]["chunk_id"] == "A_1"
    assert len(results) == 2


def test_bm25_search_filters_by_ticker_and_fiscal_quarter(tmp_path, monkeypatch):
    chunks = [
        {"chunk_id": "A_1", "ticker": "AAPL", "fiscal_quarter": "2026Q2", "section": "PreparedRemarks",
         "text": "margin pressure from tariffs"},
        {"chunk_id": "A_2", "ticker": "AAPL", "fiscal_quarter": "2026Q3", "section": "PreparedRemarks",
         "text": "margin pressure from tariffs"},
        {"chunk_id": "B_1", "ticker": "MSFT", "fiscal_quarter": "2026Q2", "section": "PreparedRemarks",
         "text": "margin pressure from tariffs"},
    ]
    path = _write_chunks(tmp_path, chunks)
    monkeypatch.setattr(bm25_mod, "CHUNKS_PATH", path)
    bm25_mod._bm25 = None
    bm25_mod._chunks = None

    results = bm25_mod.bm25_search("margin pressure", k=5, ticker="AAPL", fiscal_quarter="2026Q2")

    assert len(results) == 1
    assert results[0]["chunk_id"] == "A_1"
