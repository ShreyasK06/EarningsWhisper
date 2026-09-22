# Earnings Whisper — Week 2 (Retrieval Comparison, Generation, Faithfulness, Backtest) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Earnings Whisper build guide's remaining steps (6/9/10, 11–16) on top of the already-completed Week 1 pipeline (fetch → parse → chunk → embed → dense-only eval, Recall@5 = 0.775) — modularize retrieval into dense/BM25/hybrid/reranked, wire an eval comparison across them, add cited structured-signal generation, a faithfulness judge, a lookahead-safe backtest, and update the README with results.

**Architecture:** `src/retrieval/{dense,bm25,hybrid,rerank}.py` expose a common `search(query, k, ticker=None, fiscal_quarter=None) -> list[dict]` shape reused by `src/eval/run_eval.py` (comparison harness) and `src/generation/generate.py` (context gathering for Claude tool-use signal generation). `src/eval/faithfulness.py` grades generated reasoning against retrieved excerpts. `src/backtest/score_signals.py` scores signals against realized yfinance forward returns, entering at the next session's open when a filing lands after market close (derived from each filing's exact SEC `ACCEPTANCE-DATETIME`, not a blanket flag).

**Tech Stack:** Python, qdrant-client (local mode), sentence-transformers (bge-small-en-v1.5 + cross-encoder/ms-marco-MiniLM-L-6-v2), rank-bm25, anthropic SDK (`claude-opus-5`, tool use), pydantic v2, yfinance, pytest.

## Global Constraints

- Python-only, no Docker (matches the existing Week 1 restructure — Qdrant runs in local on-disk mode via `QdrantClient(path=...)`).
- Chunk schema is fixed from Week 1 and MUST NOT change: `{chunk_id, ticker, fiscal_quarter, filing_date, filed_datetime, accession, section, speaker, text}`. `chunk_id` format is `{TICKER}_{fiscal_quarter}_{section}_{seq:03d}` (e.g. `AAPL_2026Q2_PreparedRemarks_000`).
- Qdrant collection name is `earnings_chunks` (from `src/ingestion/embed.py`); payload indexes already exist on `ticker` and `fiscal_quarter` — reuse both for filtering, don't add new ones.
- Every module that needs `config` or `src.*` from a script invoked directly (`python -m ...` or `python src/...`) must keep the existing `ROOT = Path(__file__).resolve().parents[N]; if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))` pattern already used throughout the codebase — follow it exactly, don't introduce a different import mechanism.
- Model ID for all Claude API calls: `claude-opus-5` (current default per house policy; this project's actual signal/judge call volume is ~10–20 requests per run, so cost is not a constraint). Do not use `budget_tokens` or non-adaptive `thinking` config — omit `thinking` entirely (adaptive is the default) and rely on `tool_choice` to force structured tool output.
- `ANTHROPIC_API_KEY` is currently **not set** in `.env` on this machine (only `.env.example` exists). Tasks 8–9 (schema, generation code) and 10 (faithfulness code) can be written and unit-tested without it; Tasks 9's `generate_signal` live call, 10's `judge` live call, and 12's CLI scripts require the user to populate `.env` with a real key before they can be *run* end-to-end — flag this at that point, don't block writing the code on it.
- New source files get a matching `tests/<mirror-path>/test_*.py`. Add `tests/conftest.py` once (Task 1) to put `ROOT` on `sys.path` for every test — do not repeat that boilerplate per test file.
- Every task's commit uses `git add <specific files>` — never `git add -A` (there are gitignored `qdrant_data/`, `.env`, and `data/raw/` directories in this repo that must never be staged).

---

### Task 1: Test scaffolding + pure retrieval-metrics module

**Files:**
- Create: `tests/conftest.py`
- Create: `src/eval/metrics.py`
- Test: `tests/eval/test_metrics.py`

**Interfaces:**
- Produces: `recall_at_k(retrieved_ids: list[str], relevant_ids, k: int) -> float`, `precision_at_k(retrieved_ids, relevant_ids, k) -> float`, `reciprocal_rank(retrieved_ids, relevant_ids) -> float`, `ndcg_at_k(retrieved_ids, relevant_ids, k) -> float` — pure functions, no I/O. These replace the metric math currently inlined in `src/eval/run_eval.py` (Task 7 rewires it to import from here and adds `ndcg_at_k`, which doesn't exist yet).

- [ ] **Step 1: Create the shared test-path conftest**

```python
# tests/conftest.py
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
```

- [ ] **Step 2: Write the failing test for the metrics module**

```python
# tests/eval/test_metrics.py
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
```

- [ ] **Step 3: Run tests to verify they fail with an import error**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.metrics'`

- [ ] **Step 4: Implement the metrics module**

```python
# src/eval/metrics.py
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
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/eval/test_metrics.py -v`
Expected: PASS (6 tests)

- [ ] **Step 6: Commit**

```bash
git add tests/conftest.py src/eval/metrics.py tests/eval/test_metrics.py
git commit -m "test: add pure retrieval-metrics module (recall/precision/mrr/ndcg)"
```

---

### Task 2: Cache the embedding model and Qdrant client as singletons

**Files:**
- Modify: `src/ingestion/embed.py`
- Test: `tests/ingestion/test_embed_caching.py`

**Interfaces:**
- Consumes: existing `config.QDRANT_HOST`, `config.QDRANT_PATH`, `config.EMBED_MODEL`.
- Produces: `get_model() -> SentenceTransformer` (new — module-level singleton), `get_client() -> QdrantClient` (existing signature, now cached). `src/retrieval/dense.py` (Task 3) imports both. Without caching, every dense-retrieval call in Task 3–12 would reopen the local Qdrant storage directory and reload the embedding model, which is slow and — because Qdrant's on-disk mode holds a lock on the storage folder — risks a "storage folder already accessed by another instance" error if two call sites in the same process ever raced.

- [ ] **Step 1: Write the failing test**

```python
# tests/ingestion/test_embed_caching.py
from src.ingestion import embed as embed_mod


def test_get_model_returns_singleton(monkeypatch):
    embed_mod._model = None
    calls = []

    class FakeModel:
        pass

    def fake_ctor(name):
        calls.append(name)
        return FakeModel()

    monkeypatch.setattr(embed_mod, "SentenceTransformer", fake_ctor)

    m1 = embed_mod.get_model()
    m2 = embed_mod.get_model()

    assert m1 is m2
    assert len(calls) == 1


def test_get_client_returns_singleton(monkeypatch, tmp_path):
    embed_mod._client = None
    monkeypatch.setattr(embed_mod.config, "QDRANT_HOST", "")
    monkeypatch.setattr(embed_mod.config, "QDRANT_PATH", str(tmp_path / "qdrant_test"))

    calls = []

    class FakeClient:
        pass

    def fake_ctor(path):
        calls.append(path)
        return FakeClient()

    monkeypatch.setattr(embed_mod, "QdrantClient", fake_ctor)

    c1 = embed_mod.get_client()
    c2 = embed_mod.get_client()

    assert c1 is c2
    assert len(calls) == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/ingestion/test_embed_caching.py -v`
Expected: FAIL — `AttributeError: module 'src.ingestion.embed' has no attribute 'get_model'` (and `get_client` isn't cached yet, so the second assertion in `test_get_client_returns_singleton` fails too)

- [ ] **Step 3: Add caching to `src/ingestion/embed.py`**

Replace the existing `get_client` function and add `get_model`, and add the two module-level cache variables near the top (right after the existing constants):

```python
# add near the top of src/ingestion/embed.py, after EMBEDDING_DIM / QUERY_PREFIX
_model = None
_client = None


def get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)
    return _model


def get_client() -> QdrantClient:
    global _client
    if _client is None:
        if config.QDRANT_HOST:
            _client = QdrantClient(host=config.QDRANT_HOST, port=config.QDRANT_PORT)
        else:
            Path(config.QDRANT_PATH).mkdir(parents=True, exist_ok=True)
            _client = QdrantClient(path=config.QDRANT_PATH)
    return _client
```

Delete the old (uncached) `get_client` body it replaces. Also update `embed_and_load()` to reuse the cached model instead of constructing its own:

```python
def embed_and_load():
    chunks = load_chunks()
    print(f"Loaded {len(chunks)} chunks from {CHUNKS_PATH}")

    model = get_model()
    vectors = embed_documents(model, [c["text"] for c in chunks])
    ...
```

(leave the rest of `embed_and_load` unchanged — only the `model = SentenceTransformer(EMBEDDING_MODEL)` line becomes `model = get_model()`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/ingestion/test_embed_caching.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/ingestion/embed.py tests/ingestion/test_embed_caching.py
git commit -m "refactor: cache embedding model and Qdrant client as singletons"
```

---

### Task 3: Dense retrieval module

**Files:**
- Create: `src/retrieval/__init__.py` (empty)
- Create: `src/retrieval/dense.py`
- Test: `tests/retrieval/test_dense.py`

**Interfaces:**
- Consumes: `src.ingestion.embed.{COLLECTION_NAME, QUERY_PREFIX, get_client, get_model}` (Task 2).
- Produces: `dense_search(query: str, k: int = 10, ticker: str | None = None, fiscal_quarter: str | None = None) -> list[dict]`, each dict `{chunk_id, text, section, ticker, score}`. Consumed by `src/retrieval/hybrid.py` (Task 5) and `src/eval/run_eval.py` (Task 7).

This task's test needs the live Qdrant index built in Week 1 (`./qdrant_data`, already present in this repo) — it is a smoke/integration test, not a pure unit test, since dense search has no meaningful behavior to fake.

- [ ] **Step 1: Write the test against the live index**

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest tests/retrieval/test_dense.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval'`

- [ ] **Step 3: Implement `src/retrieval/dense.py`**

```python
# src/retrieval/__init__.py
# (empty)
```

```python
# src/retrieval/dense.py
"""Dense (embedding) retrieval against the Qdrant `earnings_chunks` collection."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.embed import COLLECTION_NAME, QUERY_PREFIX, get_client, get_model


def dense_search(
    query: str,
    k: int = 10,
    ticker: str | None = None,
    fiscal_quarter: str | None = None,
) -> list[dict]:
    model = get_model()
    client = get_client()
    query_vector = model.encode(QUERY_PREFIX + query, normalize_embeddings=True).tolist()

    query_filter = None
    if ticker or fiscal_quarter:
        from qdrant_client.http import models as qmodels

        conditions = []
        if ticker:
            conditions.append(qmodels.FieldCondition(key="ticker", match=qmodels.MatchValue(value=ticker)))
        if fiscal_quarter:
            conditions.append(
                qmodels.FieldCondition(key="fiscal_quarter", match=qmodels.MatchValue(value=fiscal_quarter))
            )
        query_filter = qmodels.Filter(must=conditions)

    hits = client.query_points(
        collection_name=COLLECTION_NAME,
        query=query_vector,
        limit=k,
        query_filter=query_filter,
    ).points

    return [
        {
            "chunk_id": h.payload["chunk_id"],
            "text": h.payload["text"],
            "section": h.payload["section"],
            "ticker": h.payload["ticker"],
            "score": h.score,
        }
        for h in hits
    ]


if __name__ == "__main__":
    for r in dense_search("What did management say about margins?", k=5):
        print(f"{r['score']:.3f}  {r['chunk_id']}\n  {r['text'][:150]}...\n")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest tests/retrieval/test_dense.py -v`
Expected: PASS (2 tests). If it fails with a Qdrant "collection not found" error, run `python scripts/ingest_all.py` first to rebuild the index — this test depends on the Week 1 index already existing.

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/__init__.py src/retrieval/dense.py tests/retrieval/test_dense.py
git commit -m "feat: extract dense retrieval into reusable src/retrieval/dense.py"
```

---

### Task 4: BM25 sparse retrieval module

**Files:**
- Create: `src/retrieval/bm25.py`
- Test: `tests/retrieval/test_bm25.py`

**Interfaces:**
- Consumes: `config.PROCESSED_DIR`.
- Produces: `bm25_search(query: str, k: int = 10, ticker: str | None = None, fiscal_quarter: str | None = None) -> list[dict]`, same dict shape as `dense_search`. Consumed by `src/retrieval/hybrid.py` (Task 5).

This is a pure, file-backed module (no model, no network) — a genuine TDD candidate with a fixture corpus instead of the real 193-chunk one.

- [ ] **Step 1: Write the failing tests**

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/retrieval/test_bm25.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval.bm25'`

- [ ] **Step 3: Implement `src/retrieval/bm25.py`**

```python
# src/retrieval/bm25.py
"""BM25 (sparse/lexical) retrieval over data/processed/chunks.jsonl.

Complements dense retrieval for exact numeric and entity matches (e.g.
"$111.2 billion", "iPhone 17") that embeddings alone can miss.
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from rank_bm25 import BM25Okapi

import config

CHUNKS_PATH = config.PROCESSED_DIR / "chunks.jsonl"

_bm25 = None
_chunks = None


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _load():
    global _bm25, _chunks
    if _bm25 is None:
        _chunks = [json.loads(line) for line in open(CHUNKS_PATH, encoding="utf-8")]
        _bm25 = BM25Okapi([_tokenize(c["text"]) for c in _chunks])
    return _bm25, _chunks


def bm25_search(
    query: str,
    k: int = 10,
    ticker: str | None = None,
    fiscal_quarter: str | None = None,
) -> list[dict]:
    bm25, chunks = _load()
    scores = bm25.get_scores(_tokenize(query))
    ranked = sorted(zip(chunks, scores), key=lambda pair: -pair[1])
    if ticker:
        ranked = [pair for pair in ranked if pair[0]["ticker"] == ticker]
    if fiscal_quarter:
        ranked = [pair for pair in ranked if pair[0]["fiscal_quarter"] == fiscal_quarter]
    return [
        {
            "chunk_id": c["chunk_id"],
            "text": c["text"],
            "section": c["section"],
            "ticker": c["ticker"],
            "score": float(s),
        }
        for c, s in ranked[:k]
    ]
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/retrieval/test_bm25.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/bm25.py tests/retrieval/test_bm25.py
git commit -m "feat: add BM25 sparse retrieval module"
```

---

### Task 5: Hybrid retrieval (Reciprocal Rank Fusion)

**Files:**
- Create: `src/retrieval/hybrid.py`
- Test: `tests/retrieval/test_hybrid.py`

**Interfaces:**
- Consumes: `dense_search` (Task 3), `bm25_search` (Task 4) — both imported by name so tests can monkeypatch them on the `hybrid` module.
- Produces: `hybrid_search(query: str, k: int = 5, ticker: str | None = None, fiscal_quarter: str | None = None, pool: int = 50) -> list[dict]`. Consumed by `src/retrieval/rerank.py` (Task 6), `src/eval/run_eval.py` (Task 7), and `src/generation/generate.py` (Task 9).

- [ ] **Step 1: Write the failing test**

```python
# tests/retrieval/test_hybrid.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/retrieval/test_hybrid.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval.hybrid'`

- [ ] **Step 3: Implement `src/retrieval/hybrid.py`**

```python
# src/retrieval/hybrid.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/retrieval/test_hybrid.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/hybrid.py tests/retrieval/test_hybrid.py
git commit -m "feat: add hybrid dense+BM25 retrieval via reciprocal rank fusion"
```

---

### Task 6: Cross-encoder reranking

**Files:**
- Create: `src/retrieval/rerank.py`
- Test: `tests/retrieval/test_rerank.py`

**Interfaces:**
- Consumes: `hybrid_search` (Task 5), imported by name for monkeypatching.
- Produces: `reranked_search(query: str, k: int = 5, ticker: str | None = None, fiscal_quarter: str | None = None, pool: int = 20) -> list[dict]`, `get_cross_encoder()`. Consumed by `src/eval/run_eval.py` (Task 7) only — kept only if the comparison in Task 7 shows it beats hybrid.

- [ ] **Step 1: Write the failing tests (fake cross-encoder, no model download)**

```python
# tests/retrieval/test_rerank.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/retrieval/test_rerank.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.retrieval.rerank'`

- [ ] **Step 3: Implement `src/retrieval/rerank.py`**

```python
# src/retrieval/rerank.py
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/retrieval/test_rerank.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/retrieval/rerank.py tests/retrieval/test_rerank.py
git commit -m "feat: add cross-encoder reranking on top of hybrid retrieval"
```

---

### Task 7: Rewire the eval harness to compare dense vs hybrid vs reranked

**Files:**
- Modify: `src/eval/metrics_db.py`
- Modify: `src/eval/run_eval.py`
- Modify: `scripts/evaluate.py`
- Test: `tests/eval/test_run_eval.py`

**Interfaces:**
- Consumes: `src.eval.metrics.{recall_at_k, precision_at_k, reciprocal_rank, ndcg_at_k}` (Task 1), `src.retrieval.dense.dense_search` (Task 3), `src.retrieval.hybrid.hybrid_search` (Task 5), `src.retrieval.rerank.reranked_search` (Task 6).
- Produces: `evaluate(search_fn=dense_search, k=5, config_hash="dense-bge-small-v1", notes="", cases=None) -> dict` with keys `recall_at_5, precision_at_5, mrr, ndcg_at_5`, replacing the old `run_eval()` entry point.

- [ ] **Step 1: Add the `ndcg_at_5` column to the metrics DB schema**

Replace the `SCHEMA` constant and `get_connection` / `record_run` in `src/eval/metrics_db.py`:

```python
SCHEMA = """
CREATE TABLE IF NOT EXISTS eval_runs (
  run_id TEXT PRIMARY KEY,
  timestamp TEXT,
  config_hash TEXT,
  recall_at_5 REAL,
  precision_at_5 REAL,
  mrr REAL,
  ndcg_at_5 REAL,
  faithfulness REAL,
  notes TEXT
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(eval_runs)")}
    if "ndcg_at_5" not in existing_cols:
        conn.execute("ALTER TABLE eval_runs ADD COLUMN ndcg_at_5 REAL")
    conn.commit()
    return conn


def record_run(run_id: str, timestamp: str, config_hash: str, recall_at_5: float,
                precision_at_5: float, mrr: float, ndcg_at_5: float | None,
                faithfulness: float | None, notes: str):
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs
                (run_id, timestamp, config_hash, recall_at_5, precision_at_5, mrr, ndcg_at_5, faithfulness, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, timestamp, config_hash, recall_at_5, precision_at_5, mrr, ndcg_at_5, faithfulness, notes),
        )
    conn.close()
```

(the `all_runs()` function and `DB_PATH` constant are unchanged — leave them as-is.)

- [ ] **Step 2: Write the failing test for the rewritten `evaluate()`**

```python
# tests/eval/test_run_eval.py
import src.eval.run_eval as run_eval_mod


def test_evaluate_computes_mean_metrics_and_records_run(monkeypatch):
    cases = [
        {"question": "q1", "relevant_chunk_ids": ["A"]},
        {"question": "q2", "relevant_chunk_ids": ["Z"]},
    ]

    def fake_search(question, k=5):
        return [{"chunk_id": cid} for cid in ["A", "B", "C", "D", "E"]]

    recorded = {}

    def fake_record_run(**kwargs):
        recorded.update(kwargs)

    monkeypatch.setattr(run_eval_mod, "record_run", fake_record_run)

    metrics = run_eval_mod.evaluate(search_fn=fake_search, k=5, config_hash="test-cfg", cases=cases)

    # q1: "A" is rank 1 -> recall=1, rr=1; q2: "Z" not retrieved -> recall=0, rr=0
    assert metrics["recall_at_5"] == 0.5
    assert metrics["mrr"] == 0.5
    assert recorded["config_hash"] == "test-cfg"
    assert recorded["recall_at_5"] == 0.5
```

- [ ] **Step 3: Run test to verify it fails**

Run: `python -m pytest tests/eval/test_run_eval.py -v`
Expected: FAIL — `AttributeError: module 'src.eval.run_eval' has no attribute 'evaluate'` (the module currently only exposes `run_eval`)

- [ ] **Step 4: Rewrite `src/eval/run_eval.py`**

```python
# src/eval/run_eval.py
"""Retrieval eval harness: Recall@k, Precision@k, MRR, NDCG@k against the
golden eval set, recorded to SQLite. Works with any retriever function of
the shape search_fn(question: str, k: int) -> list[{"chunk_id": ..., ...}].

Usage:
    python scripts/evaluate.py
    python src/eval/run_eval.py   # standalone, dense-only baseline
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.eval.metrics import ndcg_at_k, precision_at_k, reciprocal_rank, recall_at_k
from src.eval.metrics_db import record_run
from src.retrieval.dense import dense_search

EVAL_SET_PATH = config.GOLDEN_DIR / "eval_set.jsonl"
TOP_K = 5


def load_eval_set():
    with EVAL_SET_PATH.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def evaluate(search_fn=dense_search, k: int = TOP_K, config_hash: str = "dense-bge-small-v1",
             notes: str = "", cases: list[dict] | None = None) -> dict:
    cases = cases if cases is not None else load_eval_set()
    print(f"Loaded {len(cases)} eval examples")

    recalls, precisions, rrs, ndcgs = [], [], [], []
    for case in cases:
        hits = search_fn(case["question"], k=k)
        retrieved_ids = [h["chunk_id"] for h in hits]
        relevant_ids = case["relevant_chunk_ids"]

        r = recall_at_k(retrieved_ids, relevant_ids, k)
        p = precision_at_k(retrieved_ids, relevant_ids, k)
        rr = reciprocal_rank(retrieved_ids, relevant_ids)
        n = ndcg_at_k(retrieved_ids, relevant_ids, k)

        recalls.append(r)
        precisions.append(p)
        rrs.append(rr)
        ndcgs.append(n)

        status = "HIT " if r > 0 else "MISS"
        print(f"  [{status}] recall={r:.2f} rr={rr:.2f}  {case['question'][:70]}")

    metrics = {
        "recall_at_5": mean(recalls),
        "precision_at_5": mean(precisions),
        "mrr": mean(rrs),
        "ndcg_at_5": mean(ndcgs),
    }

    print()
    for name, value in metrics.items():
        print(f"{name}: {value:.3f}")

    run_id = f"run_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}_{config_hash}"
    record_run(
        run_id=run_id,
        timestamp=datetime.now(timezone.utc).isoformat(),
        config_hash=config_hash,
        recall_at_5=metrics["recall_at_5"],
        precision_at_5=metrics["precision_at_5"],
        mrr=metrics["mrr"],
        ndcg_at_5=metrics["ndcg_at_5"],
        faithfulness=None,
        notes=notes,
    )
    print(f"\nRecorded run '{run_id}' to src/eval/metrics.db")
    return metrics


if __name__ == "__main__":
    evaluate(notes="dense-only baseline (bge-small-en-v1.5), week 1")
```

- [ ] **Step 5: Run test to verify it passes**

Run: `python -m pytest tests/eval/test_run_eval.py -v`
Expected: PASS (1 test)

- [ ] **Step 6: Rewrite `scripts/evaluate.py` to compare all three configs**

```python
# scripts/evaluate.py
"""CLI: run retrieval eval across dense, hybrid, and reranked configs;
write results to SQLite.

Usage:
    python scripts/evaluate.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_eval import evaluate
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import reranked_search

if __name__ == "__main__":
    print("=== dense ===")
    dense_metrics = evaluate(dense_search, config_hash="dense-bge-small-v1", notes="dense-only baseline")

    print("\n=== hybrid (dense + BM25 RRF) ===")
    hybrid_metrics = evaluate(hybrid_search, config_hash="hybrid-rrf-k60", notes="added BM25 + RRF")

    print("\n=== reranked (hybrid + cross-encoder) ===")
    reranked_metrics = evaluate(reranked_search, config_hash="hybrid-rrf+ce", notes="added cross-encoder rerank")

    print("\n=== comparison ===")
    print(f"{'config':<12} {'recall@5':>10} {'precision@5':>12} {'mrr':>8} {'ndcg@5':>8}")
    for name, m in [("dense", dense_metrics), ("hybrid", hybrid_metrics), ("reranked", reranked_metrics)]:
        print(f"{name:<12} {m['recall_at_5']:>10.3f} {m['precision_at_5']:>12.3f} {m['mrr']:>8.3f} {m['ndcg_at_5']:>8.3f}")
```

- [ ] **Step 7: Run the full comparison against the live index and record the numbers**

Run: `python scripts/evaluate.py`
Expected: prints Recall@5/Precision@5/MRR/NDCG@5 for all three configs and a comparison table. **Write down these three rows — Task 13 (README) needs them.** Per the guide's rule: each addition (BM25, then reranking) stays only if it measurably improves Recall@5 over the previous config; if hybrid doesn't beat dense, or reranked doesn't beat hybrid, note that honestly instead of hiding it.

- [ ] **Step 8: Commit**

```bash
git add src/eval/metrics_db.py src/eval/run_eval.py scripts/evaluate.py tests/eval/test_run_eval.py
git commit -m "feat: compare dense/hybrid/reranked retrieval, add NDCG@5"
```

---

### Task 8: Signal schema

**Files:**
- Create: `src/generation/__init__.py` (empty)
- Create: `src/generation/schema.py`
- Test: `tests/generation/test_schema.py`

**Interfaces:**
- Produces: `SupportingClaim(text, chunk_id, section)`, `Signal(ticker, signal, confidence, reasoning, supporting_claims)` — pydantic v2 models. Consumed by `src/generation/generate.py` (Task 9).

- [ ] **Step 1: Write the failing tests**

```python
# tests/generation/test_schema.py
import pytest
from pydantic import ValidationError

from src.generation.schema import Signal


def _valid_claim():
    return {"text": "revenue grew 17%", "chunk_id": "AAPL_2026Q2_PreparedRemarks_000", "section": "PreparedRemarks"}


def test_signal_requires_at_least_one_supporting_claim():
    with pytest.raises(ValidationError):
        Signal(ticker="AAPL", signal="bullish", confidence=0.8, reasoning="strong quarter", supporting_claims=[])


def test_signal_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        Signal(
            ticker="AAPL", signal="bullish", confidence=1.5, reasoning="strong quarter",
            supporting_claims=[_valid_claim()],
        )


def test_signal_rejects_invalid_direction():
    with pytest.raises(ValidationError):
        Signal(
            ticker="AAPL", signal="up", confidence=0.5, reasoning="x",
            supporting_claims=[_valid_claim()],
        )


def test_signal_accepts_valid_input():
    sig = Signal(
        ticker="AAPL", signal="neutral", confidence=0.4, reasoning="mixed signals",
        supporting_claims=[_valid_claim()],
    )
    assert sig.signal == "neutral"
    assert len(sig.supporting_claims) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/generation/test_schema.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.generation'`

- [ ] **Step 3: Implement the schema**

```python
# src/generation/__init__.py
# (empty)
```

```python
# src/generation/schema.py
"""Structured trading-signal output schema.

`supporting_claims` requires at least one entry -- an uncited signal is a
pydantic ValidationError, not a silent hallucination.
"""
from typing import Literal

from pydantic import BaseModel, Field


class SupportingClaim(BaseModel):
    text: str = Field(description="Verbatim span from the source chunk")
    chunk_id: str
    section: str


class Signal(BaseModel):
    ticker: str
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    supporting_claims: list[SupportingClaim] = Field(min_length=1)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/generation/test_schema.py -v`
Expected: PASS (4 tests)

- [ ] **Step 5: Commit**

```bash
git add src/generation/__init__.py src/generation/schema.py tests/generation/test_schema.py
git commit -m "feat: add pydantic Signal schema with required citations"
```

---

### Task 9: Generation prompts + Claude tool-use call

**Files:**
- Create: `src/generation/prompts.py`
- Create: `src/generation/generate.py`
- Test: `tests/generation/test_prompts.py`
- Test: `tests/generation/test_generate.py`

**Interfaces:**
- Consumes: `src.generation.schema.Signal` (Task 8), `src.retrieval.hybrid.hybrid_search` (Task 5), `config.ANTHROPIC_API_KEY`.
- Produces: `build_user_prompt(ticker, chunks) -> str`, `SYSTEM: str`, `gather_context(ticker, fiscal_quarter=None, k_each=3) -> list[dict]`, `generate_signal(ticker, fiscal_quarter=None, retries=1) -> tuple[Signal, list[dict]]`. Consumed by `scripts/run_signal.py` and `scripts/backtest.py` (Task 12).

`generate_signal`'s live Anthropic call cannot be exercised by an automated test in this environment (no `ANTHROPIC_API_KEY` set) — only `build_user_prompt` and `gather_context`'s dedup logic get unit tests here. `generate_signal` gets verified live in Task 12 once the user supplies a key.

- [ ] **Step 1: Write the failing tests**

```python
# tests/generation/test_prompts.py
from src.generation.prompts import build_user_prompt


def test_build_user_prompt_includes_chunk_ids_and_sections():
    chunks = [{"chunk_id": "AAPL_1", "section": "PreparedRemarks", "text": "revenue grew"}]
    prompt = build_user_prompt("AAPL", chunks)
    assert "AAPL_1" in prompt
    assert "PreparedRemarks" in prompt
    assert "revenue grew" in prompt
    assert "AAPL" in prompt
```

```python
# tests/generation/test_generate.py
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/generation/test_prompts.py tests/generation/test_generate.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.generation.prompts'` (and `.generate`)

- [ ] **Step 3: Implement `src/generation/prompts.py`**

```python
# src/generation/prompts.py
"""System and user prompt construction for signal generation."""

SYSTEM = """You are a financial analyst reading earnings press-release excerpts.

Rules:
- Use ONLY the provided excerpts. Never use outside knowledge about the company.
- Every claim in your reasoning must be traceable to a provided chunk_id.
- Confidence calibration: 0.8+ only when multiple independent excerpts support
  the same direction. Use 0.3-0.5 when evidence is mixed or hedged.
- "neutral" is a valid and often correct answer. Do not manufacture a signal.
"""


def build_user_prompt(ticker: str, chunks: list[dict]) -> str:
    blocks = "\n\n".join(
        f"[chunk_id: {c['chunk_id']} | section: {c['section']}]\n{c['text']}"
        for c in chunks
    )
    return f"""Analyze these excerpts from {ticker}'s recent earnings release and emit a trading signal.

EXCERPTS:
{blocks}

Consider: forward-looking language, hedging, changes in tone, margin/demand commentary."""
```

- [ ] **Step 4: Implement `src/generation/generate.py`**

```python
# src/generation/generate.py
"""Generate a structured trading signal for a ticker via Claude tool use,
grounded in retrieved earnings-release excerpts.

Requires ANTHROPIC_API_KEY in .env.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import Anthropic
from pydantic import ValidationError

import config
from src.generation.prompts import SYSTEM, build_user_prompt
from src.generation.schema import Signal
from src.retrieval.hybrid import hybrid_search

MODEL = "claude-opus-5"

TOOL = {
    "name": "emit_signal",
    "description": "Emit a structured trading signal with citations.",
    "input_schema": Signal.model_json_schema(),
}

CONTEXT_QUERIES = [
    "outlook and guidance for coming quarters",
    "risks and headwinds to revenue",
    "margin pressure and cost trends",
    "demand trends and customer commentary",
]


def gather_context(ticker: str, fiscal_quarter: str | None = None, k_each: int = 3) -> list[dict]:
    seen: set[str] = set()
    context: list[dict] = []
    for query in CONTEXT_QUERIES:
        for hit in hybrid_search(query, k=k_each, ticker=ticker, fiscal_quarter=fiscal_quarter):
            if hit["chunk_id"] not in seen:
                seen.add(hit["chunk_id"])
                context.append(hit)
    return context


def generate_signal(ticker: str, fiscal_quarter: str | None = None, retries: int = 1):
    chunks = gather_context(ticker, fiscal_quarter=fiscal_quarter)
    label = f"{ticker} ({fiscal_quarter})" if fiscal_quarter else ticker
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": build_user_prompt(label, chunks)}]

    for attempt in range(retries + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": "emit_signal"},
            messages=messages,
        )
        tool_use = next(b for b in response.content if b.type == "tool_use")
        try:
            return Signal(**tool_use.input), chunks
        except ValidationError as e:
            if attempt == retries:
                raise
            messages += [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Validation failed: {e}. Emit again, corrected.",
                            "is_error": True,
                        }
                    ],
                },
            ]


if __name__ == "__main__":
    import json

    signal, _ = generate_signal("AAPL")
    print(json.dumps(signal.model_dump(), indent=2))
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `python -m pytest tests/generation/test_prompts.py tests/generation/test_generate.py -v`
Expected: PASS (3 tests)

- [ ] **Step 6: Commit**

```bash
git add src/generation/prompts.py src/generation/generate.py tests/generation/test_prompts.py tests/generation/test_generate.py
git commit -m "feat: add Claude tool-use signal generation with citation retries"
```

---

### Task 10: Faithfulness judge

**Files:**
- Create: `src/eval/faithfulness.py`
- Test: `tests/eval/test_faithfulness.py`

**Interfaces:**
- Consumes: `config.ANTHROPIC_API_KEY`.
- Produces: `score_claims(claims: list[dict]) -> float` (pure, tested), `judge(reasoning: str, chunks: list[dict]) -> tuple[float, list[dict]]` (live API call, not unit-tested here — same reasoning as Task 9). Consumed by `scripts/run_signal.py` (Task 12).

- [ ] **Step 1: Write the failing tests for the pure scoring helper**

```python
# tests/eval/test_faithfulness.py
from src.eval.faithfulness import score_claims


def test_score_claims_computes_supported_fraction():
    claims = [
        {"claim": "a", "label": "supported"},
        {"claim": "b", "label": "unsupported"},
        {"claim": "c", "label": "supported"},
        {"claim": "d", "label": "contradicted"},
    ]
    assert score_claims(claims) == 0.5


def test_score_claims_empty_list_is_zero():
    assert score_claims([]) == 0.0


def test_score_claims_all_supported_is_one():
    claims = [{"claim": "a", "label": "supported"}, {"claim": "b", "label": "supported"}]
    assert score_claims(claims) == 1.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/eval/test_faithfulness.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.eval.faithfulness'`

- [ ] **Step 3: Implement `src/eval/faithfulness.py`**

```python
# src/eval/faithfulness.py
"""LLM-judge faithfulness check: does the generated reasoning stay grounded
in the retrieved excerpts?

The judge itself must be spot-checked by hand against ~10 examples before
its score is trusted (compare its labels to your own reading) -- an
unvalidated judge is a number you can't defend.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import Anthropic

import config

MODEL = "claude-opus-5"

JUDGE_SYSTEM = """You are grading whether an analyst's reasoning is supported by source excerpts.

Break the reasoning into individual factual claims. For each, label:
- "supported": directly stated or clearly implied by an excerpt
- "unsupported": not found in any excerpt
- "contradicted": an excerpt says otherwise

Return ONLY JSON: {"claims":[{"claim":"...","label":"..."}]}"""


def score_claims(claims: list[dict]) -> float:
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c["label"] == "supported")
    return supported / len(claims)


def judge(reasoning: str, chunks: list[dict]):
    excerpts = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    response = client.messages.create(
        model=MODEL,
        max_tokens=1500,
        system=JUDGE_SYSTEM,
        messages=[{"role": "user", "content": f"EXCERPTS:\n{excerpts}\n\nREASONING:\n{reasoning}"}],
    )
    text = next(b.text for b in response.content if b.type == "text").strip()
    text = text.removeprefix("```json").removesuffix("```").strip()
    claims = json.loads(text)["claims"]
    return score_claims(claims), claims
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/eval/test_faithfulness.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/eval/faithfulness.py tests/eval/test_faithfulness.py
git commit -m "feat: add faithfulness judge (pure scoring helper + LLM judge call)"
```

---

### Task 11: Backtest against realized price moves

**Files:**
- Create: `src/backtest/__init__.py` (empty)
- Create: `src/backtest/score_signals.py`
- Test: `tests/backtest/test_score_signals.py`

**Interfaces:**
- Consumes: `yfinance`, `pandas`.
- Produces: `filed_after_market_close(filed_datetime: str) -> bool`, `forward_return(ticker: str, filed_datetime: str, horizon: int = 1) -> float | None`, `score(signals: list[dict]) -> pd.DataFrame`. Consumed by `scripts/backtest.py` (Task 12).

This is the highest-value test in the whole plan — it pins down the lookahead-bias logic the guide calls out as CRITICAL. Unlike the guide's version (a static `filed_after_hours` flag the caller must pass correctly), this implementation derives after-hours status directly from each filing's exact `filed_datetime` (the SEC `ACCEPTANCE-DATETIME`, already captured by `src/ingestion/parse.py`), which is both more accurate and impossible to pass wrong.

- [ ] **Step 1: Write the failing tests**

```python
# tests/backtest/test_score_signals.py
import pandas as pd
import pytest

import src.backtest.score_signals as score_mod


def _make_price_frame(dates, opens, closes):
    return pd.DataFrame({"Open": opens, "Close": closes}, index=pd.DatetimeIndex(dates))


class _FakeTicker:
    def __init__(self, prices):
        self._prices = prices

    def __call__(self, ticker):
        return self

    def history(self, start, end):
        return self._prices


def test_filed_after_market_close_detects_after_hours():
    assert score_mod.filed_after_market_close("2026-05-01T20:15:00") is True
    assert score_mod.filed_after_market_close("2026-05-01T06:00:00") is True
    assert score_mod.filed_after_market_close("2026-05-01T10:00:00") is False


def test_forward_return_uses_next_day_open_for_after_hours_filing(monkeypatch):
    # Filed 8:15pm on 2026-05-01 (after close) -> entry must be 2026-05-02's
    # open, NOT 2026-05-01's close (that would be lookahead bias).
    prices = _make_price_frame(
        ["2026-04-29", "2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03"],
        opens=[100, 101, 102, 200, 210],
        closes=[101, 102, 103, 205, 212],
    )
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    r = score_mod.forward_return("AAPL", "2026-05-01T20:15:00", horizon=1)

    assert r == pytest.approx((212 - 200) / 200)


def test_forward_return_uses_same_day_open_for_during_hours_filing(monkeypatch):
    prices = _make_price_frame(
        ["2026-04-29", "2026-04-30", "2026-05-01", "2026-05-02"],
        opens=[100, 101, 102, 103],
        closes=[101, 102, 103, 104],
    )
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    r = score_mod.forward_return("AAPL", "2026-05-01T10:00:00", horizon=1)

    assert r == pytest.approx((104 - 102) / 102)


def test_forward_return_returns_none_when_insufficient_future_data(monkeypatch):
    prices = _make_price_frame(["2026-05-01"], opens=[100], closes=[101])
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    assert score_mod.forward_return("AAPL", "2026-05-01T20:15:00", horizon=1) is None


def test_forward_return_returns_none_on_empty_history(monkeypatch):
    prices = pd.DataFrame({"Open": [], "Close": []})
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    assert score_mod.forward_return("BADTICKER", "2026-05-01T20:15:00", horizon=1) is None


def test_score_computes_hit_rate_and_treats_neutral_as_non_directional(monkeypatch):
    def fake_forward_return(ticker, filed_datetime, horizon=1):
        return {"AAPL": 0.05, "MSFT": -0.02, "NVDA": 0.01}[ticker]

    monkeypatch.setattr(score_mod, "forward_return", fake_forward_return)

    signals = [
        {"ticker": "AAPL", "filed_datetime": "2026-05-01T20:15:00", "signal": "bullish", "confidence": 0.8},
        {"ticker": "MSFT", "filed_datetime": "2026-05-01T20:15:00", "signal": "bullish", "confidence": 0.6},
        {"ticker": "NVDA", "filed_datetime": "2026-05-01T20:15:00", "signal": "neutral", "confidence": 0.4},
    ]

    df = score_mod.score(signals)

    assert len(df) == 3
    directional = df[df["hit"].notna()]
    assert len(directional) == 2  # neutral is excluded from hit-rate scoring
    assert directional["hit"].tolist() == [True, False]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest tests/backtest/test_score_signals.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'src.backtest'`

- [ ] **Step 3: Implement `src/backtest/score_signals.py`**

```python
# src/backtest/__init__.py
# (empty)
```

```python
# src/backtest/score_signals.py
"""Score generated trading signals against realized forward stock returns.

CRITICAL: entry must be the next trading session's open if the filing was
made outside regular market hours (before 9:30am or at/after 4:00pm ET) --
using the same-day close for an after-hours filing is lookahead bias and
produces fake results. We derive after-hours status from each filing's
exact `filed_datetime` (the SEC ACCEPTANCE-DATETIME captured by
src/ingestion/parse.py) rather than a caller-supplied flag, since that
timestamp already carries the Eastern local hour.
"""
from datetime import timedelta

import pandas as pd
import yfinance as yf

MARKET_CLOSE_HOUR_ET = 16
MARKET_OPEN_HOUR_ET = 9


def filed_after_market_close(filed_datetime: str) -> bool:
    """SEC ACCEPTANCE-DATETIME is Eastern local time; treat before 9am or
    at/after 4pm ET as outside regular trading hours."""
    ts = pd.Timestamp(filed_datetime)
    return ts.hour >= MARKET_CLOSE_HOUR_ET or ts.hour < MARKET_OPEN_HOUR_ET


def forward_return(ticker: str, filed_datetime: str, horizon: int = 1):
    filed_ts = pd.Timestamp(filed_datetime)
    after_hours = filed_after_market_close(filed_datetime)

    start = filed_ts - timedelta(days=5)
    end = filed_ts + timedelta(days=horizon + 10)
    prices = yf.Ticker(ticker).history(start=start, end=end)
    if prices.empty:
        return None

    dates = prices.index.normalize()
    filed_date = filed_ts.normalize()
    if dates.tz is not None and filed_date.tz is None:
        filed_date = filed_date.tz_localize(dates.tz)

    after = prices[dates > filed_date] if after_hours else prices[dates >= filed_date]
    if len(after) < horizon + 1:
        return None

    entry = after.iloc[0]["Open"]
    exit_price = after.iloc[horizon]["Close"]
    return (exit_price - entry) / entry


def score(signals: list[dict]) -> pd.DataFrame:
    """signals: [{ticker, filed_datetime, signal, confidence}]"""
    rows = []
    for s in signals:
        r = forward_return(s["ticker"], s["filed_datetime"])
        if r is None:
            continue
        predicted_up = s["signal"] == "bullish"
        actual_up = r > 0
        hit = (predicted_up == actual_up) if s["signal"] != "neutral" else None
        rows.append({**s, "return": r, "hit": hit})

    df = pd.DataFrame(rows)
    directional = df[df["hit"].notna()] if not df.empty else df
    print(f"signals: {len(df)}  directional: {len(directional)}")
    if len(directional):
        print(f"hit rate: {directional['hit'].mean():.2%}")
        high_conf = directional[directional["confidence"] >= 0.7]
        if len(high_conf):
            print(f"hit rate (conf>=0.7, n={len(high_conf)}): {high_conf['hit'].mean():.2%}")
        print(f"mean return: {directional['return'].mean():.2%}")
    return df
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `python -m pytest tests/backtest/test_score_signals.py -v`
Expected: PASS (6 tests)

- [ ] **Step 5: Commit**

```bash
git add src/backtest/__init__.py src/backtest/score_signals.py tests/backtest/test_score_signals.py
git commit -m "feat: add lookahead-safe backtest scoring against yfinance forward returns"
```

---

### Task 12: CLI wrappers for signal generation and backtesting

**Files:**
- Create: `scripts/run_signal.py`
- Create: `scripts/backtest.py`

**Interfaces:**
- Consumes: `src.generation.generate.generate_signal` (Task 9), `src.eval.faithfulness.judge` (Task 10), `src.backtest.score_signals.score` (Task 11), `src.ingestion.chunk.calendar_quarter` (existing Week 1 helper).
- Produces: two runnable CLI entry points. No new automated tests — these are thin orchestration scripts over already-tested modules, verified by running them live (requires `ANTHROPIC_API_KEY`).

- [ ] **Step 1: Implement `scripts/run_signal.py`**

```python
# scripts/run_signal.py
"""CLI: generate a signal for one ticker, judge its faithfulness, print both.

Usage:
    python scripts/run_signal.py AAPL
    python scripts/run_signal.py AAPL --quarter 2026Q2
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.faithfulness import judge
from src.generation.generate import generate_signal

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate and judge a trading signal for one ticker.")
    parser.add_argument("ticker")
    parser.add_argument("--quarter", default=None, help="e.g. 2026Q2 -- restricts retrieval to that filing")
    args = parser.parse_args()

    signal, chunks = generate_signal(args.ticker, fiscal_quarter=args.quarter)
    print(json.dumps(signal.model_dump(), indent=2))

    score, claims = judge(signal.reasoning, chunks)
    print(f"\nfaithfulness: {score:.2%}")
    for c in claims:
        print(f"  [{c['label']}] {c['claim']}")
```

- [ ] **Step 2: Implement `scripts/backtest.py`**

```python
# scripts/backtest.py
"""CLI: generate a signal for every ingested filing, backtest against
realized forward returns, print the results table.

Requires ANTHROPIC_API_KEY in .env and internet access for yfinance.

Usage:
    python scripts/backtest.py
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.backtest.score_signals import score
from src.generation.generate import generate_signal
from src.ingestion.chunk import calendar_quarter

SIGNALS_PATH = config.PROCESSED_DIR / "signals.jsonl"


def list_filings() -> list[dict]:
    filings = []
    for doc_path in sorted(config.PROCESSED_DIR.glob("*.json")):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        filings.append({
            "ticker": doc["ticker"],
            "filed_datetime": doc["filed_datetime"],
            "fiscal_quarter": calendar_quarter(doc["filing_date"]),
        })
    return filings


def run():
    signals = []
    for filing in list_filings():
        ticker, quarter = filing["ticker"], filing["fiscal_quarter"]
        print(f"generating signal for {ticker} {quarter} ({filing['filed_datetime']})...")
        signal, _ = generate_signal(ticker, fiscal_quarter=quarter)
        signals.append({
            "ticker": ticker,
            "fiscal_quarter": quarter,
            "filed_datetime": filing["filed_datetime"],
            "signal": signal.signal,
            "confidence": signal.confidence,
        })

    with SIGNALS_PATH.open("w", encoding="utf-8") as f:
        for s in signals:
            f.write(json.dumps(s) + "\n")
    print(f"\nwrote {len(signals)} signals -> {SIGNALS_PATH}")

    df = score(signals)
    print(f"\nNote: n={len(df)} signals is not a statistically meaningful sample -- "
          f"directional hit rate here is a first number, not a conclusion.")
    return df


if __name__ == "__main__":
    run()
```

- [ ] **Step 3: Verify (requires a real `ANTHROPIC_API_KEY`)**

Before this step, the user must copy `.env.example` to `.env` and fill in a real `ANTHROPIC_API_KEY` (and confirm `SEC_EDGAR_COMPANY_NAME`/`SEC_EDGAR_EMAIL` are already set from Week 1). This plan does not do that step — flag it to the user and pause here if `.env` still doesn't exist when this task is reached.

Run: `python scripts/run_signal.py AAPL --quarter 2026Q2`
Expected: prints a JSON `Signal` object citing real `chunk_id`s from `data/processed/chunks.jsonl`, followed by a per-claim faithfulness breakdown. If it cites a `chunk_id` that doesn't exist in the corpus, tighten `SYSTEM` in `src/generation/prompts.py` (per the guide's Step 12 verification note) rather than papering over it downstream.

Then run: `python scripts/backtest.py`
Expected: 10 signals generated (one per filing), `data/processed/signals.jsonl` written, and a hit-rate summary printed. **Write down the hit rate and mean return — Task 13 needs them.**

Manually validate the faithfulness judge on ~10 examples by hand (read the reasoning, read the excerpts, decide if you agree with each `supported`/`unsupported`/`contradicted` label) before trusting the faithfulness numbers in the README — an unvalidated judge is a number you can't defend.

- [ ] **Step 4: Commit**

```bash
git add scripts/run_signal.py scripts/backtest.py
git commit -m "feat: add CLI wrappers for signal generation and backtesting"
```

(Do not commit `data/processed/signals.jsonl` output from the verification run unless the user wants it tracked — check whether `data/processed/` is gitignored before staging it; if unsure, ask rather than assume.)

---

### Task 13: README update with Week 2 results

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: the three-config comparison table from Task 7 Step 7, and the backtest hit-rate/mean-return numbers from Task 12 Step 3.

- [ ] **Step 1: Update `README.md`**

Extend the existing README (keep the Week 1 sections — "Why should this text predict...", "Status: Week 1 complete", "What's built", "First real numbers", "Known limitations", "Running it" — and add the following new sections after "Known limitations"):

```markdown
## Week 2: retrieval comparison, generation, faithfulness, backtest

### What's built

- `src/retrieval/{dense,bm25,hybrid,rerank}.py` -- dense, BM25, RRF-fused hybrid, and cross-encoder-reranked retrieval behind a common `search(query, k, ticker=None, fiscal_quarter=None)` interface, so eval and generation share the exact same retrievers.
- `src/eval/metrics.py` -- Recall@k/Precision@k/MRR/NDCG@k as pure functions, unit-tested independently of any live index.
- `src/eval/run_eval.py` + `scripts/evaluate.py` -- compares dense vs hybrid vs reranked on the same 20-question golden set; every run is recorded to `src/eval/metrics.db`.
- `src/generation/{schema,prompts,generate}.py` -- Claude tool-use (`claude-opus-5`) emits a structured `Signal` (ticker/direction/confidence/reasoning/citations) grounded in hybrid-retrieved excerpts; `supporting_claims` has `min_length=1`, so an uncited signal is a validation error, not a silent hallucination.
- `src/eval/faithfulness.py` -- a second Claude call grades each claim in the generated reasoning as supported/unsupported/contradicted against the retrieved excerpts.
- `src/backtest/score_signals.py` -- scores signals against realized `yfinance` forward returns, entering at the next session's open whenever a filing landed outside regular trading hours (derived from the exact SEC `ACCEPTANCE-DATETIME`, not a guessed flag).
- `scripts/run_signal.py` / `scripts/backtest.py` -- CLI entry points tying generation, judging, and backtesting together.

### Retrieval comparison (bge-small-en-v1.5, top-5, 20-question golden set)

| Config | Recall@5 | Precision@5 | MRR | NDCG@5 |
|---|---|---|---|---|
| dense only | <FILL IN FROM scripts/evaluate.py OUTPUT> | | | |
| + BM25 (RRF) | | | | |
| + cross-encoder rerank | | | | |

<ONE SENTENCE: which configs were kept and why, based on whether each one measurably beat the previous row on Recall@5 -- per the project's own rule, an addition that doesn't improve the number doesn't stay in the default pipeline, even if it's still available as an option.>

### Faithfulness and backtest

- Faithfulness (fraction of reasoning claims labeled "supported" by the judge, spot-checked by hand against ~10 examples before trusting it): <FILL IN>
- Backtest: <N> signals scored, directional hit rate <FILL IN>, mean return <FILL IN>. **n=<N> is not a statistically meaningful sample** -- this is a first number, not a conclusion; expanding ticker/quarter coverage is future work, per the same honesty standard as Week 1's retrieval numbers.

### Design decisions and why

- **RRF over score-averaging for hybrid retrieval:** dense cosine scores live in [0,1] while BM25 scores are unbounded, so averaging them directly would let BM25 dominate. RRF uses only rank position, so no cross-scale normalization is needed.
- **`min_length=1` on `supporting_claims`:** without it, a signal with no citations is valid pydantic output -- exactly the uncited-hallucination failure mode the whole citation requirement exists to catch. Making it a schema constraint means Claude's own tool-call validation catches it, with an automatic retry that surfaces the validation error back to the model.
- **Next-session-open entry, derived from `filed_datetime` rather than a caller-supplied flag:** the original guide used a boolean the caller had to set correctly per filing. Since `src/ingestion/parse.py` already captures the SEC `ACCEPTANCE-DATETIME` for every filing, deriving after-hours status from that timestamp directly removes an entire class of "someone passed the wrong flag" lookahead-bias bugs.
- **Retrieval filtered by `fiscal_quarter` as well as `ticker`:** each ticker in this corpus has two filings (two different quarters). Filtering generation context by ticker alone would blend both quarters' chunks into one signal per ticker instead of one signal per filing, which is wrong for a per-filing backtest.

### Known limitations (honest, not swept under the rug)

- (carry forward the Week 1 limitations bullets)
- Faithfulness judge is itself an LLM call and was spot-checked against a small hand-reviewed sample, not a large held-out set -- treat the faithfulness percentage as directional, not precise.
- Backtest sample size (n=10 signals) is far too small for a real hit-rate claim; see the note above.
```

Fill in the `<FILL IN>` placeholders with the actual numbers recorded in Task 7 Step 7 and Task 12 Step 3 before committing — do not commit the literal placeholder text.

- [ ] **Step 2: Commit**

```bash
git add README.md
git commit -m "docs: add Week 2 results (retrieval comparison, faithfulness, backtest)"
```

---

## Self-Review Notes

- **Spec coverage:** Guide Steps 6 (dense retrieval reusable module) → Task 3; Step 8 extended with NDCG → Task 1 + Task 7; Step 9 (BM25 + hybrid) → Task 4 + 5; Step 10 (reranking) → Task 6; Step 11 (schema) → Task 8; Step 12 (generation) → Task 9; Step 13 (faithfulness) → Task 10; Step 14 (backtest, lookahead trap) → Task 11; Step 15 (CLI) → Task 12; Step 16 (README) → Task 13. Step 7 (golden eval set) was already completed in Week 1 and is reused as-is.
- **Deviations from the guide, and why:** chunk schema and Qdrant collection name differ from the guide (8-K earnings-release corpus with `fiscal_quarter`, not 10-Q MD&A/RiskFactors) because that's what Week 1 actually built — every task above uses the real schema, not the guide's. Retrieval functions gained an optional `fiscal_quarter` filter beyond what the guide specifies, because this corpus has two filings per ticker and per-filing backtesting requires per-filing (not per-ticker-blended) retrieval context. The tool-use retry in `generate_signal` sends a proper `tool_result` block instead of the guide's plain user-text retry message, since a `tool_use` turn must be followed by a matching `tool_result`.
- **Type/signature consistency check:** `search_fn` shape `(query, k, ticker=None, fiscal_quarter=None) -> list[dict with chunk_id]` is identical across `dense_search`, `bm25_search`, `hybrid_search`, `reranked_search`, and is what `evaluate()` (Task 7) and `gather_context()` (Task 9) both call against. `Signal`/`SupportingClaim` field names (Task 8) match what `build_user_prompt` emits (`chunk_id`, `section`) and what `judge()`'s excerpts format uses. Confirmed no placeholder/TODO text remains in any code block above.
