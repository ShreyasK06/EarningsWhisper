# EarningsWhisper

RAG system over SEC 8-K earnings-release filings that retrieves grounded excerpts, generates structured trading signals with citations, judges those signals for faithfulness, and backtests them against realized forward returns -- plus a FastAPI + React product layer on top.

## Why should this text predict price movement better than chance?

Reported numbers -- revenue, EPS, segment growth -- are priced in within seconds by algorithmic traders reading the same 8-K the moment it hits EDGAR. What isn't priced in as quickly is the *qualitative* language wrapped around those numbers: a CFO attributing a margin beat to a one-time tariff refund rather than durable pricing power, a guidance range that quietly widens or narrows quarter over quarter, or management's own hedging about whether a growth driver is repeatable. That signal is slower to be fully incorporated because it requires reading and judgment, not just parsing a number -- which is exactly the gap a retrieval-grounded language model can exploit. If this system can't out-predict a coin flip on directional hit rate once backtested, the retrieval/generation infrastructure doesn't matter, and this README says so plainly below rather than hiding it.

## Results (current: 20 tickers x up to 4 quarters, 47-question golden set)

**Retrieval** (`bge-small-en-v1.5`, 2079 chunks, dense / hybrid RRF / cross-encoder reranked):

| Config | Recall@1 | Recall@3 | Recall@5 | Recall@10 | Precision@5 | MRR | NDCG@5 |
|---|---|---|---|---|---|---|---|
| dense | 0.234 | 0.404 | 0.447 | 0.596 | 0.089 | 0.321 | 0.352 |
| hybrid (RRF) | 0.383 | 0.468 | 0.617 | 0.766 | 0.123 | 0.458 | 0.497 |
| **reranked** | **0.404** | **0.489** | **0.638** | 0.745 | **0.128** | **0.477** | **0.517** |

**Reranked wins at every k except k=10**, where hybrid edges it out (0.766 vs 0.745) -- both clearly beat dense throughout. **`reranked_search` is the default** for both signal generation and the Q&A API (see "How the default retriever changed" below).

**Faithfulness**: 100% (all claims graded "supported") on 3 live-verified signals (AAPL, MSFT, NVDA), spot-checked by hand. This is n=3, not a corpus-wide measurement -- see Limitations.

**Backtest**: 75 signals generated across the full corpus (58 bullish, 11 neutral, 6 bearish); 64 directional. **Hit rate: 48.44%, mean return: -0.28%**. This is consistent with chance, not evidence of predictive edge -- see "The backtest result" below for why that's the honest reading, and see "A backtest-hit-rate bug" for why you should trust this specific number.

## How the default retriever changed (a real, evidence-driven reversal)

Week 2 shipped a negative result: on a 20-question golden set, Recall@5 *declined* with each retrieval addition (dense 0.775 -> hybrid 0.750 -> reranked 0.725), so dense stayed the default. That result did not survive scale:

| Stage | Golden Qs | Filings | dense R@5 | hybrid R@5 | reranked R@5 | winner |
|---|---|---|---|---|---|---|
| Week 2 | 20 | 10 | 0.775 | 0.750 | 0.725 | dense |
| Week 3, golden-set expanded | 47 | 10 | 0.713 | 0.809 | 0.819 | reranked |
| Week 3, corpus also expanded | 47 | 75 | 0.447 | 0.617 | 0.638 | reranked |

Two things happened, and they're distinct effects: expanding the golden set from 20 to 47 questions **reversed the ranking** (reranked and hybrid both clearly beat dense once there were enough questions to measure the difference reliably -- the 20-question set was too small to trust). Expanding the corpus from 10 to 75 filings then **lowered every config's absolute numbers roughly in parallel** (more distractor chunks per query mechanically dilutes recall/precision/MRR) without changing which config wins. Per the project's own rule -- promote a retriever only when new evidence justifies it -- `src/generation/generate.py`'s `gather_context()` and `api/main.py`'s Q&A path were switched from dense/hybrid to `reranked_search` this week.

## The backtest result: an honest null, not a failure

48.44% hit rate on 64 directional signals is indistinguishable from a coin flip (50%), and the mean return (-0.28%) is negative. Per this project's own stated falsification criterion in the opening section: **this system does not currently demonstrate predictive edge on directional hit rate.** That's a real, reportable finding, not a bug to explain away -- n=64 is a meaningfully larger sample than Week 2's n=10, and it points the same direction (no edge). It does not rule out that a longer time horizon, a different confidence-weighting scheme, or a larger corpus would reveal something; it says that *this* system, scored this way, right now, does not beat chance.

## A backtest-hit-rate bug (why the 48.44% number, not an earlier wrong one, is the one to trust)

The first run of the expanded-corpus backtest printed a **1.56% hit rate (1 of 64)** -- a number so far from 50% that it demanded investigation rather than being reported as-is. The cause: `src/backtest/score_signals.py` computed `r > 0` on a `numpy.float64` return, which produces a `numpy.bool_`. `numpy.bool_` addition is *logical OR*, not integer addition. The `hit` column mixes these values with `None` for neutral signals (forcing pandas to store the column as `object` dtype), and `pandas.Series.mean()` on an object-dtype column reduces via Python's `+` -- so N `numpy.bool_` values silently collapsed via OR into a single `True`/`False` before dividing by N, instead of counting hits. The fix (casting `hit` to a native Python `bool` at construction, `src/backtest/score_signals.py`) is one line; the regression test (`tests/backtest/test_score_signals.py`) uses `numpy.float64` returns specifically, because the existing test's plain-float mock never exercised the failure mode. This is left in the README deliberately: a pipeline can retrieve correctly, generate correctly, and still report a nonsense headline metric from an aggregation bug three lines from the finish line -- worth remembering before trusting any single number without spot-checking the arithmetic behind it.

## Architecture

```
SEC EDGAR (8-K filings)
        |
        v
  src/ingestion/
  fetch -> parse -> chunk -> embed
        |
        v
   Qdrant (local, on-disk)
        |
        v
  src/retrieval/
  dense | bm25 | hybrid (RRF) | rerank (cross-encoder)  <-- reranked_search is default
        |
        v
  src/generation/
  llm_client (Gemini) -> generate.py -> Signal (pydantic, cited)
        |
        +---------------------------+
        v                           v
  src/eval/                  src/backtest/
  faithfulness judge         score_signals.py (yfinance forward returns)
  run_eval / metrics (Recall@k, MRR, NDCG@k)
        |
        v
  api/main.py (FastAPI: /chat, /tickers)  <-- reranked_search for Q&A
        |
        v
  frontend/ (React + Vite)
```

## What's built

**Ingestion** (`src/ingestion/`): `fetch.py` downloads Item-2.02 ("Results of Operations") 8-Ks via `sec-edgar-downloader`, over-fetching candidates and filtering out unrelated filings (officer changes, litigation, etc.). `parse.py` extracts the EX-99.1 exhibit and captures the real SEC `ACCEPTANCE-DATETIME`. `chunk.py` does section-aware chunking (regex header detection across `HEADER_PATTERNS`, sentence-packing narrative sections with `nltk`, line-packing financial tables) -- a chunk never crosses a section boundary. `embed.py` embeds with `BAAI/bge-small-en-v1.5` and upserts into Qdrant local mode (`./qdrant_data/`, gitignored).

**Retrieval** (`src/retrieval/`): dense, BM25, RRF-fused hybrid, and cross-encoder-reranked retrieval behind a common `search(query, k, ticker=None, fiscal_quarter=None)` interface, so eval and generation share the exact same retrievers.

**Eval** (`src/eval/`): `metrics.py` -- Recall@k/Precision@k/MRR/NDCG@k as pure, unit-tested functions. `run_eval.py` -- `evaluate()` (single config) and `sweep()` (every config x every k in one pass, k=1/3/5/10), every run logged to `src/eval/metrics.db` with a `config_hash`. `faithfulness.py` -- grades each generated claim as supported/unsupported/contradicted against the retrieved excerpts it was grounded in.

**Generation** (`src/generation/`): `llm_client.py` -- a provider-agnostic `LLMClient` abstraction (originally Claude tool-use, migrated to Gemini this week -- see "LLM backend" below) with retry-with-backoff on transient server errors, request timeouts, and rate-limited 429s; a hard daily-quota 429 (no `retryDelay` from the server) still fails fast rather than burning remaining quota. `generate.py` -- emits a structured `Signal` (ticker/direction/confidence/reasoning/citations) grounded in `reranked_search` excerpts; `supporting_claims` has `min_length=1`, so an uncited signal is a pydantic validation error, not a silent hallucination, with one automatic retry that surfaces the validation error back to the model.

**Backtest** (`src/backtest/`): scores signals against realized `yfinance` forward returns, entering at the next session's open whenever a filing landed at or after market open (derived from the exact SEC `ACCEPTANCE-DATETIME`, not a guessed flag) -- avoiding intraday lookahead bias.

**Product layer**: `api/main.py` -- FastAPI `/chat` endpoint routing each message to either open Q&A (`reranked_search` + a grounded answer with citations) or a directional signal call, plus `/tickers`. `frontend/` -- a React + Vite app consuming that API.

**Golden set** (`data/golden/eval_set.jsonl`): **47 questions** -- see "Golden-set provenance" below for how the 20 -> 47 expansion was done and how it's disclosed.

## Golden-set provenance (read this before trusting the "hand-labeled" framing)

The original 20 questions (Week 1) were genuinely hand-labeled: read and written by a human against the actual parsed filings, each citing real `chunk_id`s. The 27 questions added this week (Task 4) were **not** independently human-labeled the same way -- they were AI-drafted (`scripts/suggest_questions.py`, stratified sampling across ticker/section, 2 candidate questions per chunk via Gemini) and then AI-verified: each candidate's source chunk was read in full and the question kept only if it unambiguously and specifically pointed to that chunk, with 7 of 34 raw candidates rejected for ambiguity or near-duplication. This is disclosed explicitly rather than folded into "hand-labeled 47" because the distinction matters for how much to trust the golden set as ground truth -- AI-drafted-and-verified questions carry a real, if smaller, risk of circularity that independently-authored human questions don't.

## Corpus

20 tickers x up to 4 quarters = **75 filings** (18 tickers got the full 4 quarters; JPM has 2 and XOM has 1 -- SEC EDGAR's recent-filing window for those two CIKs genuinely contains fewer distinct Item-2.02 8-Ks within the fetch's over-fetch limit, not a parsing bug), **2079 chunks**. Tickers: AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, V, UNH, XOM, WMT, HD, PG, KO, DIS, NFLX, CRM, ORCL, AVGO.

Two tickers needed new `HEADER_PATTERNS` in `src/ingestion/chunk.py` because their press releases used unrecognized section-header formats and were landing almost entirely in the catch-all `PreparedRemarks` bucket: **JPM** (100% -> 3.8% PreparedRemarks after adding patterns for `SIGNIFICANT ITEMS IN...RESULTS`, `CAPITAL DISTRIBUTIONS`, `FIRMWIDE METRICS`, segment headers like `CONSUMER & COMMUNITY BANKING (CCB)`) and **XOM** (100% -> 3.2% after adding patterns for `Results Summary`, `EARNINGS AND VOLUME SUMMARY BY SEGMENT`, `CASH FLOW FROM OPERATIONS`, `ADJUSTING ITEMS`, `Cautionary Statement`). Two others were investigated and explicitly **not** fixed this week because the fix belongs in `parse.py`, not `chunk.py`: **TSLA** (93.2% PreparedRemarks -- its release is letter-spaced PDF-extracted text like `F I N A N C I A L   S T A T E M E N T S`, so header regexes can't match it) and **UNH** (83.8% -- its real headers are embedded mid-paragraph with no preceding newline, and `chunk.py`'s header classifier requires a header at line-start).

## LLM backend

Migrated from Anthropic Claude (`claude-opus-5` tool-use) to Google Gemini this week, behind the `LLMClient` abstraction in `src/generation/llm_client.py` (`LLM_BACKEND=gemini` in `.env`). Default model is `gemini-3.1-flash-lite`, not the newer `gemini-3.6-flash` -- the free-tier key hit `gemini-3.6-flash`'s **20-requests/day** hard quota mid-verification; `gemini-3.1-flash-lite` has a separate quota pool (a 15-requests/minute *rate limit*, not a daily cap) that the client now retries against using the server's own suggested `retryDelay`, while still failing fast on the kind of 429 that carries no `retryDelay` (a hard quota exhaustion, where retrying would just waste what's left). Nested pydantic schemas (`$ref`/`$defs`, e.g. `Signal.supporting_claims: list[SupportingClaim]`) are flattened before being sent as Gemini's `response_schema`, since Gemini doesn't resolve `$ref` indirection itself.

## Design decisions and why

- **RRF over score-averaging for hybrid retrieval**: dense cosine scores live in [0,1] while BM25 scores are unbounded, so averaging them directly would let BM25 dominate. RRF uses only rank position, so no cross-scale normalization is needed.
- **`min_length=1` on `supporting_claims`**: without it, a signal with no citations is valid pydantic output -- exactly the uncited-hallucination failure mode the citation requirement exists to catch.
- **Next-session-open entry, derived from `filed_datetime` rather than a caller-supplied flag**: `parse.py` already captures the SEC `ACCEPTANCE-DATETIME` for every filing, so deriving after-hours status from that timestamp directly removes an entire class of "someone passed the wrong flag" lookahead-bias bugs.
- **Retrieval filtered by `fiscal_quarter` as well as `ticker`**: each ticker has multiple filings (multiple quarters). Filtering generation context by ticker alone would blend quarters into one signal instead of one signal per filing, which is wrong for a per-filing backtest.
- **Gemini client retries transient errors but never a hard-quota 429**: a 503 (model overloaded), a stalled socket (60s timeout, since the SDK doesn't set one by default and a stalled call was observed hanging indefinitely with no error), and a rate-limited 429 (respecting the server's `retryDelay`) are all worth retrying. A 429 with no `retryDelay` means the daily allotment is gone -- retrying that burns nothing but time and API calls that won't succeed.
- **`reranked_search` promoted to the default retriever**: see "How the default retriever changed" above -- this was a plan-mandated change, made only once new evidence (the 47-question re-eval) justified it, not a preference swap.

## Known limitations (honest, not swept under the rug)

- **Golden-set provenance**: 27 of 47 questions are AI-drafted-and-verified, not independently human-labeled -- see "Golden-set provenance" above. Treat retrieval numbers as somewhat less independently-verified than a fully hand-labeled set would give.
- **Faithfulness validated on n=3**, not the full 75-signal corpus (`scripts/backtest.py` generates signals but does not call the faithfulness judge on all of them -- only `scripts/run_signal.py`'s individual runs did, for AAPL/MSFT/NVDA). Treat the 100% figure as directional, not a corpus-wide measurement.
- **Backtest n=75 (64 directional) shows no predictive edge** (48.44% hit rate, ~chance) -- see "The backtest result" above. This is the headline honest finding this week, not a number to bury.
- **No transaction costs modeled**; single data source (SEC EDGAR only); one-day forward-return horizon only.
- **Two tickers (TSLA, UNH) have degraded section-level chunking** (see Corpus above) -- their financial-table and outlook chunks are harder to retrieve precisely because most of their text landed in one large `PreparedRemarks` bucket instead of being split by section. This is a `parse.py`-level text-extraction gap, not fixed this week.
- **JPM and XOM have fewer than 4 quarters** (2 and 1 respectively) due to real SEC EDGAR filing-history availability within the fetch window, not a bug.
- Chunking is regex-header-based; a few filings (notably Amazon's) have HTML where a boilerplate paragraph runs directly into a financial-table header with no line break, so a couple of chunks mix prose and tabular text.
- `toyapp/` and `docker-compose.yml` (from an early Docker-first attempt) are left in place but dormant -- not part of the active pipeline.

## Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
cp .env.example .env            # fill in GEMINI_API_KEY, SEC_EDGAR_COMPANY_NAME/EMAIL

# small starter corpus (5 tickers x 2 quarters):
python scripts/ingest_all.py --tickers AAPL MSFT NVDA AMZN GOOGL --quarters 2
# the full corpus behind the Results numbers above (20 tickers x 4 quarters,
# config.TICKERS already lists all 20, so --quarters is the only flag needed):
python scripts/ingest_all.py --quarters 4

python scripts/evaluate.py
python scripts/eval_sweep.py

# requires GEMINI_API_KEY:
python scripts/run_signal.py AAPL --quarter 2026Q2
python scripts/backtest.py

# product layer:
uvicorn api.main:app --reload --port 8000
cd frontend && npm install && npm run dev
```
