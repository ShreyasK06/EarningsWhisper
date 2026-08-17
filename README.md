# EarningsWhisper

RAG system over earnings-related SEC filings that generates structured trading signals, evaluated with a real retrieval harness and (eventually) backtested against price movement.

## Why should this text predict price movement better than chance?

Reported numbers -- revenue, EPS, segment growth -- are priced in within seconds by algorithmic traders reading the same 8-K the moment it hits EDGAR. What isn't priced in as quickly is the *qualitative* language wrapped around those numbers: a CFO attributing a margin beat to a one-time tariff refund rather than durable pricing power (see Apple's Q3 2026 release), a guidance range that quietly widens or narrows quarter over quarter, or management's own hedging about whether a growth driver is repeatable. That signal is slower to be fully incorporated because it requires reading and judgment, not just parsing a number -- which is exactly the gap a retrieval-grounded language model can exploit. If this system can't out-predict a coin flip on directional hit rate once backtested, the retrieval/generation infrastructure doesn't matter; Phase 3 will report that number honestly, including the caveat that 10 documents is not a statistically meaningful sample.

## Status: Week 1 complete -- running Docker-free

Docker Desktop had install problems on this machine, so per plan the project runs as plain Python: no containers, no Compose, no cluster. Qdrant runs in `qdrant-client`'s embedded local mode instead of a server -- same API, one-line swap to a real server later if needed. Nothing about the retrieval/eval substance changed; only how it's run.

### What's built

- `config.py` -- central settings loaded from `.env` (API keys, `QDRANT_PATH`, `EMBED_MODEL`, SEC EDGAR identity).
- `src/ingestion/fetch.py` -- downloads 8-K filings (Exhibit 99.1 earnings press releases) for AAPL, MSFT, NVDA, AMZN, GOOGL via `sec-edgar-downloader`, over-fetching candidates and filtering to `Item 2.02 Results of Operations` filings only (many 8-Ks are unrelated -- officer changes, litigation, etc.).
- `src/ingestion/parse.py` -- extracts the EX-99.1 exhibit from each filing's SGML submission and cleans it to text via BeautifulSoup4 + lxml.
- `src/ingestion/chunk.py` -- section-aware chunking: detects headers (Business Outlook, Conference Call Information, Forward-Looking Statements, financial-statement tables, etc.) via regex, sentence-packs narrative sections with `nltk` (~230 words, ~40-word overlap), and packs financial tables by raw line instead of by sentence since their "sentences" are just numbers. A chunk never crosses a section boundary. Produces **193 chunks** across the 10 filings.
- `data/golden/eval_set.jsonl` -- **20 hand-labeled Q&A pairs**, read and written against the actual parsed filings (not synthetic), each citing real `chunk_id`s verified to exist in `chunks.jsonl`.
- `src/ingestion/embed.py` -- embeds all chunks with `BAAI/bge-small-en-v1.5` (with the asymmetric query-prefix for search queries) and upserts into Qdrant local mode (`./qdrant_data/`, gitignored, no server).
- `src/eval/run_eval.py` + `src/eval/metrics_db.py` -- dense-only retrieval eval against the golden set, recorded to SQLite (`src/eval/metrics.db`, `eval_runs` table, one row per config).
- `scripts/ingest_all.py` / `scripts/evaluate.py` -- CLI entrypoints tying the above together.

### First real numbers (dense-only retrieval, bge-small-en-v1.5, top-5)

| Metric | Value |
|---|---|
| Recall@5 | **0.775** |
| Precision@5 | 0.160 |
| MRR | 0.541 |

Re-verified after the Docker-free restructure -- identical numbers, confirming the move didn't change any retrieval behavior. 3 of 20 golden questions missed entirely at k=5 (an iPhone-segment financial-table lookup, an OpenAI GAAP/non-GAAP reconciliation question, and two NVIDIA business-highlight lookups where the relevant chunk landed just outside top-5) -- exactly the kind of gap hybrid retrieval (BM25 for exact numeric/entity matches) and reranking are supposed to close in later iterations. Per the plan, each retrieval addition (BM25, then reranking) must show a measurable improvement over this baseline or it doesn't stay in.

### Known limitations (honest, not swept under the rug)

- 10 documents / 20 eval questions is enough to get a first number, not enough to draw strong conclusions -- expanding ticker/quarter coverage comes after the pipeline is proven, per the plan.
- Chunking is regex-header-based; a few filings (notably Amazon's) have HTML where a boilerplate "About Amazon" paragraph runs directly into a financial-table header with no line break, so a couple of chunks mix prose and tabular text. Not fixed this week -- noted for a future iteration (HTML-aware `<table>` detection instead of post-text-extraction regex).
- `toyapp/` and `docker-compose.yml` (from the original Docker-first attempt) are left in place but dormant -- harmless, and there if Docker gets revisited later (Phase 4 Option C in the revised plan). They're not part of the active pipeline.

### Running it

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows; macOS/Linux: source .venv/bin/activate
pip install --upgrade pip
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU-only, avoids a 2.5GB CUDA download
pip install -r requirements.txt
cp .env.example .env            # fill in ANTHROPIC_API_KEY, SEC_EDGAR_COMPANY_NAME/EMAIL

python scripts/ingest_all.py --tickers AAPL MSFT NVDA AMZN GOOGL --quarters 2
python scripts/evaluate.py
```

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
| dense only | 0.775 | 0.160 | 0.541 | 0.586 |
| + BM25 (RRF) | 0.750 | 0.160 | 0.629 | 0.656 |
| + cross-encoder rerank | 0.725 | 0.150 | 0.625 | 0.647 |

Per the project's own stated rule -- an addition stays only if it measurably improves Recall@5 -- **neither hybrid nor reranking clears the bar**: Recall@5 actually declines with each addition (0.775 -> 0.750 -> 0.725) on this 20-question golden set, so the default pipeline stays dense-only, with hybrid and reranking left available as opt-in configs rather than promoted to default. That said, this isn't a clean "hybrid failed" story: MRR (0.541 -> 0.629) and NDCG@5 (0.586 -> 0.656) both improve meaningfully going from dense to hybrid, and stay roughly flat to slightly down from there through reranking -- meaning that when the right chunk is retrieved, hybrid tends to rank it higher, it just also displaces a few correct chunks out of the top-5 entirely on this small set. That's a real, if different, kind of signal, and exactly the kind of nuance this evaluation methodology exists to surface rather than hide behind a single headline number.

### Faithfulness and backtest

Generation, faithfulness judging, and backtesting are implemented (`src/generation/`, `src/eval/faithfulness.py`, `src/backtest/score_signals.py`, `scripts/run_signal.py`, `scripts/backtest.py`) but **have not yet been run live in this environment** -- this worktree has no `ANTHROPIC_API_KEY` configured, and both generation and faithfulness judging require live Claude API calls. To produce real numbers:

```bash
# fill in ANTHROPIC_API_KEY in .env, then:
python scripts/run_signal.py   # generate signals for the ingested filings
python scripts/backtest.py     # judge faithfulness + score against realized returns
```

Once run, expect a faithfulness percentage (fraction of reasoning claims labeled "supported" by the judge, which should be spot-checked by hand against ~10 examples before being trusted) and a backtest summary of directional hit rate and mean return over the available filings (n<=10, since there are 10 filings across 5 tickers x 2 quarters). **That n=10 is not a statistically meaningful sample regardless of which way the hit rate lands** -- consistent with the honesty standard applied to every other number in this README, we're not filling this row in with an invented figure just to have one.

### Design decisions and why

- **RRF over score-averaging for hybrid retrieval:** dense cosine scores live in [0,1] while BM25 scores are unbounded, so averaging them directly would let BM25 dominate. RRF uses only rank position, so no cross-scale normalization is needed.
- **`min_length=1` on `supporting_claims`:** without it, a signal with no citations is valid pydantic output -- exactly the uncited-hallucination failure mode the whole citation requirement exists to catch. Making it a schema constraint means Claude's own tool-call validation catches it, with an automatic retry that surfaces the validation error back to the model.
- **Next-session-open entry, derived from `filed_datetime` rather than a caller-supplied flag:** the original guide used a boolean the caller had to set correctly per filing. Since `src/ingestion/parse.py` already captures the SEC `ACCEPTANCE-DATETIME` for every filing, deriving after-hours status from that timestamp directly removes an entire class of "someone passed the wrong flag" lookahead-bias bugs.
- **Retrieval filtered by `fiscal_quarter` as well as `ticker`:** each ticker in this corpus has two filings (two different quarters). Filtering generation context by ticker alone would blend both quarters' chunks into one signal per ticker instead of one signal per filing, which is wrong for a per-filing backtest.

### Known limitations (honest, not swept under the rug)

- 10 documents / 20 eval questions is enough to get a first number, not enough to draw strong conclusions -- expanding ticker/quarter coverage comes after the pipeline is proven, per the plan.
- Chunking is regex-header-based; a few filings (notably Amazon's) have HTML where a boilerplate "About Amazon" paragraph runs directly into a financial-table header with no line break, so a couple of chunks mix prose and tabular text. Not fixed this week -- noted for a future iteration (HTML-aware `<table>` detection instead of post-text-extraction regex).
- `toyapp/` and `docker-compose.yml` (from the original Docker-first attempt) are left in place but dormant -- harmless, and there if Docker gets revisited later (Phase 4 Option C in the revised plan). They're not part of the active pipeline.
- Hybrid retrieval and reranking improve MRR/NDCG@5 but not Recall@5 on this 20-question golden set, so neither is promoted to the default pipeline -- see the retrieval comparison table above. This may look different on a larger golden set; it isn't re-tuned away this week because that would defeat the point of measuring honestly.
- Generation, faithfulness, and backtest numbers are not yet populated in this README -- they require `ANTHROPIC_API_KEY` and a live run (`scripts/run_signal.py`, `scripts/backtest.py`), which hasn't happened in this environment yet. Once run, the backtest's n<=10 sample size caveat applies exactly as it does for every other small-sample number in this project.
- Faithfulness judging, once run, will itself be an LLM call spot-checked against a small hand-reviewed sample rather than a large held-out set -- treat any resulting faithfulness percentage as directional, not precise.
