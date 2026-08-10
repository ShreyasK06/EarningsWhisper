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
