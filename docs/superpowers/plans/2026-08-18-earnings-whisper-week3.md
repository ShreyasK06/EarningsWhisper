# Earnings Whisper — Week 3 (Live Verification, Error Analysis, Golden-Set & Corpus Expansion, README) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Finish the Week 3 handoff brief on top of the merged Week 2 codebase (retrieval stack, eval harness, generation, faithfulness, backtest — all code-complete and unit-tested, but generation/faithfulness/backtest never run against the live Anthropic API): live-verify the untested paths and fix what breaks; explain *why* hybrid/reranking failed to beat dense on Recall@5 via per-question error analysis and a Recall@k sweep; expand the golden set from 20→50 questions; expand the corpus from 5 tickers×2 quarters to ~20 tickers×4 quarters; write the final README with honestly-reported results including the negative result as a first-class finding; clean up an orphaned worktree directory.

**Architecture:** No new architectural layers — this is verification, analysis, and scale-up of the Week 2 pipeline. `scripts/error_analysis.py` and a new `sweep()` function in `src/eval/run_eval.py` (plus `scripts/eval_sweep.py`) both reuse the existing `search_fn(query, k) -> list[dict]` retrieval contract and `src/eval/metrics.py`. `scripts/suggest_questions.py` reuses the existing Anthropic client pattern from `src/generation/generate.py` to draft (never auto-label) golden-set candidates.

**Tech Stack:** Same as Week 1/2 — Python, qdrant-client, sentence-transformers, rank-bm25, anthropic SDK (`claude-opus-5`), pydantic v2, yfinance, pandas, pytest.

## Global Constraints

- Work happens on `main` directly in the primary checkout (`C:\Users\Shreyas Kakkar\Documents\GitHub\EarningsWhisper`) unless the user asks for an isolated worktree — confirm with them at plan-execution time per `superpowers:using-git-worktrees`, don't assume.
- **`ANTHROPIC_API_KEY` is NOT currently in `.env`** despite the brief's claim — verified: no `.env` file exists in this checkout, and the orphaned Week 2 worktree directory doesn't have one either. Task 1 is genuinely blocked on this. Do not fabricate a key, do not skip Task 1 silently — surface this to the user and wait.
- The brief's background section says "HTML parse (MD&A / Risk Factors)" — this is a description mismatch carried over from the *original* 10-Q-oriented build guide. The actual corpus (confirmed in the codebase) is 8-K earnings press releases with sections `{About, BusinessOutlook, ConferenceCall, Contacts, FinancialTables, ForwardLooking, PreparedRemarks}`, chunked by `src/ingestion/chunk.py`'s `HEADER_PATTERNS` regex list. Every task below uses the real section labels and the real `src/ingestion/{fetch,parse,chunk}.py` module names — not "MD&A/Risk Factors."
- Chunk schema unchanged from Week 2: `{chunk_id, ticker, fiscal_quarter, filing_date, filed_datetime, accession, section, speaker, text}`. `chunk_id` format `{TICKER}_{fiscal_quarter}_{section}_{seq:03d}`.
- `src/eval/metrics.db` is gitignored — the copy in this checkout is **stale** (only has 4 old Week-1 dense-only rows, pre-`ndcg_at_5` migration; the real Week 2 dense/hybrid/reranked comparison numbers were logged in the now-orphaned worktree's copy, not this one). The numbers themselves (dense 0.775/0.160/0.541/0.586, hybrid 0.750/0.160/0.629/0.656, reranked 0.725/0.150/0.625/0.647) are preserved in `README.md`'s "Week 2" section, committed to git — reuse those from the README, don't try to recover the old DB file.
- `src/eval/metrics_db.py`'s existing columns (`recall_at_5`, `precision_at_5`, `ndcg_at_5`) keep their historical at-k=5 meaning forever — never write a k≠5 value into them (this is the brief's explicit "don't silently overload existing columns" constraint). Task 3 adds new generic `k`/`recall_at_k`/`precision_at_k`/`ndcg_at_k` columns via migration instead.
- Do not add LangChain, LlamaIndex, or any retrieval framework — keep RRF, chunking, and fusion hand-rolled per the brief's explicit constraint.
- Do not change the default retriever away from dense-only without a new eval number that actually justifies it — the negative result from Week 2 stays the default unless Tasks 2/3 genuinely change the picture (e.g. if the k=10 sweep shows hybrid winning, that's a legitimate, evidence-based reason to revisit the default — update `src/generation/generate.py`'s import only if this happens, and say so explicitly in the README).
- Every retrieval/eval config change gets logged to `src/eval/metrics.db`'s `eval_runs` table with a real `config_hash` — no untracked tuning runs.
- New source files get a matching `tests/<mirror-path>/test_*.py` where the code is deterministic/pure enough to unit test (Tasks 2, 3, 7). Tasks 1, 5's live/data-collection work is verified by running it and inspecting output, not by new unit tests, matching how Week 2 handled `generate_signal`/`judge`.
- Every commit uses `git add <specific files>` — never `git add -A` (gitignored `qdrant_data/`, `.env`, `data/raw/`, `src/eval/metrics.db`, `data/golden/candidates.jsonl` if generated must never be staged unless a task explicitly says to commit it).

---

### Task 1: Live verification of generation, faithfulness judge, and backtest

**Files:**
- Modify (only if live testing surfaces a real bug): `src/generation/schema.py`, `src/generation/generate.py`, `src/eval/faithfulness.py`, `src/backtest/score_signals.py`
- Test (only for the specific defensive-parsing fix, if needed): `tests/eval/test_faithfulness.py`

**Interfaces:**
- Consumes: `ANTHROPIC_API_KEY` from `.env` (blocking prerequisite — see below), the already-built `qdrant_data/` index and `data/processed/chunks.jsonl` (10 filings, already present in this checkout).
- Produces: confirmation that `scripts/run_signal.py` and `scripts/backtest.py` run cleanly end-to-end against the live API, with any real bugs found along the way fixed and covered by a regression test.

This task is **exploratory verification + fix**, not "write this exact code" — the plan can't predict which of the brief's four flagged risk areas will actually break, if any. Use `superpowers:systematic-debugging` for any failure encountered (reproduce, isolate, root-cause, minimal fix) rather than guessing at patches.

- [ ] **Step 0: Confirm the API key is actually available before starting**

Check: `test -f .env && grep -q ANTHROPIC_API_KEY .env`. If this fails, STOP and tell the user directly — the brief claims a key is already configured but it is not present in this checkout (verified during planning). Ask them to copy `.env.example` to `.env` and fill in a real `ANTHROPIC_API_KEY` before continuing. Do not proceed past this step without it.

- [ ] **Step 1: Run the first live signal generation**

```bash
python scripts/run_signal.py AAPL --quarter 2026Q2
```

Watch for:
- A raw API error mentioning `$defs`/`$ref`/schema validation — `Signal.model_json_schema()` (in `src/generation/generate.py`) emits a nested-model JSON Schema for `SupportingClaim`. If the Anthropic tool-use endpoint rejects it, the fix is to flatten the schema: either inline `SupportingClaim`'s fields directly into a hand-written `input_schema` dict (bypassing `model_json_schema()`), or resolve the `$ref` yourself before passing it as `TOOL["input_schema"]`. Confirm which approach before implementing — inlining a hand-written schema is simpler and more explicit; only chase automatic `$ref` resolution if there's a strong reason to keep using `model_json_schema()` directly.
- A successful call that prints a valid `Signal` JSON citing real `chunk_id`s that exist in `data/processed/chunks.jsonl` for AAPL 2026Q2. If a citation doesn't exist in the corpus, that's a prompt-tightening problem in `src/generation/prompts.py`'s `SYSTEM` string, not a schema problem — don't conflate the two failure modes.

- [ ] **Step 2: Run the faithfulness judge as part of the same script**

`scripts/run_signal.py` calls `judge()` right after `generate_signal()` — this exercises `src/eval/faithfulness.py` live. Watch for:
- The judge wrapping its JSON response in markdown fences the current code doesn't handle (`judge()` only strips a leading ` ```json ` / trailing ` ``` ` pair) or in surrounding prose ("Here is my analysis: {...}"). If this happens, make parsing defensive: extract the first balanced `{...}` block from the response text (e.g. find the first `{` and its matching closing `}` by bracket-depth counting, or a regex like `re.search(r"\{.*\}", text, re.DOTALL)` as a simpler first attempt) before `json.loads`, and retry the API call once on a `json.JSONDecodeError` with a follow-up message asking for JSON only. Write this as a small pure helper function (e.g. `_extract_json_object(text: str) -> str`) so it's unit-testable without hitting the API — add `tests/eval/test_faithfulness.py` cases feeding it fenced JSON, prose-wrapped JSON, and plain JSON to confirm all three parse correctly.
- Only make this change if the live run actually fails on parsing — don't add speculative robustness for a failure mode that doesn't occur. If the first few live judge calls parse cleanly, leave `judge()` as-is and note in your report that this risk didn't materialize.

- [ ] **Step 3: Run the full backtest**

```bash
python scripts/backtest.py
```

Watch for:
- A crash in `forward_return()` (`src/backtest/score_signals.py`) around `dates.tz` / `filed_date.tz_localize(...)`. The existing guard is `if dates.tz is not None and filed_date.tz is None: filed_date = filed_date.tz_localize(dates.tz)` — this only handles the case where `yfinance`'s returned index IS tz-aware and the filing timestamp ISN'T. If `yfinance` ever returns a tz-**naive** index (behavior can vary by ticker/version), the current guard correctly no-ops rather than crashing, since both sides would already be comparable — but verify this by actually observing what `prices.index.tz` is for at least 2-3 of the 10 real tickers during this live run, and only add a fix if you observe an actual mismatch, not a hypothetical one.
- Confirm the filing dates flowing into the backtest are real EDGAR data, not inferred: `scripts/backtest.py`'s `list_filings()` reads `doc["filed_datetime"]` directly from `data/processed/*.json`, which was captured by `src/ingestion/parse.py`'s `parse_filing_dates()` from the SEC submission's `<ACCEPTANCE-DATETIME>` tag — this is already real EDGAR metadata, not hardcoded or inferred. Confirm this by spot-checking one `data/processed/*.json` file's `filed_datetime` against the corresponding SEC filing's actual acceptance timestamp (or at minimum, confirm it's a plausible ISO datetime, not a placeholder). Report this confirmation either way — don't silently assume it's fine.
- `data/processed/signals.jsonl` gets written with 10 real signals (one per filing) and a hit-rate summary prints. Given only 10 filings and the README's own honesty standard, do NOT be alarmed by a hit rate near 50% — report the actual number, don't editorialize about it here (that's Task 6's job).

- [ ] **Step 4: Re-run each of the three scripts one more time to confirm stability**

Not every live LLM call is deterministic — a single clean run doesn't fully rule out a rare parsing edge case. Run `scripts/run_signal.py` on 2 more tickers (e.g. `MSFT --quarter 2026Q2`, `NVDA --quarter 2026Q1`) to get 3 total clean runs across different tickers/quarters, per the brief's "both scripts run clean on at least 3 tickers" deliverable.

- [ ] **Step 5: Run the full test suite to confirm no regressions from any fixes made**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 6: Commit any fixes made**

If Steps 1-4 required no code changes (schema and parsing worked cleanly first try), skip this step — there's nothing to commit, and the report should say so plainly rather than inventing a defensive-code commit for a problem that didn't occur. If fixes were needed, `git add` exactly the modified files (never `-A`) and commit with a message describing the specific bug fixed, e.g.:

```bash
git commit -m "fix: flatten nested Signal schema for Anthropic tool-use compatibility"
# or
git commit -m "fix: make faithfulness judge JSON parsing defensive against fenced/prose-wrapped output"
```

---

### Task 2: Per-question error analysis

**Files:**
- Create: `scripts/error_analysis.py`
- Create (output, not committed — gitignored like other `data/` derived artifacts... check `.gitignore`; if `data/analysis/` isn't covered by an existing broad rule, add it): `data/analysis/error_analysis.csv`

**Interfaces:**
- Consumes: `src.retrieval.dense.dense_search`, `src.retrieval.hybrid.hybrid_search`, `src.retrieval.rerank.reranked_search` (all Week 2, merged), `config.GOLDEN_DIR / "eval_set.jsonl"` (20 questions), `src.eval.metrics.recall_at_k`.
- Produces: a per-question CSV/stdout table and worst-regression chunk dumps, testing the hypothesis that Recall@5 regressions are concentrated in multi-gold-chunk questions (rank-compression effect) rather than a general retrieval-quality failure.

- [ ] **Step 1: Check whether `data/analysis/` needs a `.gitignore` entry**

Run: `git check-ignore -q data/analysis 2>/dev/null; echo $?` (or just check `.gitignore`'s content directly). If `data/analysis/` isn't already covered, add a line for it (`data/analysis/` or `*.csv` under `data/` — match whatever convention the rest of `.gitignore` uses) and commit that change separately before writing the script, so the CSV output never gets accidentally staged.

- [ ] **Step 2: Implement `scripts/error_analysis.py`**

```python
"""Per-question error analysis across dense/hybrid/reranked retrieval.

Tests the hypothesis that Recall@5 regressions from Week 2's hybrid/rerank
additions are concentrated in golden-set questions with multiple relevant
chunks (a rank-compression effect: RRF/reranking promotes one strong chunk
to rank 1, raising MRR/NDCG, while pushing a second relevant chunk out of
the top-5 window) rather than a general retrieval-quality regression.

Usage:
    python scripts/error_analysis.py
"""
import csv
import json
import sys
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config
from src.eval.metrics import recall_at_k
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import reranked_search

K = 5
CONFIGS = [
    ("dense", dense_search),
    ("hybrid", hybrid_search),
    ("reranked", reranked_search),
]
OUT_CSV = ROOT / "data" / "analysis" / "error_analysis.csv"


def load_cases():
    with (config.GOLDEN_DIR / "eval_set.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def rank_of_first_hit(retrieved_ids: list[str], relevant_ids) -> int | None:
    relevant = set(relevant_ids)
    for i, r in enumerate(retrieved_ids, start=1):
        if r in relevant:
            return i
    return None


def analyze():
    cases = load_cases()
    rows = []

    for case in cases:
        relevant_ids = case["relevant_chunk_ids"]
        row = {
            "question": case["question"][:60],
            "n_relevant": len(relevant_ids),
        }
        hits_by_config = {}
        for name, search_fn in CONFIGS:
            hits = search_fn(case["question"], k=K)
            retrieved_ids = [h["chunk_id"] for h in hits]
            hits_by_config[name] = retrieved_ids
            row[f"recall_{name}"] = recall_at_k(retrieved_ids, relevant_ids, K)
            row[f"rank_of_first_hit_{name}"] = rank_of_first_hit(retrieved_ids, relevant_ids)

        row["delta_dense_to_hybrid"] = row["recall_hybrid"] - row["recall_dense"]
        row["_hits_by_config"] = hits_by_config  # not written to CSV, used for the dump below
        rows.append(row)

    rows.sort(key=lambda r: r["delta_dense_to_hybrid"])

    fieldnames = [
        "question", "n_relevant",
        "recall_dense", "recall_hybrid", "recall_reranked",
        "rank_of_first_hit_dense", "rank_of_first_hit_hybrid", "rank_of_first_hit_reranked",
        "delta_dense_to_hybrid",
    ]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fieldnames})

    print(f"{'question':<62} {'n_rel':>5} {'recall_d':>9} {'recall_h':>9} {'recall_r':>9} {'delta_h':>8}")
    for row in rows:
        print(
            f"{row['question']:<62} {row['n_relevant']:>5} "
            f"{row['recall_dense']:>9.2f} {row['recall_hybrid']:>9.2f} {row['recall_reranked']:>9.2f} "
            f"{row['delta_dense_to_hybrid']:>8.2f}"
        )

    print("\n=== grouped by n_relevant (the actual hypothesis test) ===")
    single_gold = [r for r in rows if r["n_relevant"] == 1]
    multi_gold = [r for r in rows if r["n_relevant"] > 1]
    for label, group in [("single-gold (n_relevant=1)", single_gold), ("multi-gold (n_relevant>1)", multi_gold)]:
        if not group:
            print(f"{label}: no questions in this group")
            continue
        print(f"{label} (n={len(group)}):")
        for name, _ in CONFIGS:
            print(f"  mean recall_{name}: {mean(r[f'recall_{name}'] for r in group):.3f}")

    print("\n=== 3 worst regressions (dense -> hybrid) ===")
    for row in rows[:3]:
        print(f"\nQuestion: {row['question']}")
        print(f"  delta_dense_to_hybrid: {row['delta_dense_to_hybrid']:.2f}")
        print(f"  dense top-5:  {row['_hits_by_config']['dense']}")
        print(f"  hybrid top-5: {row['_hits_by_config']['hybrid']}")

    print(f"\nWrote {len(rows)} rows to {OUT_CSV}")
    return rows


if __name__ == "__main__":
    analyze()
```

- [ ] **Step 3: Run it against the live index and read the output**

```bash
python scripts/error_analysis.py
```

This requires the `qdrant_data/` index and `data/processed/chunks.jsonl` already present in this checkout (confirmed present) — no live API key needed (pure retrieval, no generation). Record in your report: the grouped mean-recall numbers for single-gold vs. multi-gold questions per config, and whether they support or refute the rank-compression hypothesis. This is real analytical output the plan cannot predict — report what actually comes out, don't assume the hypothesis is confirmed.

- [ ] **Step 4: Commit**

```bash
git add scripts/error_analysis.py .gitignore
git commit -m "feat: add per-question error analysis for retrieval config comparison"
```

(Do not commit `data/analysis/error_analysis.csv` itself — it's derived, regenerable output, gitignored per Step 1.)

---

### Task 3: Recall@k sweep (k=1,3,5,10) across all three retrievers

**Files:**
- Modify: `src/eval/metrics_db.py` (add `k`, `recall_at_k`, `precision_at_k`, `ndcg_at_k` generic columns via migration)
- Modify: `src/eval/run_eval.py` (add a `sweep()` function; keep `evaluate()`'s existing signature and behavior for backward compatibility with `scripts/evaluate.py`)
- Create: `scripts/eval_sweep.py`
- Modify: `tests/eval/test_run_eval.py` (add a test for `sweep()`)

**Interfaces:**
- Consumes: `src.retrieval.{dense,bm25,hybrid,rerank}` search functions, `src.eval.metrics.*`.
- Produces: `sweep(search_fns: dict[str, callable], k_values: list[int] = [1, 3, 5, 10], config_hash_prefix: str = "", cases: list[dict] | None = None) -> dict[tuple[str, int], dict]` — logs one `eval_runs` row per (config, k) pair.

- [ ] **Step 1: Add the generic k-aware columns to `src/eval/metrics_db.py`**

Read the current file first (it already has the `ndcg_at_5` migration pattern from Week 2 — follow the same style). Add:

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
  notes TEXT,
  k INTEGER,
  recall_at_k REAL,
  precision_at_k REAL,
  ndcg_at_k REAL
);
"""


def get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.execute(SCHEMA)
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(eval_runs)")}
    if "ndcg_at_5" not in existing_cols:
        conn.execute("ALTER TABLE eval_runs ADD COLUMN ndcg_at_5 REAL")
    for col in ("k", "recall_at_k", "precision_at_k", "ndcg_at_k"):
        if col not in existing_cols:
            col_type = "INTEGER" if col == "k" else "REAL"
            conn.execute(f"ALTER TABLE eval_runs ADD COLUMN {col} {col_type}")
    conn.commit()
    return conn


def record_run(run_id: str, timestamp: str, config_hash: str, recall_at_5, precision_at_5,
                mrr, ndcg_at_5, faithfulness, notes: str,
                k: int | None = None, recall_at_k: float | None = None,
                precision_at_k: float | None = None, ndcg_at_k: float | None = None):
    conn = get_connection()
    with conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO eval_runs
                (run_id, timestamp, config_hash, recall_at_5, precision_at_5, mrr, ndcg_at_5,
                 faithfulness, notes, k, recall_at_k, precision_at_k, ndcg_at_k)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (run_id, timestamp, config_hash, recall_at_5, precision_at_5, mrr, ndcg_at_5,
             faithfulness, notes, k, recall_at_k, precision_at_k, ndcg_at_k),
        )
    conn.close()
```

Note `recall_at_5`/`precision_at_5`/`ndcg_at_5` stay REQUIRED-shaped in the call signature (existing callers pass them positionally/by-keyword already) but are now `None`-able in practice — callers that only care about a non-5 k pass `None` for these three and populate the new `*_at_k` fields plus `k` instead, so the legacy columns are never given a k≠5 value. Callers at k=5 (existing `evaluate()` calls) should populate BOTH the legacy fixed columns AND the new generic ones (with `k=5`) for continuity — that's the one case where both sets of columns get real, mutually-consistent values.

- [ ] **Step 2: Update `evaluate()` in `src/eval/run_eval.py` to also populate the generic columns when k=5**

Locate the existing `record_run(...)` call inside `evaluate()`. Change it to also pass `k=k, recall_at_k=metrics["recall_at_5"], precision_at_k=metrics["precision_at_5"], ndcg_at_k=metrics["ndcg_at_5"]` (reusing the already-computed values — `evaluate()`'s own metric dict keys stay as `recall_at_5` etc. for backward compatibility with `scripts/evaluate.py` and `tests/eval/test_run_eval.py`, which are NOT being changed by this task; only the DB write gets richer). Do not rename `evaluate()`'s return dict keys — that would break Week 2's `scripts/evaluate.py` and its test for no benefit.

- [ ] **Step 3: Add `sweep()` to `src/eval/run_eval.py`**

```python
def sweep(search_fns: dict, k_values: list[int] = None, config_hash_prefix: str = "",
          cases: list[dict] | None = None) -> dict:
    """Run evaluate() for every (config_name, k) pair in search_fns x k_values.

    search_fns: {"dense": dense_search, "hybrid": hybrid_search, "reranked": reranked_search}
    Returns {(config_name, k): metrics_dict}.
    """
    k_values = k_values if k_values is not None else [1, 3, 5, 10]
    cases = cases if cases is not None else load_eval_set()

    results = {}
    for name, search_fn in search_fns.items():
        for k in k_values:
            config_hash = f"{config_hash_prefix}{name}-k{k}" if config_hash_prefix else f"{name}-k{k}"
            print(f"\n=== {name} @ k={k} ===")
            metrics = evaluate(search_fn, k=k, config_hash=config_hash,
                                notes=f"sweep: {name} at k={k}", cases=cases)
            results[(name, k)] = metrics
    return results
```

Note: `evaluate()`'s current `record_run` call (after Step 2) always writes `recall_5=metrics["recall_5"]`-style legacy columns regardless of what `k` actually is — **this needs one more fix**: legacy columns should only get real values when `k == 5`, otherwise `None`. Adjust the `record_run(...)` call inside `evaluate()` so the legacy `recall_at_5`/`precision_at_5`/`ndcg_at_5` arguments are `metrics["recall_at_5"] if k == 5 else None` (and the same pattern for the other two) — this is the actual mechanism that prevents `sweep()`'s k=1/3/10 runs from polluting the k=5-only legacy columns.

- [ ] **Step 4: Write the failing test for `sweep()`**

```python
# add to tests/eval/test_run_eval.py
def test_sweep_runs_every_config_k_pair_and_records_each(monkeypatch):
    cases = [{"question": "q1", "relevant_chunk_ids": ["A"]}]

    def fake_search(question, k=5):
        return [{"chunk_id": cid} for cid in ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J"]]

    recorded = []

    def fake_record_run(**kwargs):
        recorded.append(kwargs)

    monkeypatch.setattr(run_eval_mod, "record_run", fake_record_run)

    results = run_eval_mod.sweep({"fake": fake_search}, k_values=[1, 5], cases=cases)

    assert set(results.keys()) == {("fake", 1), ("fake", 5)}
    assert len(recorded) == 2
    # k=1: "A" still rank-1 hit -> recall=1.0 at k=1 too
    assert results[("fake", 1)]["recall_at_5"] == 1.0
    # legacy columns only populated for the k=5 call
    k1_call = next(r for r in recorded if r["k"] == 1)
    k5_call = next(r for r in recorded if r["k"] == 5)
    assert k1_call["recall_at_5"] is None
    assert k5_call["recall_at_5"] == 1.0
    assert k1_call["recall_at_k"] == 1.0
    assert k1_call["k"] == 1
```

Run: `python -m pytest tests/eval/test_run_eval.py -v` — expect this new test to FAIL first (AttributeError: no `sweep`), confirming TDD RED, then implement Steps 1-3 (if not already done) and re-run for GREEN.

- [ ] **Step 5: Create `scripts/eval_sweep.py`**

```python
"""CLI: run the Recall@k sweep (k=1,3,5,10) across dense/hybrid/reranked,
logging every (config, k) run to src/eval/metrics.db.

Usage:
    python scripts/eval_sweep.py
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.eval.run_eval import sweep
from src.retrieval.dense import dense_search
from src.retrieval.hybrid import hybrid_search
from src.retrieval.rerank import reranked_search

if __name__ == "__main__":
    results = sweep(
        {"dense": dense_search, "hybrid": hybrid_search, "reranked": reranked_search},
        k_values=[1, 3, 5, 10],
    )

    print("\n=== sweep summary ===")
    print(f"{'config':<10} {'k':>3} {'recall':>8} {'precision':>10} {'mrr':>8} {'ndcg':>8}")
    for (name, k), m in sorted(results.items(), key=lambda item: (item[0][0], item[0][1])):
        print(f"{name:<10} {k:>3} {m['recall_at_5']:>8.3f} {m['precision_at_5']:>10.3f} "
              f"{m['mrr']:>8.3f} {m['ndcg_at_5']:>8.3f}")
```

- [ ] **Step 6: Run the sweep live and record the numbers**

```bash
python scripts/eval_sweep.py
```

Expected: 12 rows printed (3 configs × 4 k-values) and 12 new rows in `eval_runs`. **Write down the full table — Task 6 (README) needs it**, specifically whether hybrid's Recall@10 beats dense's Recall@10 (the brief's key hypothesis test: "if hybrid loses at k=5 but wins at k=10, that confirms rank compression").

- [ ] **Step 7: Run the full test suite**

```bash
python -m pytest tests/ -v
```

- [ ] **Step 8: Commit**

```bash
git add src/eval/metrics_db.py src/eval/run_eval.py scripts/eval_sweep.py tests/eval/test_run_eval.py
git commit -m "feat: add Recall@k sweep (k=1,3,5,10) across all retrieval configs"
```

---

### Task 4: `scripts/suggest_questions.py` — golden-set candidate drafting

**Files:**
- Create: `scripts/suggest_questions.py`

**Interfaces:**
- Consumes: `data/processed/chunks.jsonl`, live Anthropic API (`ANTHROPIC_API_KEY` — same blocking dependency as Task 1).
- Produces: `data/golden/candidates.jsonl`, each line `{"question": str, "chunk_id": str, "ticker": str, "section": str}` — candidates only, no `relevant_chunk_ids` field pre-filled with the source chunk as if it were verified gold (that would be the circular-labeling problem the brief explicitly warns against). The human labeler reviews each candidate, decides the real relevant chunk_id(s) (usually but not always the source chunk), and manually appends verified entries to `data/golden/eval_set.jsonl` in the existing format.

- [ ] **Step 1: Implement `scripts/suggest_questions.py`**

```python
"""Draft candidate golden-set questions for human labeling.

Samples chunks stratified across ticker and section, asks Claude to draft
2 candidate questions each chunk would answer, and writes candidates to
data/golden/candidates.jsonl for a human to review and label.

IMPORTANT: these are candidates only. A human must read each one, decide
the real relevant_chunk_id(s) by hand, and add verified entries to
data/golden/eval_set.jsonl in that file's existing format. An AI-drafted
question paired with an AI-assumed answer chunk is circular and would
invalidate the eval set -- this script never writes to eval_set.jsonl.

Usage:
    python scripts/suggest_questions.py --n-chunks 30
"""
import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import Anthropic

import config

MODEL = "claude-opus-5"
CANDIDATES_PATH = config.GOLDEN_DIR / "candidates.jsonl"

DRAFT_SYSTEM = """You are drafting evaluation questions for a retrieval system over
earnings-release excerpts. Given one excerpt, write exactly 2 specific,
answerable questions that this excerpt (and ideally only this excerpt)
would answer. Questions should reference concrete facts, numbers, or
statements from the excerpt -- not generic questions.

Return ONLY JSON: {"questions": ["...", "..."]}"""


def load_chunks():
    with (config.PROCESSED_DIR / "chunks.jsonl").open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def stratified_sample(chunks: list[dict], n: int, seed: int = 42) -> list[dict]:
    """Sample roughly evenly across (ticker, section) groups."""
    groups = defaultdict(list)
    for c in chunks:
        groups[(c["ticker"], c["section"])].append(c)

    rng = random.Random(seed)
    for group in groups.values():
        rng.shuffle(group)

    sampled = []
    group_keys = list(groups.keys())
    rng.shuffle(group_keys)
    i = 0
    while len(sampled) < n and any(groups[k] for k in group_keys):
        key = group_keys[i % len(group_keys)]
        if groups[key]:
            sampled.append(groups[key].pop())
        i += 1
    return sampled[:n]


def draft_questions_for_chunk(client: Anthropic, chunk: dict) -> list[str]:
    response = client.messages.create(
        model=MODEL,
        max_tokens=500,
        system=DRAFT_SYSTEM,
        messages=[{"role": "user", "content": f"EXCERPT:\n{chunk['text']}"}],
    )
    text = next(b.text for b in response.content if b.type == "text").strip()
    text = text.removeprefix("```json").removesuffix("```").strip()
    return json.loads(text)["questions"]


def suggest(n_chunks: int = 30):
    chunks = load_chunks()
    sampled = stratified_sample(chunks, n_chunks)
    print(f"Sampled {len(sampled)} chunks (stratified by ticker/section) from {len(chunks)} total")

    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    candidates = []
    for chunk in sampled:
        try:
            questions = draft_questions_for_chunk(client, chunk)
        except Exception as e:
            print(f"  SKIP {chunk['chunk_id']}: {e}")
            continue
        for q in questions:
            candidates.append({
                "question": q,
                "chunk_id": chunk["chunk_id"],
                "ticker": chunk["ticker"],
                "section": chunk["section"],
            })
        print(f"  {chunk['chunk_id']}: {len(questions)} candidate(s)")

    config.GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    with CANDIDATES_PATH.open("w", encoding="utf-8") as f:
        for c in candidates:
            f.write(json.dumps(c) + "\n")
    print(f"\nWrote {len(candidates)} candidates -> {CANDIDATES_PATH}")
    print("Next: review each candidate by hand, verify/correct the relevant chunk_id(s),")
    print("and add confirmed entries to data/golden/eval_set.jsonl in its existing format.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Draft candidate golden-set questions for human labeling.")
    parser.add_argument("--n-chunks", type=int, default=30, help="number of chunks to sample")
    args = parser.parse_args()
    suggest(n_chunks=args.n_chunks)
```

- [ ] **Step 2: Run it (requires the live API key from Task 1)**

```bash
python scripts/suggest_questions.py --n-chunks 30
```

30 chunks × 2 questions each ≈ 60 candidates is enough headroom to hand-pick ~30 new verified questions (20 existing + 30 new = 50 target). Adjust `--n-chunks` up if the hand-review pass rejects more candidates than expected.

- [ ] **Step 3: Commit the script (not its output)**

```bash
git add scripts/suggest_questions.py
git commit -m "feat: add golden-set candidate-question drafting script"
```

`data/golden/candidates.jsonl` is scratch for the human labeling pass — do not commit it. Confirm it's covered by an existing `.gitignore` pattern (`data/golden/` isn't currently ignored since `eval_set.jsonl` IS committed — add a specific `data/golden/candidates.jsonl` line rather than ignoring the whole directory) as part of this commit.

- [ ] **Step 4: Hand off the labeling step to the user**

This step cannot be automated — per the brief, "the human labels the real gold chunk IDs by hand." After Step 2 produces candidates, tell the user the candidates file is ready for review and ask them to confirm when `data/golden/eval_set.jsonl` has reached 50 entries (or however many they're comfortable hand-labeling) before Task 6 (README) uses the expanded set's numbers. Do not fabricate or auto-approve labels to hit the 50-question target faster.

---

### Task 5: Corpus expansion (5 tickers×2 quarters → ~20 tickers×4 quarters)

**Files:**
- No new files — reruns existing `scripts/ingest_all.py` (fetch → parse → chunk → embed) with a wider ticker list and quarter count.
- Possibly modify: `src/ingestion/chunk.py`'s `HEADER_PATTERNS` (only if new tickers' filings use section headers the current regex list doesn't recognize).

**Interfaces:**
- Consumes: `src.ingestion.fetch.fetch_all(tickers, quarters)`, `src.ingestion.parse.parse_all`, `src.ingestion.chunk.chunk_all`, `src.ingestion.embed.embed_and_load` — all existing, unchanged Week 1 code.
- Produces: an expanded `data/processed/chunks.jsonl` and `qdrant_data/` index; a log of which filings (if any) the parser failed to extract sections from, per ticker.

This task depends on Tasks 1-4 being done first (per the brief's explicit ordering) and involves real network calls to SEC EDGAR across ~4x the current filing volume — expect it to take meaningfully longer than Week 1's original ingestion and to surface parser edge cases Week 1's narrower 5-ticker sample didn't hit.

- [ ] **Step 1: Choose the expanded ticker list with the user**

The brief doesn't name the 15 additional tickers. Propose a reasonable set spanning a few sectors beyond the current tech-heavy 5 (AAPL, MSFT, NVDA, AMZN, GOOGL) — e.g. adding names like META, TSLA, JPM, V, UNH, XOM, WMT, HD, PG, KO, DIS, NFLX, CRM, ORCL, AVGO (15 more → 20 total) — but confirm this list with the user before fetching, since it shapes every downstream number (retrieval eval, backtest) and they may have specific coverage preferences (sector balance, market cap range, personal interest).

- [ ] **Step 2: Run ingestion with the expanded list and 4 quarters**

```bash
python scripts/ingest_all.py --tickers AAPL MSFT NVDA AMZN GOOGL <15 more> --quarters 4
```

Note `src/ingestion/fetch.py`'s `MIN_CANDIDATE_LIMIT = 8` and `candidate_limit = max(MIN_CANDIDATE_LIMIT, quarters * 4)` — at `quarters=4` this already over-fetches 16 candidates per ticker to account for non-earnings 8-Ks, consistent with the existing design; no code change needed there.

- [ ] **Step 3: Confirm the parser's section regexes hold — log failures, don't silently drop them**

`src/ingestion/parse.py` already prints `SKIP {ticker}/{accession}: not an earnings 8-K` or `SKIP ... no EX-99.1 found` for filings it can't use — this existing behavior already satisfies "log which ones rather than silently dropping them" at the filing level. What's NOT currently logged: `src/ingestion/chunk.py`'s `split_into_sections()` silently falls back to `DEFAULT_SECTION = "PreparedRemarks"` for any text that doesn't match a known header pattern — for a *new* ticker with a differently-formatted press release, this could mean most of its content gets dumped into one oversized `PreparedRemarks` chunk sequence instead of being properly split into `BusinessOutlook`/`ForwardLooking`/`FinancialTables` etc.

After running Step 2, spot-check: for each of the 15 new tickers, look at `data/processed/{TICKER}_*.json`'s parsed text and compare the resulting chunks' `section` distribution (`python -c "import json; ..."` counting sections per ticker from `chunks.jsonl`) against the 5 original tickers' typical distribution. A new ticker whose chunks are ~90%+ `PreparedRemarks` with almost no `FinancialTables`/`BusinessOutlook` chunks likely has an unrecognized header format — if you find this, add the missing pattern(s) to `HEADER_PATTERNS` in `src/ingestion/chunk.py` (following the existing tuple format: `(regex, section_label, is_table)`) and re-run chunking (`python -m src.ingestion.chunk`) for that ticker's data. Report which tickers (if any) needed a pattern addition, and which patterns were added.

- [ ] **Step 4: Re-run the retrieval eval to confirm metrics hold at the larger corpus size**

```bash
python scripts/evaluate.py
python scripts/eval_sweep.py
```

The existing 20 (or 50, if Task 4's labeling finished) golden questions still target specific `chunk_id`s from the original 5-ticker corpus — those should still resolve correctly since chunking is deterministic and ticker/quarter-scoped, but a larger corpus means more candidate chunks compete for the same top-5 slots, which could shift Recall@5 even for unrelated questions if embedding similarity ties are broken differently. **Report whether the metrics moved meaningfully from the Week 2 baseline (0.775/0.750/0.725) purely from added corpus size, separate from any golden-set expansion effect from Task 4** — this distinction matters for how Task 6 explains the final numbers.

- [ ] **Step 5: Re-run the backtest with the larger signal count**

```bash
python scripts/backtest.py
```

Now ~80 filings × across ~20 tickers × 4 quarters (exact count depends on how many 8-Ks per ticker turn out to be genuine earnings releases) generates a much more reportable n for the backtest than Week 2's n=10. **Write down the new n, hit rate, and mean return — Task 6 needs them.**

- [ ] **Step 6: Commit**

```bash
git add src/ingestion/chunk.py   # only if HEADER_PATTERNS changed
git commit -m "feat: expand corpus to <N> tickers x 4 quarters, log parser section-pattern gaps"
```

(`data/processed/`, `qdrant_data/`, `data/raw/` are either gitignored or, for `data/processed/`, were already committed in Week 1/2 — check current tracking status of `data/processed/*.json` and `chunks.jsonl` with `git status` before staging; if they're tracked, include the regenerated versions in this commit so the repo's checked-in corpus matches what the reported numbers were computed against.)

---

### Task 6: README rewrite with final numbers

**Files:**
- Modify: `README.md`

**Interfaces:**
- Consumes: real numbers from Tasks 1 (live verification confirmation), 2 (error analysis grouped results), 3 (Recall@k sweep table), 4/5 (expanded golden-set size and corpus size), and the existing Week 1/2 content already in `README.md`.

This is the last task — do not start it until Tasks 1-5 have produced real numbers to report. Do not write this section with placeholder or projected numbers.

- [ ] **Step 1: Restructure `README.md` per the brief's specified order**

Read the full current `README.md` first (it has real Week 1 and Week 2 sections already). Reorganize/rewrite into this order, folding in the existing honest Week 1/2 content rather than duplicating it wholesale:

1. **One paragraph: why should this work?** — reuse/tighten the existing "Why should this text predict..." paragraph from Week 1.
2. **Results table first** — Recall@k (k=1,3,5,10 from Task 3's sweep), MRR, NDCG@5, and faithfulness (from Task 1's live verification), across dense/hybrid/reranked, using the FINAL corpus/golden-set size (post-Task 4/5) if those completed, or clearly labeled as "pre-expansion (n=20 questions, 10 filings)" numbers if Task 4/5 didn't fully complete before this task ran — never blend numbers from different corpus sizes without labeling which is which.
3. **The negative result, prominently** — state plainly that hybrid and reranking did not clear the pre-registered "must improve Recall@5" bar, so dense stays the default retriever (unless Task 3's k=10 sweep genuinely changed this conclusion — if so, say so explicitly and explain the rank-compression finding from Task 2 as the mechanism, with the specific single-gold vs. multi-gold mean-recall numbers from `scripts/error_analysis.py`'s output).
4. **Architecture diagram** — simple ASCII or Mermaid diagram: `ingestion → retrieval → generation → eval/backtest`, showing the module names (`src/ingestion/`, `src/retrieval/`, `src/generation/`, `src/eval/`, `src/backtest/`).
5. **Design decisions and why** — carry forward Week 2's existing bullets (RRF over score-averaging, `min_length=1` citations, `filed_datetime`-derived lookahead-bias avoidance, `fiscal_quarter` filtering) plus anything new from this week (e.g. the schema-flattening fix from Task 1, if it happened; the stratified-sampling approach in `suggest_questions.py`).
6. **Honest limitations** — update the sample-size caveat with the ACTUAL final golden-set size and backtest n (not the old n=20/n=10 if Tasks 4/5 completed), note the faithfulness judge was validated against however many hand-checked examples actually happened in Task 1, no transaction costs modeled, single data source (SEC EDGAR only).

- [ ] **Step 2: Verify the numbers in the README against their actual sources**

Before committing, spot-check at least 3 numbers in the new README table against the actual `eval_runs` rows in `src/eval/metrics.db` or the actual printed output captured during Tasks 2/3/5 — do not transcribe from memory of what an earlier task's report claimed without re-confirming, since report claims are the same "trust but verify" standard that applied to every subagent's report throughout Weeks 2-3.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: rewrite README with Week 3 results (error analysis, k-sweep, expanded corpus)"
```

---

### Task 7: Housekeeping — orphaned worktree cleanup

**Files:** none (filesystem cleanup only, not a code change)

- [ ] **Step 1: Attempt cleanup**

```bash
git worktree list
git worktree prune
```

If `.claude/worktrees/earnings-whisper-week2/` still shows as a stale/orphaned directory on disk (git's own tracking was already confirmed clean as of the end of Week 2 — this is a pure filesystem lock issue, not a git-state issue), attempt:

```bash
rm -rf "C:\Users\Shreyas Kakkar\Documents\GitHub\EarningsWhisper\.claude\worktrees\earnings-whisper-week2"
```

- [ ] **Step 2: If the lock persists, leave it and note it**

Per the brief: "if the lock persists, leave it and note it — it's harmless and clears after a restart." Do not escalate this (no forced unlock tools, no process-killing) — report the outcome either way (removed, or still locked) and move on. This step never blocks any other task.

---

## Self-Review Notes

- **Spec coverage:** All 7 brief tasks map 1:1 to Tasks 1-7 above, in the brief's own specified order.
- **Deviations from the brief, and why:** (1) The brief's background section describes the corpus as "MD&A / Risk Factors" from 10-Qs; the actual codebase uses 8-K earnings-release sections — every task above uses the real section/module names. (2) The brief says an API key is "now in .env" — verified false during planning; Task 1 Step 0 makes this an explicit checkpoint rather than assuming it's ready. (3) Task 3's SQLite migration design (generic `k`/`*_at_k` columns, legacy `*_at_5` columns only populated when k==5) is a concrete resolution of the brief's open-ended "pick one, do it cleanly" instruction. (4) Task 5's exact expanded-ticker list isn't specified by the brief — Task 5 Step 1 confirms it with the user rather than guessing.
- **Type/signature consistency:** `sweep()`'s `search_fns` dict values must match the existing `search_fn(query, k) -> list[dict]` contract already used by `evaluate()` and all four retrieval modules — confirmed consistent. `scripts/error_analysis.py` reuses `recall_at_k` from `src/eval/metrics.py` (Week 2) with the same `(retrieved_ids, relevant_ids, k)` signature. No placeholder code blocks remain in any task above except where a step is explicitly procedural/investigative (Task 1's live debugging, Task 5's ticker-list confirmation and section-pattern spot-check) rather than a code-writing step.
