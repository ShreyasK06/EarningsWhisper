# scripts/backtest.py
"""CLI: generate a signal for every ingested filing, backtest against
realized forward returns, print the results table.

Requires GEMINI_API_KEY in .env and internet access for yfinance.

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
    skipped = []
    with SIGNALS_PATH.open("w", encoding="utf-8") as f:
        for filing in list_filings():
            ticker, quarter = filing["ticker"], filing["fiscal_quarter"]
            print(f"generating signal for {ticker} {quarter} ({filing['filed_datetime']})...")
            try:
                signal, _ = generate_signal(ticker, fiscal_quarter=quarter)
            except Exception as e:
                print(f"  SKIP {ticker} {quarter}: {type(e).__name__}: {e}")
                skipped.append((ticker, quarter))
                continue
            s = {
                "ticker": ticker,
                "fiscal_quarter": quarter,
                "filed_datetime": filing["filed_datetime"],
                "signal": signal.signal,
                "confidence": signal.confidence,
            }
            signals.append(s)
            f.write(json.dumps(s) + "\n")
            f.flush()

    print(f"\nwrote {len(signals)} signals -> {SIGNALS_PATH}")
    if skipped:
        print(f"skipped {len(skipped)} filings after a generation failure: {skipped}")

    df = score(signals)
    print(f"\nNote: n={len(df)} signals is not a statistically meaningful sample -- "
          f"directional hit rate here is a first number, not a conclusion.")
    return df


if __name__ == "__main__":
    run()
