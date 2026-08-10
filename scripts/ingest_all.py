"""CLI: download -> parse -> chunk -> embed, end to end.

Usage:
    python scripts/ingest_all.py --tickers AAPL MSFT NVDA AMZN GOOGL --quarters 2
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion.fetch import DEFAULT_TICKERS, DEFAULT_QUARTERS, fetch_all
from src.ingestion.parse import parse_all
from src.ingestion.chunk import chunk_all
from src.ingestion.embed import embed_and_load


def main():
    parser = argparse.ArgumentParser(description="Download, parse, chunk, and embed earnings filings.")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS)
    parser.add_argument("--quarters", type=int, default=DEFAULT_QUARTERS)
    args = parser.parse_args()

    print(f"=== 1/4 fetch ({', '.join(args.tickers)}) ===")
    fetch_all(tickers=args.tickers, quarters=args.quarters)

    print("\n=== 2/4 parse ===")
    parse_all(keep_per_ticker=args.quarters)

    print("\n=== 3/4 chunk ===")
    chunk_all()

    print("\n=== 4/4 embed ===")
    embed_and_load()

    print("\nDone. Run `python scripts/evaluate.py` to check retrieval quality.")


if __name__ == "__main__":
    main()
