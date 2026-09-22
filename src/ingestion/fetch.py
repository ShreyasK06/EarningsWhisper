"""Download 8-K filings (with Exhibit 99.1 earnings press releases / prepared
remarks) for a fixed set of tickers via SEC EDGAR.

Usage:
    python scripts/ingest_all.py --tickers AAPL MSFT NVDA AMZN GOOGL --quarters 2
    python src/ingestion/fetch.py   # standalone, uses the defaults below
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sec_edgar_downloader import Downloader

import config

DEFAULT_TICKERS = config.TICKERS
DEFAULT_QUARTERS = 2

# Not every 8-K is earnings-related (some are officer changes, other events,
# etc.) -- over-fetch candidates per ticker; parse.py keeps only the ones with
# an EX-99.1 earnings exhibit and takes the N most recent of those. Roughly
# half of a company's 8-Ks are earnings-related, so 4x the requested quarters
# (floor 8) is a reasonable candidate pool.
MIN_CANDIDATE_LIMIT = 8


def fetch_all(tickers: list[str] = None, quarters: int = DEFAULT_QUARTERS):
    tickers = tickers or DEFAULT_TICKERS
    candidate_limit = max(MIN_CANDIDATE_LIMIT, quarters * 4)
    config.RAW_DIR.mkdir(parents=True, exist_ok=True)
    dl = Downloader(config.SEC_EDGAR_COMPANY_NAME, config.SEC_EDGAR_EMAIL, config.RAW_DIR)

    for ticker in tickers:
        print(f"Fetching up to {candidate_limit} 8-K filings for {ticker}...")
        try:
            dl.get("8-K", ticker, limit=candidate_limit, download_details=True)
        except Exception as e:
            print(f"  FAILED for {ticker}: {e}")


if __name__ == "__main__":
    fetch_all()
