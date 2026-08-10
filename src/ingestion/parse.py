"""Extract the EX-99.1 exhibit (earnings press release / prepared remarks)
from each downloaded 8-K SGML submission and clean it to plain text.

Reads:  data/raw/sec-edgar-filings/{TICKER}/8-K/{accession}/full-submission.txt
Writes: data/processed/{TICKER}_{accession}.json
"""

import json
import re
import sys
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import config

RAW_DIR = config.RAW_DIR / "sec-edgar-filings"
PROCESSED_DIR = config.PROCESSED_DIR

DOCUMENT_RE = re.compile(r"<DOCUMENT>(.*?)</DOCUMENT>", re.DOTALL)
TYPE_RE = re.compile(r"<TYPE>([^\n<]+)")
TEXT_RE = re.compile(r"<TEXT>(.*?)(?:</TEXT>|\Z)", re.DOTALL)
FILED_RE = re.compile(r"FILED AS OF DATE:\s*(\d{8})")
ACCEPTED_RE = re.compile(r"<ACCEPTANCE-DATETIME>(\d{14})")
ITEM_INFO_RE = re.compile(r"ITEM INFORMATION:\s*([^\n]+)")


def is_earnings_8k(raw_text: str) -> bool:
    items = ITEM_INFO_RE.findall(raw_text)
    return any("results of operations" in item.lower() for item in items)


def extract_ex99_1(raw_text: str) -> str | None:
    for match in DOCUMENT_RE.finditer(raw_text):
        block = match.group(1)
        type_match = TYPE_RE.search(block)
        if type_match and type_match.group(1).strip().upper().startswith("EX-99.1"):
            text_match = TEXT_RE.search(block)
            if text_match:
                return text_match.group(1)
    return None


def html_to_clean_text(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = soup.get_text(separator="\n")
    lines = [line.strip() for line in text.splitlines()]
    lines = [line for line in lines if line]
    return "\n".join(lines)


def parse_filing_dates(raw_text: str) -> tuple[str, str]:
    filed = FILED_RE.search(raw_text)
    accepted = ACCEPTED_RE.search(raw_text)
    filing_date = ""
    filed_datetime = ""
    if filed:
        filing_date = datetime.strptime(filed.group(1), "%Y%m%d").date().isoformat()
    if accepted:
        filed_datetime = datetime.strptime(
            accepted.group(1), "%Y%m%d%H%M%S"
        ).isoformat()
    return filing_date, filed_datetime


DEFAULT_KEEP_PER_TICKER = 2


def parse_all(keep_per_ticker: int = DEFAULT_KEEP_PER_TICKER):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    candidates: dict[str, list[dict]] = {}

    for submission_path in sorted(RAW_DIR.glob("*/8-K/*/full-submission.txt")):
        ticker = submission_path.parts[-4]
        accession = submission_path.parts[-2]
        raw_text = submission_path.read_text(encoding="utf-8", errors="replace")

        if not is_earnings_8k(raw_text):
            print(f"  SKIP {ticker}/{accession}: not an earnings 8-K (no Item 2.02)")
            continue

        ex99_html = extract_ex99_1(raw_text)
        if not ex99_html:
            print(f"  SKIP {ticker}/{accession}: earnings 8-K but no EX-99.1 found")
            continue

        clean_text = html_to_clean_text(ex99_html)
        filing_date, filed_datetime = parse_filing_dates(raw_text)

        candidates.setdefault(ticker, []).append(
            {
                "ticker": ticker,
                "accession": accession,
                "filing_date": filing_date,
                "filed_datetime": filed_datetime,
                "text": clean_text,
            }
        )

    count = 0
    for ticker, filings in candidates.items():
        filings.sort(key=lambda f: f["filing_date"], reverse=True)
        kept = filings[:keep_per_ticker]
        dropped = len(filings) - len(kept)
        if dropped:
            print(f"  {ticker}: keeping {len(kept)} most recent, dropping {dropped} older earnings 8-Ks")

        for out in kept:
            out_path = PROCESSED_DIR / f"{out['ticker']}_{out['accession']}.json"
            out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")
            print(f"  wrote {out_path.name} ({len(out['text'])} chars)")
            count += 1

    print(f"Parsed {count} filings.")


if __name__ == "__main__":
    parse_all()
