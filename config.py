"""Central settings, loaded from .env (or the environment)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")

ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")

LLM_BACKEND = os.environ.get("LLM_BACKEND", "gemini")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
GEMINI_MODEL = os.environ.get("GEMINI_MODEL", "gemini-3.1-flash-lite")

TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "V", "UNH",
    "XOM", "WMT", "HD", "PG", "KO",
    "DIS", "NFLX", "CRM", "ORCL", "AVGO",
]

# Qdrant: local on-disk mode by default (no server needed). Set QDRANT_HOST
# to point at a real Qdrant server instead (e.g. once Docker is available).
QDRANT_PATH = os.environ.get("QDRANT_PATH", str(ROOT / "qdrant_data"))
QDRANT_HOST = os.environ.get("QDRANT_HOST", "")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")

# SEC EDGAR requires a descriptive User-Agent identifying the requester.
SEC_EDGAR_COMPANY_NAME = os.environ.get("SEC_EDGAR_COMPANY_NAME", "EarningsWhisper Research")
SEC_EDGAR_EMAIL = os.environ.get("SEC_EDGAR_EMAIL", "svk6639@psu.edu")

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
GOLDEN_DIR = DATA_DIR / "golden"
