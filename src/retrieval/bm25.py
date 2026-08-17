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
