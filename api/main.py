"""Chat endpoint bridging the RAG pipeline to the frontend.

Routes each message to either open Q&A (dense retrieval + grounded answer)
or a directional call (the structured signal-generation pipeline).
"""
import json
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import config
from config import TICKERS
from src.generation.generate import generate_signal
from src.generation.llm_client import get_client
from src.ingestion.chunk import calendar_quarter
from src.retrieval.rerank import reranked_search

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],  # Vite dev server
    allow_methods=["*"],
    allow_headers=["*"],
)

client = get_client()
sessions: dict[str, list[dict]] = {}  # session_id -> message history, in-memory for v1


class ChatRequest(BaseModel):
    session_id: str | None = None
    ticker: str
    message: str


ROUTER_PROMPT = """Classify the user's message as exactly one word:
"directional" — if they're asking whether a stock will go up/down, for an
  outlook, a buy/sell take, or a signal/prediction
"question" — anything else (asking what was said, why, details, comparisons)
Respond with only the single word."""

DIRECTIONAL_TRIGGERS = [
    "going to go up", "going to go down", "will it go up",
    "should i buy", "should i sell", "bullish", "bearish",
    "worth buying",
]


def route(message: str) -> str:
    # cheap keyword pass first — skips an LLM call on the obvious cases.
    # Deliberately narrow (full phrases, not bare words like "outlook" or
    # "signal") -- those appear constantly in ordinary grounded Q&A ("what
    # was management's outlook on margins?" is ordinary Q&A, not a buy/sell
    # ask), so a bare-word trigger misrouted them into a directional call.
    lowered = message.lower()
    if any(t in lowered for t in DIRECTIONAL_TRIGGERS):
        return "directional"
    # ambiguous — let the model decide
    verdict = client.text_call(ROUTER_PROMPT, message).strip().lower()
    return "directional" if "directional" in verdict else "question"


def latest_fiscal_quarter(ticker: str) -> str | None:
    """Most recent ingested fiscal_quarter for `ticker`, by filed_datetime.

    generate_signal(ticker, fiscal_quarter=None) retrieves across every
    ingested quarter for that ticker and blends them into one signal --
    exactly the failure mode this project's own README documents as wrong
    (one signal per filing, not one signal per ticker). Chat has no field
    for the caller to specify a quarter, so default to the latest one.
    """
    latest = None
    for doc_path in config.PROCESSED_DIR.glob(f"{ticker}_*.json"):
        doc = json.loads(doc_path.read_text(encoding="utf-8"))
        if doc["ticker"] != ticker:
            continue
        if latest is None or doc["filed_datetime"] > latest["filed_datetime"]:
            latest = doc
    return calendar_quarter(latest["filing_date"]) if latest else None


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": req.message})

    kind = route(req.message)

    if kind == "directional":
        signal, _chunks = generate_signal(req.ticker, fiscal_quarter=latest_fiscal_quarter(req.ticker))
        reply = {
            "type": "signal",
            "session_id": session_id,
            "signal": signal.model_dump(),
        }
    else:
        hits = reranked_search(req.message, k=5, ticker=req.ticker)
        context = "\n\n".join(f"[{h['chunk_id']}] {h['text']}" for h in hits)
        answer = client.text_call(
            "Answer the question using ONLY the excerpts. Cite chunk_ids you used in brackets. "
            "If the excerpts don't cover it, say so plainly.",
            f"EXCERPTS:\n{context}\n\nQUESTION:\n{req.message}",
        )
        reply = {
            "type": "answer",
            "session_id": session_id,
            "text": answer,
            "citations": [{"chunk_id": h["chunk_id"], "section": h["section"]} for h in hits],
        }

    history.append({"role": "assistant", "content": reply})
    return reply


@app.get("/tickers")
def tickers():
    return {"tickers": TICKERS}
