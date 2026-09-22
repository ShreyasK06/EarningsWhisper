"""Chat endpoint bridging the RAG pipeline to the frontend.

Routes each message to either open Q&A (dense retrieval + grounded answer)
or a directional call (the structured signal-generation pipeline).
"""
import sys
import uuid
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import TICKERS
from src.generation.generate import generate_signal
from src.generation.llm_client import get_client
from src.retrieval.dense import dense_search

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
    "outlook", "signal", "prediction", "worth buying",
]


def route(message: str) -> str:
    # cheap keyword pass first — skips an LLM call on the obvious cases
    lowered = message.lower()
    if any(t in lowered for t in DIRECTIONAL_TRIGGERS):
        return "directional"
    # ambiguous — let the model decide
    verdict = client.text_call(ROUTER_PROMPT, message).strip().lower()
    return "directional" if "directional" in verdict else "question"


@app.post("/chat")
def chat(req: ChatRequest):
    session_id = req.session_id or str(uuid.uuid4())
    history = sessions.setdefault(session_id, [])
    history.append({"role": "user", "content": req.message})

    kind = route(req.message)

    if kind == "directional":
        signal, _chunks = generate_signal(req.ticker)
        reply = {
            "type": "signal",
            "session_id": session_id,
            "signal": signal.model_dump(),
        }
    else:
        hits = dense_search(req.message, k=5, ticker=req.ticker)
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
