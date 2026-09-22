"""Generate a structured trading signal for a ticker via a schema-constrained
LLM call, grounded in retrieved earnings-release excerpts.

Requires GEMINI_API_KEY in .env.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pydantic import ValidationError

from src.generation.llm_client import get_client
from src.generation.prompts import SYSTEM, build_user_prompt
from src.generation.schema import Signal
from src.retrieval.hybrid import hybrid_search

CONTEXT_QUERIES = [
    "outlook and guidance for coming quarters",
    "risks and headwinds to revenue",
    "margin pressure and cost trends",
    "demand trends and customer commentary",
]


def gather_context(ticker: str, fiscal_quarter: str | None = None, k_each: int = 3) -> list[dict]:
    seen: set[str] = set()
    context: list[dict] = []
    for query in CONTEXT_QUERIES:
        for hit in hybrid_search(query, k=k_each, ticker=ticker, fiscal_quarter=fiscal_quarter):
            if hit["chunk_id"] not in seen:
                seen.add(hit["chunk_id"])
                context.append(hit)
    return context


def generate_signal(ticker: str, fiscal_quarter: str | None = None, retries: int = 1):
    chunks = gather_context(ticker, fiscal_quarter=fiscal_quarter)
    label = f"{ticker} ({fiscal_quarter})" if fiscal_quarter else ticker
    client = get_client()
    schema = Signal.model_json_schema()
    user_prompt = build_user_prompt(label, chunks)

    for attempt in range(retries + 1):
        raw = client.structured_call(SYSTEM, user_prompt, schema)
        try:
            return Signal(**raw), chunks
        except ValidationError as e:
            if attempt == retries:
                raise
            user_prompt += f"\n\nYour previous response failed validation: {e}. Emit again, corrected."


if __name__ == "__main__":
    import json

    signal, _ = generate_signal("AAPL")
    print(json.dumps(signal.model_dump(), indent=2))
