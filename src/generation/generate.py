"""Generate a structured trading signal for a ticker via Claude tool use,
grounded in retrieved earnings-release excerpts.

Requires ANTHROPIC_API_KEY in .env.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from anthropic import Anthropic
from pydantic import ValidationError

import config
from src.generation.prompts import SYSTEM, build_user_prompt
from src.generation.schema import Signal
from src.retrieval.hybrid import hybrid_search

MODEL = "claude-opus-5"

TOOL = {
    "name": "emit_signal",
    "description": "Emit a structured trading signal with citations.",
    "input_schema": Signal.model_json_schema(),
}

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
    client = Anthropic(api_key=config.ANTHROPIC_API_KEY)
    messages = [{"role": "user", "content": build_user_prompt(label, chunks)}]

    for attempt in range(retries + 1):
        response = client.messages.create(
            model=MODEL,
            max_tokens=2000,
            system=SYSTEM,
            tools=[TOOL],
            tool_choice={"type": "tool", "name": "emit_signal"},
            messages=messages,
        )
        tool_use = next(b for b in response.content if b.type == "tool_use")
        try:
            return Signal(**tool_use.input), chunks
        except ValidationError as e:
            if attempt == retries:
                raise
            messages += [
                {"role": "assistant", "content": response.content},
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": tool_use.id,
                            "content": f"Validation failed: {e}. Emit again, corrected.",
                            "is_error": True,
                        }
                    ],
                },
            ]


if __name__ == "__main__":
    import json

    signal, _ = generate_signal("AAPL")
    print(json.dumps(signal.model_dump(), indent=2))
