"""LLM-judge faithfulness check: does the generated reasoning stay grounded
in the retrieved excerpts?

The judge itself must be spot-checked by hand against ~10 examples before
its score is trusted (compare its labels to your own reading) -- an
unvalidated judge is a number you can't defend.
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.generation.llm_client import get_client

JUDGE_SYSTEM = """You are grading whether an analyst's reasoning is supported by source excerpts.

Break the reasoning into individual factual claims. For each, label:
- "supported": directly stated or clearly implied by an excerpt
- "unsupported": not found in any excerpt
- "contradicted": an excerpt says otherwise

Return ONLY JSON: {"claims":[{"claim":"...","label":"..."}]}"""


def score_claims(claims: list[dict]) -> float:
    if not claims:
        return 0.0
    supported = sum(1 for c in claims if c["label"] == "supported")
    return supported / len(claims)


def judge(reasoning: str, chunks: list[dict]):
    excerpts = "\n\n".join(f"[{c['chunk_id']}] {c['text']}" for c in chunks)
    client = get_client()
    text = client.text_call(
        JUDGE_SYSTEM, f"EXCERPTS:\n{excerpts}\n\nREASONING:\n{reasoning}"
    ).strip()
    text = text.removeprefix("```json").removesuffix("```").strip()
    claims = json.loads(text)["claims"]
    return score_claims(claims), claims
