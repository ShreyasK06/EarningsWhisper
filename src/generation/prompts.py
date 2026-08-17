"""System and user prompt construction for signal generation."""

SYSTEM = """You are a financial analyst reading earnings press-release excerpts.

Rules:
- Use ONLY the provided excerpts. Never use outside knowledge about the company.
- Every claim in your reasoning must be traceable to a provided chunk_id.
- Confidence calibration: 0.8+ only when multiple independent excerpts support
  the same direction. Use 0.3-0.5 when evidence is mixed or hedged.
- "neutral" is a valid and often correct answer. Do not manufacture a signal.
"""


def build_user_prompt(ticker: str, chunks: list[dict]) -> str:
    blocks = "\n\n".join(
        f"[chunk_id: {c['chunk_id']} | section: {c['section']}]\n{c['text']}"
        for c in chunks
    )
    return f"""Analyze these excerpts from {ticker}'s recent earnings release and emit a trading signal.

EXCERPTS:
{blocks}

Consider: forward-looking language, hedging, changes in tone, margin/demand commentary."""
