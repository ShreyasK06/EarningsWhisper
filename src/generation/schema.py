"""Structured trading-signal output schema.

`supporting_claims` requires at least one entry -- an uncited signal is a
pydantic ValidationError, not a silent hallucination.
"""
from typing import Literal

from pydantic import BaseModel, Field


class SupportingClaim(BaseModel):
    text: str = Field(description="Verbatim span from the source chunk")
    chunk_id: str
    section: str


class Signal(BaseModel):
    ticker: str
    signal: Literal["bullish", "bearish", "neutral"]
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    supporting_claims: list[SupportingClaim] = Field(min_length=1)
