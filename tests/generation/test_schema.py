import pytest
from pydantic import ValidationError

from src.generation.schema import Signal


def _valid_claim():
    return {"text": "revenue grew 17%", "chunk_id": "AAPL_2026Q2_PreparedRemarks_000", "section": "PreparedRemarks"}


def test_signal_requires_at_least_one_supporting_claim():
    with pytest.raises(ValidationError):
        Signal(ticker="AAPL", signal="bullish", confidence=0.8, reasoning="strong quarter", supporting_claims=[])


def test_signal_rejects_confidence_out_of_range():
    with pytest.raises(ValidationError):
        Signal(
            ticker="AAPL", signal="bullish", confidence=1.5, reasoning="strong quarter",
            supporting_claims=[_valid_claim()],
        )


def test_signal_rejects_invalid_direction():
    with pytest.raises(ValidationError):
        Signal(
            ticker="AAPL", signal="up", confidence=0.5, reasoning="x",
            supporting_claims=[_valid_claim()],
        )


def test_signal_accepts_valid_input():
    sig = Signal(
        ticker="AAPL", signal="neutral", confidence=0.4, reasoning="mixed signals",
        supporting_claims=[_valid_claim()],
    )
    assert sig.signal == "neutral"
    assert len(sig.supporting_claims) == 1
