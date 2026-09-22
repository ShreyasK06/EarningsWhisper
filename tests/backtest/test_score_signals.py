import numpy as np
import pandas as pd
import pytest

import src.backtest.score_signals as score_mod


def _make_price_frame(dates, opens, closes):
    return pd.DataFrame({"Open": opens, "Close": closes}, index=pd.DatetimeIndex(dates))


class _FakeTicker:
    def __init__(self, prices):
        self._prices = prices

    def __call__(self, ticker):
        return self

    def history(self, start, end):
        return self._prices


def test_filed_after_market_close_detects_after_hours():
    assert score_mod.filed_after_market_close("2026-05-01T20:15:00") is True  # after close
    assert score_mod.filed_after_market_close("2026-05-01T06:00:00") is False  # genuinely pre-market
    assert score_mod.filed_after_market_close("2026-05-01T10:00:00") is True  # intraday, no valid same-day entry
    assert score_mod.filed_after_market_close("2026-05-01T09:29:00") is False  # still pre-market
    assert score_mod.filed_after_market_close("2026-05-01T09:30:00") is True  # exactly at open


def test_forward_return_uses_next_day_open_for_after_hours_filing(monkeypatch):
    # Filed 8:15pm on 2026-05-01 (after close) -> entry must be 2026-05-02's
    # open, NOT 2026-05-01's close (that would be lookahead bias).
    prices = _make_price_frame(
        ["2026-04-29", "2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03"],
        opens=[100, 101, 102, 200, 210],
        closes=[101, 102, 103, 205, 212],
    )
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    r = score_mod.forward_return("AAPL", "2026-05-01T20:15:00", horizon=1)

    assert r == pytest.approx((212 - 200) / 200)


def test_forward_return_uses_same_day_open_for_premarket_filing(monkeypatch):
    # Filed 7:00am on 2026-05-01 (genuinely pre-market, before 9:30am ET) ->
    # that day's own open occurs chronologically AFTER the filing, so it's a
    # valid entry price.
    prices = _make_price_frame(
        ["2026-04-29", "2026-04-30", "2026-05-01", "2026-05-02"],
        opens=[100, 101, 102, 103],
        closes=[101, 102, 103, 104],
    )
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    r = score_mod.forward_return("AAPL", "2026-05-01T07:00:00", horizon=1)

    assert r == pytest.approx((104 - 102) / 102)


def test_forward_return_uses_next_day_open_for_intraday_filing(monkeypatch):
    # Filed 12:00pm on 2026-05-01 (during regular trading hours) -> that
    # day's open (9:30am) already happened before the filing, so it's not a
    # valid entry; must wait for 2026-05-02's open, same as an after-hours
    # filing.
    prices = _make_price_frame(
        ["2026-04-29", "2026-04-30", "2026-05-01", "2026-05-02", "2026-05-03"],
        opens=[100, 101, 102, 200, 210],
        closes=[101, 102, 103, 205, 212],
    )
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    r = score_mod.forward_return("AAPL", "2026-05-01T12:00:00", horizon=1)

    assert r == pytest.approx((212 - 200) / 200)


def test_forward_return_returns_none_when_insufficient_future_data(monkeypatch):
    prices = _make_price_frame(["2026-05-01"], opens=[100], closes=[101])
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    assert score_mod.forward_return("AAPL", "2026-05-01T20:15:00", horizon=1) is None


def test_forward_return_returns_none_on_empty_history(monkeypatch):
    prices = pd.DataFrame({"Open": [], "Close": []})
    monkeypatch.setattr(score_mod.yf, "Ticker", _FakeTicker(prices))

    assert score_mod.forward_return("BADTICKER", "2026-05-01T20:15:00", horizon=1) is None


def test_score_computes_hit_rate_and_treats_neutral_as_non_directional(monkeypatch):
    def fake_forward_return(ticker, filed_datetime, horizon=1):
        return {"AAPL": 0.05, "MSFT": -0.02, "NVDA": 0.01}[ticker]

    monkeypatch.setattr(score_mod, "forward_return", fake_forward_return)

    signals = [
        {"ticker": "AAPL", "filed_datetime": "2026-05-01T20:15:00", "signal": "bullish", "confidence": 0.8},
        {"ticker": "MSFT", "filed_datetime": "2026-05-01T20:15:00", "signal": "bullish", "confidence": 0.6},
        {"ticker": "NVDA", "filed_datetime": "2026-05-01T20:15:00", "signal": "neutral", "confidence": 0.4},
    ]

    df = score_mod.score(signals)

    assert len(df) == 3
    directional = df[df["hit"].notna()]
    assert len(directional) == 2  # neutral is excluded from hit-rate scoring
    assert directional["hit"].tolist() == [True, False]


def test_score_hit_rate_is_correct_with_real_numpy_returns_and_neutrals(monkeypatch, capsys):
    # forward_return's real implementation returns numpy.float64 (from a
    # pandas Series division), which makes `r > 0` a numpy.bool_ rather than
    # a Python bool. numpy.bool_ addition is logical OR, not arithmetic sum --
    # mixed with None for neutral signals (forcing the `hit` column to object
    # dtype), a naive pandas.Series.mean() silently collapses N numpy.bool_
    # values into a single True/False before dividing, instead of counting.
    # This regression-tests that score() avoids that collapse. 3 hits, 1 miss,
    # 1 neutral -> correct hit rate is 3/4 = 75%, not 1/4 = 25% (what the
    # object-dtype-mean bug would silently report instead).
    returns = {
        "A": np.float64(0.05),
        "B": np.float64(-0.02),
        "C": np.float64(0.01),
        "D": np.float64(-0.03),
        "E": np.float64(0.02),
    }

    def fake_forward_return(ticker, filed_datetime, horizon=1):
        return returns[ticker]

    monkeypatch.setattr(score_mod, "forward_return", fake_forward_return)

    signals = [
        {"ticker": "A", "filed_datetime": "2026-05-01T20:15:00", "signal": "bullish", "confidence": 0.8},
        {"ticker": "B", "filed_datetime": "2026-05-01T20:15:00", "signal": "bearish", "confidence": 0.8},
        {"ticker": "C", "filed_datetime": "2026-05-01T20:15:00", "signal": "bullish", "confidence": 0.8},
        {"ticker": "D", "filed_datetime": "2026-05-01T20:15:00", "signal": "bullish", "confidence": 0.8},
        {"ticker": "E", "filed_datetime": "2026-05-01T20:15:00", "signal": "neutral", "confidence": 0.4},
    ]

    df = score_mod.score(signals)
    directional = df[df["hit"].notna()]

    assert isinstance(directional["hit"].iloc[0], bool)  # native Python bool, not numpy.bool_
    assert directional["hit"].mean() == pytest.approx(0.75)
    assert "hit rate: 75.00%" in capsys.readouterr().out
