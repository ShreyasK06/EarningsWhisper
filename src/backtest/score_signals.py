"""Score generated trading signals against realized forward stock returns.

CRITICAL: entry must be the next trading session's open if the filing was
made outside regular market hours (before 9:30am or at/after 4:00pm ET) --
using the same-day close for an after-hours filing is lookahead bias and
produces fake results. We derive after-hours status from each filing's
exact `filed_datetime` (the SEC ACCEPTANCE-DATETIME captured by
src/ingestion/parse.py) rather than a caller-supplied flag, since that
timestamp already carries the Eastern local hour.
"""
from datetime import timedelta

import pandas as pd
import yfinance as yf

MARKET_CLOSE_HOUR_ET = 16
MARKET_OPEN_HOUR_ET = 9


def filed_after_market_close(filed_datetime: str) -> bool:
    """SEC ACCEPTANCE-DATETIME is Eastern local time; treat before 9am or
    at/after 4pm ET as outside regular trading hours."""
    ts = pd.Timestamp(filed_datetime)
    return ts.hour >= MARKET_CLOSE_HOUR_ET or ts.hour < MARKET_OPEN_HOUR_ET


def forward_return(ticker: str, filed_datetime: str, horizon: int = 1):
    filed_ts = pd.Timestamp(filed_datetime)
    after_hours = filed_after_market_close(filed_datetime)

    start = filed_ts - timedelta(days=5)
    end = filed_ts + timedelta(days=horizon + 10)
    prices = yf.Ticker(ticker).history(start=start, end=end)
    if prices.empty:
        return None

    dates = prices.index.normalize()
    filed_date = filed_ts.normalize()
    if dates.tz is not None and filed_date.tz is None:
        filed_date = filed_date.tz_localize(dates.tz)

    after = prices[dates > filed_date] if after_hours else prices[dates >= filed_date]
    if len(after) < horizon + 1:
        return None

    entry = after.iloc[0]["Open"]
    exit_price = after.iloc[horizon]["Close"]
    return (exit_price - entry) / entry


def score(signals: list[dict]) -> pd.DataFrame:
    """signals: [{ticker, filed_datetime, signal, confidence}]"""
    rows = []
    for s in signals:
        r = forward_return(s["ticker"], s["filed_datetime"])
        if r is None:
            continue
        predicted_up = s["signal"] == "bullish"
        actual_up = r > 0
        hit = (predicted_up == actual_up) if s["signal"] != "neutral" else None
        rows.append({**s, "return": r, "hit": hit})

    df = pd.DataFrame(rows)
    directional = df[df["hit"].notna()] if not df.empty else df
    print(f"signals: {len(df)}  directional: {len(directional)}")
    if len(directional):
        print(f"hit rate: {directional['hit'].mean():.2%}")
        high_conf = directional[directional["confidence"] >= 0.7]
        if len(high_conf):
            print(f"hit rate (conf>=0.7, n={len(high_conf)}): {high_conf['hit'].mean():.2%}")
        print(f"mean return: {directional['return'].mean():.2%}")
    return df
