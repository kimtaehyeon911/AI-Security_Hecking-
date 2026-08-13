"""Shared test helpers."""

from __future__ import annotations

from datetime import datetime, timezone


def utc(y: int, m: int, d: int, hh: int = 0, mm: int = 0, ss: int = 0) -> datetime:
    """Construct a tz-aware UTC datetime succinctly."""
    return datetime(y, m, d, hh, mm, ss, tzinfo=timezone.utc)


def bar(symbol: str, day: int, close: float, *, month: int = 1, year: int = 2024, volume: float = 1_000_000):
    """A single daily OHLCV bar stamped at the session close (21:00 UTC)."""
    from vts.pit.schema import OHLCVBar

    t = utc(year, month, day, 21)
    return OHLCVBar(
        symbol=symbol, event_time=t, knowledge_time=t, source="test",
        open=close, high=close, low=close, close=close, volume=volume,
    )
