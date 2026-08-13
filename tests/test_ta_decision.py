"""TradingAgents adapter: pure parsing and the gated reflection resolver."""

from __future__ import annotations

from datetime import date

from conftest import bar, utc

from vts.backtest.reflection_gate import PendingEntry
from vts.decision import Rating
from vts.integration.ta_decision import (
    build_decision,
    rating_from_text,
    resolvable_reflections,
)
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore


def test_rating_from_text_explicit_label():
    assert rating_from_text("Rating: **Overweight**") == Rating.OVERWEIGHT
    assert rating_from_text("FINAL — Rating: Sell") == Rating.SELL


def test_rating_from_text_freeform_and_default():
    assert rating_from_text("On balance we recommend a Buy here.") == Rating.BUY
    assert rating_from_text("no rating word at all") == Rating.HOLD  # conservative default


def test_build_decision_defaults_confidence():
    clock = AsOfClock.at(utc(2024, 6, 3, 21))
    d = build_decision("AAPL", clock, "Rating: Hold", thesis="balanced")
    assert d.rating == Rating.HOLD
    assert d.confidence == 0.5
    assert d.ticker == "AAPL"


def test_resolvable_reflections_gates_and_scores():
    store = PointInTimeStore()
    # 9 consecutive bars day2..day10, closes 100..108.
    store.append_many([bar("AAPL", 2 + i, 100 + i) for i in range(9)])
    pending = [
        PendingEntry("AAPL", date(2024, 1, 2)),   # matured (2+5=7 <= trade_date)
        PendingEntry("AAPL", date(2024, 1, 24)),  # not matured (24+5=29 > 25) -> gated out
    ]
    out = resolvable_reflections(store, pending, utc(2024, 1, 25), holding_days=5)
    assert len(out) == 1
    entry, ret = out[0]
    assert entry.entry_date == date(2024, 1, 2)
    assert ret > 0  # day2 100 -> day7 105
