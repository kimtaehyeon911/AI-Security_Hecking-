"""Falsification tests for look-ahead leakage.

Per the Step 1 mandate ("성능이 좋게 나오면 먼저 데이터 누수를 의심하고 반증
테스트를 먼저 써라"): rather than assert the happy path, we actively try to make
the future leak and assert it cannot.
"""

from __future__ import annotations

import pytest
from conftest import utc

from vts.pit.clock import AsOfClock
from vts.pit.guard import LookaheadError, filter_visible
from vts.pit.schema import NewsItem, OHLCVBar
from vts.pit.store import PointInTimeStore


def _bar(day: int, close: float) -> OHLCVBar:
    t = utc(2024, 1, day, 21)
    return OHLCVBar(
        symbol="AAPL", event_time=t, knowledge_time=t, source="t",
        open=close, high=close, low=close, close=close, volume=1000,
    )


def test_future_ingestion_cannot_change_a_past_asof_view():
    """The core property: an as-of read is a pure function of past knowledge.

    Snapshot the world as-of T, then ingest a mountain of future data, then read
    as-of T again. The two reads must be byte-identical.
    """
    store = PointInTimeStore()
    store.append_many([_bar(2, 100), _bar(3, 101)])
    clock = AsOfClock.at(utc(2024, 1, 3, 21))

    before = store.get_ohlcv("AAPL", clock)
    before_json = [b.model_dump_json() for b in before]

    # Ingest future bars AND a future news bombshell.
    store.append_many([_bar(4, 200), _bar(5, 300), _bar(6, 400)])
    store.append(
        NewsItem(symbol="AAPL", event_time=utc(2024, 1, 5, 9), knowledge_time=utc(2024, 1, 5, 9),
                 source="t", title="rally!", url="z")
    )

    after = store.get_ohlcv("AAPL", clock)
    after_json = [b.model_dump_json() for b in after]

    assert before_json == after_json, "future ingestion leaked into a past as-of view"


def test_store_never_returns_a_future_record():
    """No matter the query window, results are always <= as_of."""
    store = PointInTimeStore()
    for d in range(2, 10):
        store.append(_bar(d, 100 + d))
    clock = AsOfClock.at(utc(2024, 1, 5, 21))

    # Deliberately ask for an event window that extends into the future.
    bars = store.get_ohlcv("AAPL", clock, start=utc(2024, 1, 1), end=utc(2024, 1, 31))
    assert max(b.knowledge_time for b in bars) <= clock.as_of
    assert all(b.event_time.day <= 5 for b in bars)


def test_guard_catches_a_hand_crafted_leak():
    """If buggy code ever hands a future record to the guarded egress, it raises."""
    clock = AsOfClock.at(utc(2024, 1, 5))
    leaked = [_bar(4, 100), _bar(9, 999)]  # day 9 is the future
    with pytest.raises(LookaheadError):
        filter_visible(leaked, clock, context="unit", strict=True)


def test_shuffling_future_prices_does_not_move_the_decision_window():
    """Replace all future closes with garbage; the as-of view is unchanged."""
    store = PointInTimeStore()
    real = [_bar(d, 100 + d) for d in range(2, 6)]
    store.append_many(real)
    clock = AsOfClock.at(utc(2024, 1, 4, 21))
    view_a = [(b.event_time, b.close) for b in store.get_ohlcv("AAPL", clock)]

    # Corrupt the future (days 5+) with absurd values.
    store.append_many([_bar(5, 99999), _bar(6, 1)])
    view_b = [(b.event_time, b.close) for b in store.get_ohlcv("AAPL", clock)]

    assert view_a == view_b
