"""Decision cache, walk-forward splitter, and PIT realized returns."""

from __future__ import annotations

import pytest
from conftest import bar, utc

from vts.backtest.cache import DecisionCache, decision_key
from vts.backtest.returns import realized_return
from vts.backtest.splitter import walk_forward
from vts.decision import Decision, Rating
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore


# --- cache -------------------------------------------------------------------
def test_cache_roundtrip_and_miss():
    cache = DecisionCache()
    key = decision_key("AAPL", "2024-06-03T21:00:00+00:00", "m", "prompt", 0)
    assert cache.get(key) is None
    d = Decision(ticker="AAPL", as_of="2024-06-03T21:00:00+00:00", rating=Rating.BUY, confidence=0.8)
    cache.put(key, d)
    got = cache.get(key)
    assert got is not None and got.rating == Rating.BUY


def test_cache_key_deterministic_and_prompt_sensitive():
    a = decision_key("AAPL", "t", "m", "prompt-A", 0)
    b = decision_key("AAPL", "t", "m", "prompt-A", 0)
    c = decision_key("AAPL", "t", "m", "prompt-B", 0)
    assert a == b and a != c


# --- splitter ----------------------------------------------------------------
def test_walk_forward_non_overlapping_test():
    dates = [utc(2024, 1, d + 1) for d in range(10)]
    folds = walk_forward(dates, train_size=3, test_size=2)
    # start=0,2,4 fit (test_end 5,7,9); start=6 would need index 10 -> 3 folds.
    assert [f.index for f in folds] == [0, 1, 2]
    # test windows tile without overlap
    seen: set = set()
    for f in folds:
        for d in f.test:
            assert d not in seen
            seen.add(d)


def test_walk_forward_anchored_grows_train():
    dates = [utc(2024, 1, d + 1) for d in range(10)]
    rolling = walk_forward(dates, 3, 2, anchored=False)
    anchored = walk_forward(dates, 3, 2, anchored=True)
    assert len(rolling[1].train) == 3
    assert len(anchored[1].train) > 3  # expanding window


def test_walk_forward_rejects_unsorted():
    with pytest.raises(ValueError):
        walk_forward([utc(2024, 1, 3), utc(2024, 1, 1)], 1, 1)


# --- returns -----------------------------------------------------------------
def test_realized_return_capped_and_correct():
    store = PointInTimeStore()
    # 9 consecutive bars day2..day10 with closes 100..108 (linear +1).
    store.append_many([bar("AAPL", 2 + i, 100 + i) for i in range(9)])
    # entry = day2 (idx0, close 100); holding 5 trading bars -> idx5 = day7, close 105.
    r = realized_return(store, "AAPL", utc(2024, 1, 2, 21), holding_days=5,
                        ceiling=AsOfClock.at(utc(2024, 1, 20)))
    assert r == pytest.approx(0.05)


def test_realized_return_none_when_window_not_elapsed():
    store = PointInTimeStore()
    store.append_many([bar("AAPL", 2 + i, 100 + i) for i in range(3)])  # only 3 bars
    # ceiling on the entry day -> only the entry bar is visible -> window not elapsed.
    r = realized_return(store, "AAPL", utc(2024, 1, 2, 21), holding_days=5,
                        ceiling=AsOfClock.at(utc(2024, 1, 2, 21)))
    assert r is None
