"""Falsification: future prices cannot change any backtest decision."""

from __future__ import annotations

from datetime import timedelta

from conftest import utc

from vts.backtest.engine import Backtester, BacktestConfig
from vts.backtest.model import FakeMomentumModel
from vts.pit.clock import AsOfClock
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore


def _append_path(store, symbol, start, n, p0, daily):
    price = p0
    dts = []
    for i in range(n):
        tt = (start + timedelta(days=i)).replace(hour=21)
        store.append(OHLCVBar(symbol=symbol, event_time=tt, knowledge_time=tt, source="t",
                              open=price, high=price, low=price, close=price, volume=1_000_000))
        dts.append(tt)
        price *= 1.0 + daily
    return dts


def test_model_decision_unaffected_by_future_bars():
    store = PointInTimeStore()
    dts = _append_path(store, "UP", utc(2024, 1, 2), 30, 100.0, +0.01)
    model = FakeMomentumModel(store)
    clock = AsOfClock.at(dts[25])

    before = model.decide("UP", clock).rating
    # Inject absurd FUTURE bars (after the clock); they must be invisible at `clock`.
    _append_path(store, "UP", dts[26] + timedelta(days=1), 5, 1e6, -0.9)
    after = model.decide("UP", clock).rating
    assert before == after


def test_future_ingestion_does_not_change_engine_ratings():
    store = PointInTimeStore()
    dts = _append_path(store, "UP", utc(2024, 1, 2), 45, 100.0, +0.01)
    _append_path(store, "DN", utc(2024, 1, 2), 45, 100.0, -0.01)
    model = FakeMomentumModel(store)
    decision_dates = [dts[25], dts[30], dts[35], dts[40]]

    baseline = Backtester(store, model, config=BacktestConfig()).run(["UP", "DN"], decision_dates)
    baseline_ratings = [r.ratings for r in baseline.records]

    # Ingest garbage bars strictly AFTER the last decision date; no decision can see them.
    _append_path(store, "UP", dts[44] + timedelta(days=1), 10, 5e5, -0.5)
    _append_path(store, "DN", dts[44] + timedelta(days=1), 10, 1.0, +0.5)

    rerun = Backtester(store, model, config=BacktestConfig()).run(["UP", "DN"], decision_dates)
    rerun_ratings = [r.ratings for r in rerun.records]

    assert baseline_ratings == rerun_ratings, "future ingestion leaked into decisions"
