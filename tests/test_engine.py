"""End-to-end walk-forward backtest with the deterministic momentum model."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import utc

from vts.backtest.engine import Backtester, BacktestConfig
from vts.backtest.model import FakeMomentumModel
from vts.backtest.splitter import walk_forward
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore


def _price_path(store: PointInTimeStore, symbol: str, start, n: int, p0: float, daily: float):
    price = p0
    dts = []
    for i in range(n):
        t = start + timedelta(days=i)
        tt = t.replace(hour=21)
        store.append(OHLCVBar(symbol=symbol, event_time=tt, knowledge_time=tt, source="test",
                              open=price, high=price, low=price, close=price, volume=1_000_000))
        dts.append(tt)
        price *= 1.0 + daily
    return dts


def _make_store():
    store = PointInTimeStore()
    start = utc(2024, 1, 2)
    up = _price_path(store, "UP", start, 45, 100.0, +0.01)   # steady uptrend
    _price_path(store, "DN", start, 45, 100.0, -0.01)        # steady downtrend
    return store, up


def test_backtest_runs_and_longs_the_uptrend():
    store, dts = _make_store()
    model = FakeMomentumModel(store)
    bt = Backtester(store, model, config=BacktestConfig(n_samples=3))
    decision_dates = [dts[25], dts[30], dts[35], dts[40]]
    result = bt.run(["UP", "DN"], decision_dates)

    assert len(result.records) == 4
    first = result.records[0]
    # UP is in an uptrend -> bullish rating -> positive weight; DN -> long-only 0.
    assert first.weights["UP"] > 0
    assert first.weights["DN"] == 0.0
    assert first.ratings["UP"] in {"Buy", "Overweight"}
    assert first.ratings["DN"] in {"Sell", "Underweight"}
    # Deterministic model -> unanimous vote -> zero dispersion.
    assert first.mean_dispersion == 0.0


def test_costs_are_charged_and_reduce_equity():
    store, dts = _make_store()
    model = FakeMomentumModel(store)
    decision_dates = [dts[25], dts[30], dts[35], dts[40]]

    free = Backtester(store, model, config=BacktestConfig()).run(["UP", "DN"], decision_dates)
    # A zero-cost model should end richer than one paying costs on the same path.
    from vts.backtest.costs import CostModel, CostParams
    zero = Backtester(store, model, cost_model=CostModel(CostParams(0, 0, 0, 0)),
                      config=BacktestConfig()).run(["UP", "DN"], decision_dates)
    assert free.records[0].cost > 0
    assert zero.final_equity >= free.final_equity


def test_segment_reports_tag_contamination_clean_for_exempt_momentum():
    store, dts = _make_store()
    model = FakeMomentumModel(store)
    decision_dates = [dts[i] for i in (25, 28, 31, 34, 37, 40)]
    bt = Backtester(store, model)
    result = bt.run(["UP", "DN"], decision_dates)
    folds = walk_forward(decision_dates, train_size=2, test_size=2)
    reports = bt.segment_reports(result, folds)
    assert reports, "expected at least one fold report"
    # The deterministic momentum model ships contamination-exempt (it cannot memorize
    # outcomes), so every segment reads 'clean' — not 'unknown'.
    assert all(r.contamination == "clean" for r in reports)


def test_uptrend_long_makes_money_gross():
    store, dts = _make_store()
    from vts.backtest.costs import CostModel, CostParams
    model = FakeMomentumModel(store)
    decision_dates = [dts[25], dts[30], dts[35], dts[40]]
    result = Backtester(store, model, cost_model=CostModel(CostParams(0, 0, 0, 0))).run(
        ["UP", "DN"], decision_dates
    )
    assert result.total_return > 0  # long the uptrend, no costs -> positive
