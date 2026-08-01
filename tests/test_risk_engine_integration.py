"""RiskEngine wired into the Backtester: halt, gate and clamps drive real weights."""

from __future__ import annotations

import pytest
from conftest import utc

from vts.backtest.costs import CostModel, CostParams
from vts.backtest.engine import Backtester, BacktestConfig
from vts.decision import Decision, Rating
from vts.pit.schema import OHLCVBar
from vts.risk import RiskEngine, RiskLimits
from vts.pit.store import PointInTimeStore

FREE = CostModel(CostParams(0, 0, 0, 0))


def _bar(sym, day, close):
    t = utc(2024, 1, day, 21)
    return OHLCVBar(symbol=sym, event_time=t, knowledge_time=t, source="t",
                    open=close, high=close, low=close, close=close, volume=1_000_000)


class _Buy:
    model_id = "buy"

    def __init__(self, confidence=0.9):
        self._c = confidence

    def prompt_for(self, ticker, clock):
        return f"buy:{ticker}@{clock.as_of.isoformat()}"

    def decide(self, ticker, clock):
        return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                        rating=Rating.BUY, confidence=self._c)


def test_daily_loss_breach_halts_and_flattens_rest_of_run():
    store = PointInTimeStore()
    # Crash between day 9 and day 16 (-40%), then recovery.
    for day, close in [(2, 100), (9, 100), (16, 60), (23, 90), (30, 120)]:
        store.append(_bar("A", day, close))
    dates = [utc(2024, 1, d, 21) for d in (2, 9, 16, 23, 30)]
    risk = RiskEngine(RiskLimits(daily_loss_limit=0.03, max_weight_per_symbol=1.0))
    bt = Backtester(store, _Buy(), cost_model=FREE,
                    config=BacktestConfig(n_samples=1), risk=risk)
    result = bt.run(["A"], dates)

    by_day = {r.date.day: r for r in result.records}
    assert by_day[16].halted is True          # -40% period breaches the 3% limit
    assert by_day[16].weights["A"] == 0.0     # flattened at the halt
    assert by_day[23].halted is True          # latched: recovery days stay flat
    assert by_day[30].weights["A"] == 0.0
    # Equity is frozen from the halt onward (flat position, zero costs).
    assert result.final_equity == pytest.approx(by_day[23].equity)


def test_low_confidence_model_never_gets_a_position():
    store = PointInTimeStore()
    for day, close in [(2, 100), (9, 110), (16, 121)]:
        store.append(_bar("A", day, close))
    dates = [utc(2024, 1, d, 21) for d in (2, 9, 16)]
    risk = RiskEngine(RiskLimits(min_confidence=0.5))
    bt = Backtester(store, _Buy(confidence=0.2), cost_model=FREE,
                    config=BacktestConfig(n_samples=1), risk=risk)
    result = bt.run(["A"], dates)
    assert all(r.weights["A"] == 0.0 for r in result.records)
    assert all(r.forced_holds == 1 for r in result.records)
    assert result.total_return == pytest.approx(0.0)  # never invested


def test_symbol_cap_limits_single_name_concentration():
    store = PointInTimeStore()
    for day, close in [(2, 100), (9, 110)]:
        store.append(_bar("A", day, close))
    dates = [utc(2024, 1, d, 21) for d in (2, 9)]
    risk = RiskEngine(RiskLimits(max_weight_per_symbol=0.20))
    bt = Backtester(store, _Buy(), cost_model=FREE,
                    config=BacktestConfig(n_samples=1), risk=risk)
    result = bt.run(["A"], dates)
    assert result.records[0].weights["A"] == pytest.approx(0.20)  # Buy wants 1.0, cap wins


def test_engine_without_risk_layer_unchanged():
    store = PointInTimeStore()
    for day, close in [(2, 100), (9, 110)]:
        store.append(_bar("A", day, close))
    dates = [utc(2024, 1, d, 21) for d in (2, 9)]
    bt = Backtester(store, _Buy(), cost_model=FREE, config=BacktestConfig(n_samples=1))
    result = bt.run(["A"], dates)
    assert result.records[0].weights["A"] == pytest.approx(1.0)
    assert result.records[0].halted is False
