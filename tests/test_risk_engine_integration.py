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
    assert result.risk_enabled is False       # un-gated run is surfaced, not hidden


def test_single_day_crash_caught_under_weekly_cadence():
    """DAILY marks are fed even when rebalancing weekly: a one-day -5% mid-week halts."""
    store = PointInTimeStore()
    # Weekly decisions on day 2 and day 9, but DAILY bars exist in between.
    # Day 5 drops -5% intraday-to-close then recovers by day 9 -> week net small.
    closes = {2: 100, 3: 100, 4: 100, 5: 95, 6: 97, 7: 99, 8: 100, 9: 100}
    for day, c in closes.items():
        store.append(_bar("A", day, c))
    dates = [utc(2024, 1, 2, 21), utc(2024, 1, 9, 21)]  # weekly cadence
    risk = RiskEngine(RiskLimits(daily_loss_limit=0.03, max_weight_per_symbol=1.0,
                                 max_drawdown_limit=None))
    bt = Backtester(store, _Buy(), cost_model=FREE,
                    config=BacktestConfig(n_samples=1), risk=risk)
    result = bt.run(["A"], dates)
    # The week's net return is ~0, but the day-4->5 single-day -5% breaches the halt.
    assert result.records[1].halted is True


def test_stale_halt_latch_raises_on_reuse():
    store = PointInTimeStore()
    for day, close in [(2, 100), (9, 50)]:  # -50% crash latches
        store.append(_bar("A", day, close))
    dates = [utc(2024, 1, d, 21) for d in (2, 9)]
    risk = RiskEngine(RiskLimits(daily_loss_limit=0.03, max_weight_per_symbol=1.0))
    bt = Backtester(store, _Buy(), cost_model=FREE,
                    config=BacktestConfig(n_samples=1), risk=risk)
    bt.run(["A"], dates)
    assert risk.halt.halted is True
    with pytest.raises(RuntimeError, match="already halted"):
        bt.run(["A"], dates)               # reused engine still latched -> loud error


def test_backtest_refuses_to_run_under_engaged_kill_switch(monkeypatch):
    from vts.risk.killswitch import KILL_SWITCH_ENV
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    store = PointInTimeStore()
    store.append_many([_bar("A", 2, 100), _bar("A", 9, 110)])
    dates = [utc(2024, 1, d, 21) for d in (2, 9)]
    bt = Backtester(store, _Buy(), cost_model=FREE,
                    config=BacktestConfig(n_samples=1), risk=RiskEngine(RiskLimits()))
    with pytest.raises(RuntimeError, match="KILL_SWITCH"):
        bt.run(["A"], dates)               # no silent flat backtest
