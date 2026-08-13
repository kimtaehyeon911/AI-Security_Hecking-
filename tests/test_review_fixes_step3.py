"""Regression tests for the Step 3 adversarial-review findings (round 3)."""

from __future__ import annotations

import copy
import pickle

import pytest
from conftest import utc

from vts.backtest.benchmarks import buy_and_hold_curve, entry_basket, sixty_forty_curve
from vts.backtest.costs import CostModel, CostParams
from vts.backtest.engine import Backtester, BacktestConfig
from vts.backtest.evaluate import evaluate, render_report
from vts.backtest.llm_costs import LLMCostTracker
from vts.backtest.metrics import avg_win_loss_ratio, cagr, compute_metrics, max_drawdown
from vts.decision import Decision, Rating
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore

FREE = CostModel(CostParams(0, 0, 0, 0))


def _bar(sym, day, close, month=1):
    t = utc(2024, month, day, 21)
    return OHLCVBar(symbol=sym, event_time=t, knowledge_time=t, source="t",
                    open=close, high=close, low=close, close=close, volume=1_000_000)


class _Const:
    def __init__(self, rating, model_id="const"):
        self.model_id = model_id
        self._rating = rating

    def prompt_for(self, ticker, clock):
        return f"{self.model_id}:{ticker}@{clock.as_of.isoformat()}"

    def decide(self, ticker, clock):
        return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                        rating=self._rating, confidence=0.9)


# --- metrics: overflow, wipe-out consistency, sentinels, drawdown -------------
def test_cagr_no_overflow_on_short_interval():
    curve = [(utc(2024, 1, 2, 10), 100.0), (utc(2024, 1, 2, 11), 110.0)]  # +10% in 1h
    v = cagr(curve)  # must not raise OverflowError
    assert v > 0 and v != float("inf")


def test_wiped_out_curve_metrics_are_internally_consistent():
    # Equity dies at day 4; post-crash points must not desync n_periods/ppy.
    curve = [(utc(2024, 1, d, 21), v) for d, v in
             [(2, 100.0), (3, 50.0), (4, 0.0), (5, 10.0), (6, 20.0)]]
    m = compute_metrics(curve)
    assert m.n_periods == 2            # 100->50, 50->0; post-crash points truncated
    assert m.total_return == -1.0
    assert m.cagr == -1.0
    assert m.max_drawdown == 1.0


def test_period_return_clamped_at_minus_100pct():
    curve = [(utc(2024, 1, 2, 21), 100.0), (utc(2024, 1, 3, 21), -32.0)]
    m = compute_metrics(curve)
    assert m.total_return == -1.0      # clamped, not -132%


def test_all_flat_curve_win_loss_is_none():
    flat = [(utc(2024, 1, d, 21), 100.0) for d in range(2, 7)]
    assert avg_win_loss_ratio(flat) is None  # undefined, not 0.0


def test_losses_but_no_wins_is_zero():
    falling = [(utc(2024, 1, d, 21), v) for d, v in [(2, 100.0), (3, 90.0), (4, 80.0)]]
    assert avg_win_loss_ratio(falling) == 0.0


def test_all_negative_curve_drawdown_is_total():
    assert max_drawdown([(utc(2024, 1, 2, 21), -10.0), (utc(2024, 1, 3, 21), -50.0)]) == 1.0


# --- engine: final-date cost reaches the gated curve --------------------------
def test_final_date_cost_reflected_in_total_return():
    store = PointInTimeStore()
    store.append(_bar("A", 2, 100))
    bt = Backtester(store, _Const(Rating.BUY), config=BacktestConfig(n_samples=1))
    result = bt.run(["A"], [utc(2024, 1, 2, 21)])  # single date: Buy, cost charged
    assert result.records[0].cost > 0
    assert result.final_equity == pytest.approx(100_000 - result.records[0].cost)
    assert result.total_return < 0
    assert result.equity_curve[-1][1] == result.final_equity


# --- engine + benchmarks: zero-close bars never divide ------------------------
def test_zero_close_bar_does_not_crash_engine_or_benchmarks():
    store = PointInTimeStore()
    store.append_many([_bar("A", 2, 100), _bar("A", 9, 110)])
    store.append_many([_bar("Z", 2, 0.0), _bar("Z", 9, 0.0)])  # schema-valid zero close
    dates = [utc(2024, 1, 2, 21), utc(2024, 1, 9, 21)]
    bt = Backtester(store, _Const(Rating.BUY), cost_model=FREE, config=BacktestConfig(n_samples=1))
    result = bt.run(["A", "Z"], dates)             # must not raise
    assert "Z" not in result.records[0].ratings    # zero-close excluded from priced
    curve = buy_and_hold_curve(store, ["A", "Z"], dates, cost_model=FREE)
    assert curve[0][1] == pytest.approx(100_000)


# --- benchmarks: entry cost drags returns; no pre-entry padding ---------------
def test_benchmark_entry_cost_drags_total_return():
    store = PointInTimeStore()
    store.append_many([_bar("A", 2, 100), _bar("A", 9, 110)])
    dates = [utc(2024, 1, 2, 21), utc(2024, 1, 9, 21)]
    costed = CostModel(CostParams(commission_bps=10.0, sell_tax_bps=0,
                                  half_spread_bps=0, impact_coef_bps=0))
    curve = buy_and_hold_curve(store, ["A"], dates, cost_model=costed)
    total = curve[-1][1] / curve[0][1] - 1.0
    assert total < 0.10                      # strictly below the pre-cost +10%
    assert total == pytest.approx(0.10 - 100_000 * 0.001 / 100_000, rel=1e-6)


def test_benchmark_curve_starts_at_entry_no_flat_stub():
    store = PointInTimeStore()
    store.append_many([_bar("A", 16, 100), _bar("A", 23, 110)])  # lists at day 16
    dates = [utc(2024, 1, d, 21) for d in (2, 9, 16, 23)]
    curve = buy_and_hold_curve(store, ["A"], dates, cost_model=FREE)
    assert len(curve) == 2                   # days 2 and 9 dropped, not padded flat
    assert curve[0][0] == utc(2024, 1, 16, 21)


def test_entry_basket_discloses_membership():
    store = PointInTimeStore()
    store.append_many([_bar("A", 2, 100)])   # B never priced
    basket, entry = entry_basket(store, ["A", "B"], [utc(2024, 1, 2, 21)])
    assert basket == ["A"] and entry == utc(2024, 1, 2, 21)


# --- engine: idle cash earns rf when configured -------------------------------
def test_strategy_idle_cash_accrues_rf():
    store = PointInTimeStore()
    store.append_many([
        OHLCVBar(symbol="A", event_time=utc(2023, 1, 2, 21), knowledge_time=utc(2023, 1, 2, 21),
                 source="t", open=100, high=100, low=100, close=100, volume=1_000_000),
        OHLCVBar(symbol="A", event_time=utc(2024, 1, 2, 21), knowledge_time=utc(2024, 1, 2, 21),
                 source="t", open=100, high=100, low=100, close=100, volume=1_000_000),
    ])
    dates = [utc(2023, 1, 2, 21), utc(2024, 1, 2, 21)]
    bt = Backtester(store, _Const(Rating.HOLD), cost_model=FREE,
                    config=BacktestConfig(n_samples=1, rf_annual=0.10))
    result = bt.run(["A"], dates)
    # All-cash strategy over ~1 year at 10% rf.
    assert result.final_equity == pytest.approx(110_000, rel=2e-3)


# --- evaluate: per-run cost accounting, run-rate budget -----------------------
def test_evaluate_uses_per_run_costs_not_tracker_lifetime():
    store = PointInTimeStore()
    for i, c in enumerate([100, 90, 80, 70]):
        store.append(_bar("DN", 2 + i * 7, c))
    dates = [utc(2024, 1, 2 + i * 7, 21) for i in range(4)]
    tracker = LLMCostTracker(est_cost_per_call_usd=0.01)
    bt = Backtester(store, _Const(Rating.SELL), cost_model=FREE,
                    config=BacktestConfig(n_samples=1), llm_tracker=tracker)
    r1 = bt.run(["DN"], dates)
    r2 = bt.run(["DN"], dates)  # identical run: all cache hits, zero new cost

    assert r1.llm_calls == 4 and r1.llm_cost_usd == pytest.approx(0.04)
    assert r2.llm_calls == 0 and r2.llm_cache_hits == 4 and r2.llm_cost_usd == 0.0

    report2 = evaluate(bt, r2, ["DN"], monthly_budget_usd=30.0)
    assert report2.costs.llm_calls == 0          # not the lifetime 4
    assert report2.costs.total_cost_usd == 0.0   # not the lifetime $0.04


def test_budget_uses_monthly_run_rate():
    store = PointInTimeStore()
    # 3-day window: day 2 -> day 5.
    for i, c in enumerate([100, 99, 98]):
        store.append(_bar("DN", 2 + i, c))
    dates = [utc(2024, 1, 2 + i, 21) for i in range(3)]
    tracker = LLMCostTracker(est_cost_per_call_usd=3.0)  # $9 over a 3-day window
    bt = Backtester(store, _Const(Rating.SELL), cost_model=FREE,
                    config=BacktestConfig(n_samples=1), llm_tracker=tracker)
    result = bt.run(["DN"], dates)
    report = evaluate(bt, result, ["DN"], monthly_budget_usd=10.0)
    # $9 in 3 days is a ~$91/month run-rate -> OVER a $10/month budget.
    assert report.costs.monthly_run_rate_usd > 10.0
    assert report.costs.within_budget is False


# --- tracker: copy/pickle ------------------------------------------------------
def test_tracker_deepcopy_and_pickle():
    t = LLMCostTracker(est_cost_per_call_usd=0.01)
    t.record_call()
    clone = copy.deepcopy(t)
    assert clone.calls == 1
    restored = pickle.loads(pickle.dumps(t))
    assert restored.total_cost_usd == pytest.approx(0.01)
    restored.record_call()  # lock restored and functional
    assert restored.calls == 2


# --- report: assumptions disclosed --------------------------------------------
def test_report_discloses_benchmark_assumptions():
    store = PointInTimeStore()
    for i, c in enumerate([100, 90, 80, 70]):
        store.append(_bar("DN", 2 + i * 7, c))
    store.append(_bar("A", 2, 100))
    dates = [utc(2024, 1, 2 + i * 7, 21) for i in range(4)]
    bt = Backtester(store, _Const(Rating.SELL), cost_model=FREE,
                    config=BacktestConfig(n_samples=1))
    result = bt.run(["DN"], dates)
    text = render_report(evaluate(bt, result, ["DN"], rf_annual=0.04))
    assert "cash proxy, not bonds" in text
    assert "rf_annual=4.00%" in text
    assert "frozen at entry" in text
