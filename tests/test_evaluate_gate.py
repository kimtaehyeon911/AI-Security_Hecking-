"""Benchmarks, the after-cost gate, and LLM cost reporting."""

from __future__ import annotations

import pytest
from conftest import utc

from vts.backtest.benchmarks import buy_and_hold_curve, sixty_forty_curve
from vts.backtest.costs import CostModel, CostParams
from vts.backtest.cutoff import CutoffRegistry
from vts.backtest.engine import Backtester, BacktestConfig
from vts.backtest.evaluate import evaluate, render_report
from vts.backtest.llm_costs import LLMCostTracker
from vts.decision import Decision, Rating
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore

FREE = CostModel(CostParams(0, 0, 0, 0))


def _bar(sym, day, close, month=1):
    t = utc(2024, month, day, 21)
    return OHLCVBar(symbol=sym, event_time=t, knowledge_time=t, source="t",
                    open=close, high=close, low=close, close=close, volume=1_000_000)


class _RatingModel:
    """Deterministic model emitting a fixed rating per ticker."""

    model_id = "const"

    def __init__(self, ratings: dict[str, Rating]):
        self._r = ratings

    def prompt_for(self, ticker, clock):
        return f"const({self._r[ticker].value}):{ticker}@{clock.as_of.isoformat()}"

    def decide(self, ticker, clock):
        return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                        rating=self._r[ticker], confidence=0.9)


# --- benchmarks ---------------------------------------------------------------
def test_buy_and_hold_is_share_based_not_constant_mix():
    store = PointInTimeStore()
    # A: 100 -> 150 -> 100 ; B: flat 100. Equal-weight $100k.
    store.append_many([_bar("A", 2, 100), _bar("A", 9, 150), _bar("A", 16, 100)])
    store.append_many([_bar("B", 2, 100), _bar("B", 9, 100), _bar("B", 16, 100)])
    dates = [utc(2024, 1, d, 21) for d in (2, 9, 16)]
    curve = buy_and_hold_curve(store, ["A", "B"], dates, cost_model=FREE)
    # True buy&hold ends where it started (A round-trips); constant-mix would end ~+4.17%.
    assert curve[0][1] == pytest.approx(100_000)
    assert curve[1][1] == pytest.approx(125_000)
    assert curve[-1][1] == pytest.approx(100_000)


def test_sixty_forty_cash_sleeve_accrues_rf():
    store = PointInTimeStore()
    store.append_many([
        OHLCVBar(symbol="A", event_time=utc(2023, 1, 2, 21), knowledge_time=utc(2023, 1, 2, 21),
                 source="t", open=100, high=100, low=100, close=100, volume=1_000_000),
        OHLCVBar(symbol="A", event_time=utc(2024, 1, 2, 21), knowledge_time=utc(2024, 1, 2, 21),
                 source="t", open=100, high=100, low=100, close=100, volume=1_000_000),
    ])
    dates = [utc(2023, 1, 2, 21), utc(2024, 1, 2, 21)]
    curve = sixty_forty_curve(store, ["A"], dates, rf_annual=0.10, cost_model=FREE)
    # Equity 60k flat; cash 40k grows ~10% over ~1 year -> ~104k total.
    assert curve[-1][1] == pytest.approx(104_000, rel=2e-3)


# --- gate ---------------------------------------------------------------------
def _downtrend_setup():
    """DN falls 100->50; a Sell-rated strategy stays flat and must beat benchmarks."""
    store = PointInTimeStore()
    closes = [100, 90, 80, 65, 50]
    for i, c in enumerate(closes):
        store.append(_bar("DN", 2 + i * 7, c))
    dates = [utc(2024, 1, 2 + i * 7, 21) for i in range(len(closes))]
    return store, dates


def test_gate_pass_when_strategy_beats_after_costs():
    store, dates = _downtrend_setup()
    model = _RatingModel({"DN": Rating.SELL})
    bt = Backtester(store, model, cost_model=FREE, config=BacktestConfig(n_samples=1))
    result = bt.run(["DN"], dates)
    report = evaluate(bt, result, ["DN"])
    assert report.gate.passed is True           # flat beats a -50% benchmark
    assert report.gate.certifiable is False     # bundled registry: cutoff unknown
    assert report.gate.contamination == "unknown"
    assert "참고용" in report.gate.verdict


def test_gate_fail_when_strategy_matches_benchmark():
    """A fully-long strategy on one asset can never strictly beat buy&hold of it."""
    store = PointInTimeStore()
    for i, c in enumerate([100, 110, 121, 133, 146]):
        store.append(_bar("UP", 2 + i * 7, c))
    dates = [utc(2024, 1, 2 + i * 7, 21) for i in range(5)]
    model = _RatingModel({"UP": Rating.BUY})
    bt = Backtester(store, model, cost_model=FREE, config=BacktestConfig(n_samples=1))
    result = bt.run(["UP"], dates)
    report = evaluate(bt, result, ["UP"])
    assert report.gate.passed is False
    assert "FAIL" in report.gate.verdict
    # After-cost alpha vs buy&hold ~ 0, vs 60/40 positive but not ALL beaten.
    names = {c.name: c.beat for c in report.gate.checks}
    assert names["buy&hold"] is False


def test_gate_certifiable_with_verified_clean_cutoff():
    store, dates = _downtrend_setup()
    model = _RatingModel({"DN": Rating.SELL})
    reg = CutoffRegistry({"const": {"cutoff": "2023-12-01", "verified": True}})
    bt = Backtester(store, model, cost_model=FREE, cutoff=reg, config=BacktestConfig(n_samples=1))
    result = bt.run(["DN"], dates)
    report = evaluate(bt, result, ["DN"])
    assert report.gate.passed is True
    assert report.gate.certifiable is True
    assert report.gate.contamination == "clean"


# --- LLM costs ----------------------------------------------------------------
def test_cost_tracker_counts_calls_and_cache_hits():
    store, dates = _downtrend_setup()
    model = _RatingModel({"DN": Rating.SELL})
    tracker = LLMCostTracker(est_cost_per_call_usd=0.01)
    cfg = BacktestConfig(n_samples=3)
    bt = Backtester(store, model, cost_model=FREE, config=cfg, llm_tracker=tracker)
    result = bt.run(["DN"], dates)

    n_calls = len(dates) * 3  # 1 ticker x 5 dates x 3 samples, all misses first run
    assert tracker.calls == n_calls
    assert tracker.cache_hits == 0
    assert tracker.total_cost_usd == pytest.approx(n_calls * 0.01)

    report = evaluate(bt, result, ["DN"], monthly_budget_usd=30.0)
    assert report.costs.cost_per_call_usd == pytest.approx(0.01)
    assert report.costs.n_decisions == len(dates)
    assert report.costs.cost_per_decision_usd == pytest.approx(0.03)
    assert report.costs.within_budget is True

    # Second run with the same cache: all hits, zero new cost.
    bt.run(["DN"], dates)
    assert tracker.cache_hits == n_calls
    assert tracker.calls == n_calls


def test_render_report_contains_mandated_fields():
    store, dates = _downtrend_setup()
    model = _RatingModel({"DN": Rating.SELL})
    bt = Backtester(store, model, cost_model=FREE, config=BacktestConfig(n_samples=1))
    result = bt.run(["DN"], dates)
    text = render_report(evaluate(bt, result, ["DN"], monthly_budget_usd=30.0))
    for needle in ("CAGR", "Sharpe", "Sortino", "Max drawdown", "Win rate",
                   "Avg win/loss", "Annual turnover", "buy&hold", "60/40",
                   "LLM 호출당 비용", "결정 1건당 총 비용", "After-cost alpha", "Gate"):
        assert needle in text, f"missing {needle}"
