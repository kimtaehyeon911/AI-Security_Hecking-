"""Regression tests for the Step 2 adversarial-review findings."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import utc

from vts.backtest.cutoff import Contamination, CutoffRegistry
from vts.backtest.engine import Backtester, BacktestConfig
from vts.backtest.model import FakeMomentumModel, MomentumParams
from vts.backtest.costs import CostModel, CostParams
from vts.decision import Decision, Rating, aggregate_decisions
from vts.integration.ta_decision import rating_from_text
from vts.pit.clock import AsOfClock
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore


# --- HIGH: rating_from_text anchors to the labeled value ---------------------
def test_rating_label_anchored_not_first_word():
    assert rating_from_text("Sell-off risk remains; Rating: Hold") == Rating.HOLD
    assert rating_from_text("Downgrade from Buy. Rating: Underweight") == Rating.UNDERWEIGHT
    assert rating_from_text("**Rating**: Overweight") == Rating.OVERWEIGHT


def test_rating_freeform_fallback_when_no_label():
    assert rating_from_text("On balance we recommend a Sell") == Rating.SELL
    assert rating_from_text("no verdict here") == Rating.HOLD


# --- LOW: tie agreement reflects the emitted (Hold) rating -------------------
def test_tie_agreement_is_hold_fraction():
    def d(r):
        return Decision(ticker="AAPL", as_of="t", rating=r, confidence=0.5)
    agg = aggregate_decisions([d(Rating.BUY), d(Rating.BUY), d(Rating.SELL), d(Rating.SELL), d(Rating.HOLD)])
    assert agg.rating == Rating.HOLD
    assert agg.tie_broken_to_hold is True
    assert agg.agreement == pytest.approx(1 / 5)  # Hold count, not the top count 2/5


# --- MEDIUM: malformed verified cutoff degrades to UNKNOWN (no crash) --------
def test_malformed_cutoff_degrades_to_unknown():
    reg = CutoffRegistry({"m": {"cutoff": "2023/13/40", "verified": True}})
    assert reg.classify("m", utc(2024, 1, 1)) == Contamination.UNKNOWN  # does not raise


# --- MEDIUM: prompt_for encodes all params -> cache busts on param change ----
def test_prompt_for_encodes_thresholds():
    store = PointInTimeStore()
    clock = AsOfClock.at(utc(2024, 6, 3, 21))
    a = FakeMomentumModel(store, MomentumParams(strong_threshold=0.10))
    b = FakeMomentumModel(store, MomentumParams(strong_threshold=0.15))
    assert a.prompt_for("AAPL", clock) != b.prompt_for("AAPL", clock)


# --- MEDIUM: drift is charged; benchmark is a true buy&hold ------------------
class _ConstModel:
    model_id = "const-buy"

    def prompt_for(self, ticker, clock):
        return f"const:{ticker}@{clock.as_of.isoformat()}"

    def decide(self, ticker, clock):
        return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                        rating=Rating.BUY, confidence=0.9)


def _bar(sym, day, close):
    t = utc(2024, 1, day, 21)
    return OHLCVBar(symbol=sym, event_time=t, knowledge_time=t, source="t",
                    open=close, high=close, low=close, close=close, volume=1_000_000)


def test_drift_incurs_turnover_cost_even_with_unchanged_target():
    store = PointInTimeStore()
    # Two tickers; A doubles between the two decision dates, B flat. Target stays 50/50.
    store.append_many([_bar("A", 2, 100), _bar("A", 9, 200)])
    store.append_many([_bar("B", 2, 100), _bar("B", 9, 100)])
    bt = Backtester(store, _ConstModel(), config=BacktestConfig(n_samples=1))
    result = bt.run(["A", "B"], [utc(2024, 1, 2, 21), utc(2024, 1, 9, 21)])
    # Second rebalance must charge cost to correct the drift back toward 50/50.
    assert result.records[1].turnover > 0
    assert result.records[1].cost > 0


def test_benchmark_is_true_buy_and_hold_single_asset():
    store = PointInTimeStore()
    store.append_many([_bar("A", 2, 100), _bar("A", 9, 100), _bar("A", 16, 150)])
    bt = Backtester(store, _ConstModel(),
                    cost_model=CostModel(CostParams(0, 0, 0, 0)),
                    config=BacktestConfig(n_samples=1))
    dates = [utc(2024, 1, 2, 21), utc(2024, 1, 9, 21), utc(2024, 1, 16, 21)]
    result = bt.run(["A"], dates)
    # Buy&hold a single asset that goes 100 -> 150 => benchmark up 50% from entry.
    entry = result.benchmark_curve[0][1]
    end = result.benchmark_curve[-1][1]
    assert end / entry == pytest.approx(1.5, rel=1e-6)
