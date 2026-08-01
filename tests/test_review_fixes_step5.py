"""Regression tests for the Step 5 adversarial-review findings."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import utc

from vts.backtest.costs import CostModel, CostParams
from vts.backtest.engine import BacktestConfig
from vts.decision import Decision, Rating
from vts.paper import PaperState, PaperTrader
from vts.paper.shortfall import make_record
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore
from vts.risk import RiskEngine, RiskLimits
from vts.risk.limits import US_EQUITY

FREE = CostModel(CostParams(0, 0, 0, 0))


def _daily(store, symbol, start, closes):
    dates = []
    for i, c in enumerate(closes):
        t = (start + timedelta(days=i)).replace(hour=21)
        store.append(OHLCVBar(symbol=symbol, event_time=t, knowledge_time=t, source="t",
                              open=c, high=c, low=c, close=c, volume=1_000_000))
        dates.append(t)
    return dates


class _Buy:
    model_id = "buy"

    def prompt_for(self, ticker, clock):
        return f"buy:{ticker.upper()}@{clock.as_of.isoformat()}"

    def decide(self, ticker, clock):
        return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                        rating=Rating.BUY, confidence=0.9)


def _trader(tmp_path, store, *, risk=None, model=None, name="paper.json"):
    return PaperTrader(
        store, model or _Buy(),
        risk or RiskEngine(RiskLimits(max_weight_per_symbol=0.5, max_drawdown_limit=None)),
        venue=US_EQUITY, state_path=tmp_path / name, cost_model=FREE,
        config=BacktestConfig(n_samples=1, initial_capital=100_000.0),
    )


# --- HIGH: cumulative_shortfall no longer a monotonic artifact ---------------
def test_gap_is_current_level_not_running_sum():
    # A persistent 1000 gap must report gap==1000 every day, not 1000*N.
    prior = 0.0
    for _ in range(40):
        rec = make_record("2024-01-01T21:00:00+00:00", 99_000.0, 100_000.0, 100_000.0, prior)
        assert rec.gap == pytest.approx(1000.0)          # total drift, constant
        prior = rec.gap
    # daily increment is 0 after the gap first appears (it is not widening).
    rec2 = make_record("d2", 99_000.0, 100_000.0, 100_000.0, 1000.0)
    assert rec2.daily_shortfall == pytest.approx(0.0)


# --- HIGH: symbol case asymmetry -> no phantom re-buys / correct marks --------
def test_lowercase_symbol_positions_track_correctly(tmp_path):
    store = PointInTimeStore()
    dates = _daily(store, "BRK.B", utc(2024, 1, 2), [100.0] * 6)  # flat price
    trader = _trader(tmp_path, store)
    state = trader.run(["brk.b"], dates)                # lowercase universe
    # One position built and then held flat — NOT re-bought from a phantom zero base.
    assert set(state.positions) == {"BRK.B"}
    shares = state.positions["BRK.B"]
    # Flat price + steady target => position stable after entry (no daily re-buy).
    assert state.decision_log[-1]["fills"] == 0
    assert shares > 0
    # Equity is marked correctly (not collapsed to cash-only).
    assert state.equity_curve[-1][1] == pytest.approx(100_000.0, rel=1e-6)


# --- HIGH: held name leaving the universe is liquidated, not stranded/zero ----
def test_out_of_universe_holding_is_liquidated(tmp_path):
    store = PointInTimeStore()
    _daily(store, "A", utc(2024, 1, 2), [100.0] * 6)
    _daily(store, "B", utc(2024, 1, 2), [100.0] * 6)
    dates = _daily(store, "_ANCHOR", utc(2024, 1, 2), [100.0] * 6)  # dummy for date axis

    trader = _trader(tmp_path, store)
    trader.step(["A", "B"], dates[0], None)
    assert "A" in trader.state.positions
    # Next day the universe drops A -> it must be sold, not stranded or zero-marked.
    trader.step(["B"], dates[1], None)
    assert "A" not in trader.state.positions
    assert trader.state.halted is False   # no false halt from a zero mark


# --- HIGH: atomic save leaves no partial file --------------------------------
def test_save_is_atomic_and_leaves_no_tmp(tmp_path):
    path = tmp_path / "paper.json"
    st = PaperState.new(100_000.0)
    st.save(path)
    assert path.exists()
    assert not (tmp_path / "paper.json.tmp").exists()
    PaperState.load(path)  # valid JSON round-trips


# --- MED: identical trades -> ~0 gap (pre/post-cost aligned) ------------------
def test_identical_book_reports_near_zero_gap(tmp_path):
    store = PointInTimeStore()
    # Price divisible into whole shares so paper matches the fractional backtest.
    dates = _daily(store, "A", utc(2024, 1, 2), [100.0 + i for i in range(8)])
    risk = RiskEngine(RiskLimits(max_weight_per_symbol=1.0, max_drawdown_limit=None))
    trader = _trader(tmp_path, store, risk=risk)
    state = trader.run(["A"], dates)
    # With free costs and near-whole-share sizing the gap is small (bps), not one
    # day's turnover cost every day.
    gaps = [r["gap"] for r in state.shortfall_log if r["gap"] is not None]
    assert all(abs(g) < 0.02 * 100_000 for g in gaps)   # < 2% of capital
    assert state.shortfall_log[-1]["reference_halted"] is False


# --- MED: halt rehydration guard ---------------------------------------------
def test_halted_engine_with_unhalted_state_raises(tmp_path):
    store = PointInTimeStore()
    _daily(store, "A", utc(2024, 1, 2), [100.0, 101.0])
    risk = RiskEngine(RiskLimits(daily_loss_limit=0.03, max_drawdown_limit=None))
    risk.observe_daily_return(-0.5)      # latch the engine BEFORE handing it in
    assert risk.halt.halted
    with pytest.raises(RuntimeError, match="halted RiskEngine"):
        PaperTrader(store, _Buy(), risk, venue=US_EQUITY,
                    state_path=tmp_path / "paper.json", cost_model=FREE,
                    config=BacktestConfig(n_samples=1))


# --- MED: sell-before-buy funds a rotation -----------------------------------
def test_rotation_sells_before_buys(tmp_path):
    store = PointInTimeStore()
    # Day 1: only A priced -> go long A. Day 2: rotate to B (A's cash funds B).
    _daily(store, "A", utc(2024, 1, 2), [100.0, 100.0])
    # B only prices on day 2.
    store.append(OHLCVBar(symbol="B", event_time=utc(2024, 1, 3, 21),
                          knowledge_time=utc(2024, 1, 3, 21), source="t",
                          open=100, high=100, low=100, close=100, volume=1_000_000))
    dates = [utc(2024, 1, 2, 21), utc(2024, 1, 3, 21)]

    class _RotateModel:
        model_id = "rot"

        def prompt_for(self, ticker, clock):
            return f"rot:{ticker}@{clock.as_of.isoformat()}"

        def decide(self, ticker, clock):
            # Day 1 buy A; day 2 buy B, sell A (B rated Buy, A rated Sell).
            day = clock.as_of.day
            rating = Rating.BUY
            if day == 3 and ticker == "A":
                rating = Rating.SELL
            return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                            rating=rating, confidence=0.9)

    risk = RiskEngine(RiskLimits(max_weight_per_symbol=1.0, max_drawdown_limit=None))
    trader = _trader(tmp_path, store, risk=risk, model=_RotateModel())
    state = trader.run(["A", "B"], dates)
    # After rotation the book holds B (funded by A's sale), not stuck in cash.
    assert "B" in state.positions and state.positions.get("A", 0) == 0
