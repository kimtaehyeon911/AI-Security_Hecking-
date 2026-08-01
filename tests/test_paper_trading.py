"""Paper-trading loop: dry-run guard, 8-week run, daily shortfall, resume, halts."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import utc

from vts.backtest.costs import CostModel, CostParams
from vts.backtest.engine import BacktestConfig
from vts.decision import Decision, Rating
from vts.paper import LiveTradingNotEnabled, PaperBroker, PaperState, PaperTrader
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore
from vts.risk import RiskEngine, RiskLimits
from vts.risk.killswitch import KILL_SWITCH_ENV
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

    def __init__(self, confidence=0.9):
        self._c = confidence

    def prompt_for(self, ticker, clock):
        return f"buy:{ticker}@{clock.as_of.isoformat()}"

    def decide(self, ticker, clock):
        return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                        rating=Rating.BUY, confidence=self._c)


def _trader(tmp_path, store, dates_len, *, risk=None, dry_run=True, model=None):
    return PaperTrader(
        store, model or _Buy(),
        risk or RiskEngine(RiskLimits(max_weight_per_symbol=0.5, max_drawdown_limit=None)),
        venue=US_EQUITY, state_path=tmp_path / "paper.json",
        cost_model=FREE, config=BacktestConfig(n_samples=1, initial_capital=100_000.0),
        dry_run=dry_run,
    )


# --- dry-run guard ------------------------------------------------------------
def test_paper_broker_defaults_dry_run_and_refuses_live():
    assert PaperBroker(FREE).dry_run is True
    with pytest.raises(LiveTradingNotEnabled):
        PaperBroker(FREE, dry_run=False)


def test_paper_state_default_dry_run_true():
    assert PaperState.new(100_000.0).dry_run is True


# --- 8-week run + daily shortfall log -----------------------------------------
def test_eight_week_run_logs_shortfall_daily(tmp_path):
    store = PointInTimeStore()
    # 44 sequential daily bars (> 8 trading weeks) with a gentle uptrend.
    closes_a = [100.0 * (1.005 ** i) for i in range(44)]
    closes_b = [50.0 * (1.003 ** i) for i in range(44)]
    dates = _daily(store, "A", utc(2024, 1, 2), closes_a)
    _daily(store, "B", utc(2024, 1, 2), closes_b)

    trader = _trader(tmp_path, store, len(dates))
    state = trader.run(["A", "B"], dates)

    assert len(state.processed_dates) == 44           # >= 40 trading days (8 weeks)
    assert len(state.shortfall_log) == 44
    # Every day has an implementation-shortfall observation vs the reference backtest.
    assert all(r["backtest_equity"] is not None for r in state.shortfall_log)
    assert all(r["shortfall"] is not None for r in state.shortfall_log)
    # Paper actually traded (long the uptrend).
    assert state.positions
    assert state.equity_curve[-1][1] > 100_000  # made money gross


def test_paper_decisions_match_reference_pipeline(tmp_path):
    store = PointInTimeStore()
    dates = _daily(store, "A", utc(2024, 1, 2), [100.0 + i for i in range(42)])
    trader = _trader(tmp_path, store, len(dates))
    state = trader.run(["A"], dates)
    # Same decision pipeline -> every day rated Buy (the model is constant Buy).
    assert all(d["ratings"]["A"] == "Buy" for d in state.decision_log)


# --- persistence / resume -----------------------------------------------------
def test_resume_continues_without_reprocessing(tmp_path):
    store = PointInTimeStore()
    dates = _daily(store, "A", utc(2024, 1, 2), [100.0 + i for i in range(42)])

    first = _trader(tmp_path, store, len(dates))
    for d in dates[:20]:
        first.step(["A"], d, None)
    assert len(first.state.processed_dates) == 20

    # New process: reload state, run the full range — only the remaining days run.
    resumed = _trader(tmp_path, store, len(dates))
    assert len(resumed.state.processed_dates) == 20  # loaded from disk
    resumed.run(["A"], dates)
    assert len(resumed.state.processed_dates) == 42
    # No date processed twice.
    assert len(set(resumed.state.processed_dates)) == 42


def test_resume_dry_run_mismatch_raises(tmp_path):
    store = PointInTimeStore()
    dates = _daily(store, "A", utc(2024, 1, 2), [100.0, 101.0])
    _trader(tmp_path, store, len(dates)).step(["A"], dates[0], None)
    with pytest.raises(ValueError, match="dry_run"):
        PaperState.load_or_new(tmp_path / "paper.json", 100_000.0, dry_run=False)


# --- risk integration in the live path ----------------------------------------
def test_kill_switch_flattens_paper_not_raises(tmp_path, monkeypatch):
    store = PointInTimeStore()
    dates = _daily(store, "A", utc(2024, 1, 2), [100.0 + i for i in range(5)])
    trader = _trader(tmp_path, store, len(dates))
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    state = trader.run(["A"], dates)  # paper flattens (does not raise like backtest)
    assert all(d["halted"] for d in state.decision_log)
    assert not state.positions  # never took a position under the engaged kill switch


def test_daily_loss_halt_persists_across_restart(tmp_path):
    store = PointInTimeStore()
    # Day 3 crashes -30% -> daily-loss halt latches.
    dates = _daily(store, "A", utc(2024, 1, 2), [100.0, 100.0, 70.0, 90.0, 110.0])
    risk = RiskEngine(RiskLimits(daily_loss_limit=0.03, max_weight_per_symbol=1.0,
                                 max_drawdown_limit=None))
    trader = _trader(tmp_path, store, len(dates), risk=risk)
    for d in dates[:3]:
        trader.step(["A"], d, None)
    assert trader.state.halted is True

    # Restart: a fresh RiskEngine + trader must load the latched halt and stay flat.
    resumed = _trader(tmp_path, store, len(dates),
                      risk=RiskEngine(RiskLimits(daily_loss_limit=0.03,
                                                 max_weight_per_symbol=1.0,
                                                 max_drawdown_limit=None)))
    assert resumed.risk.halt.halted is True
    resumed.run(["A"], dates)
    # No new positions opened after the persisted halt.
    assert not resumed.state.positions


def test_low_confidence_never_trades(tmp_path):
    store = PointInTimeStore()
    dates = _daily(store, "A", utc(2024, 1, 2), [100.0 + i for i in range(10)])
    risk = RiskEngine(RiskLimits(min_confidence=0.5, max_drawdown_limit=None))
    trader = _trader(tmp_path, store, len(dates), risk=risk, model=_Buy(confidence=0.1))
    state = trader.run(["A"], dates)
    assert not state.positions
    assert all(d["forced_holds"] == 1 for d in state.decision_log)
