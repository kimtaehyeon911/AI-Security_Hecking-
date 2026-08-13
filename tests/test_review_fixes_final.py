"""Regression tests for the final wrap-up review findings (round 8)."""

from __future__ import annotations

from datetime import timedelta

import pytest
from conftest import utc

from vts import cli
from vts.backtest.cache import DecisionCache, decision_key
from vts.backtest.costs import CostModel, CostParams
from vts.backtest.engine import BacktestConfig
from vts.backtest.model import FakeMomentumModel
from vts.decision import Decision, Rating
from vts.live import LiveConfig, LiveState, LiveTrader, SimulatedBroker
from vts.live.config import LIVE_ARM_ENV, LIVE_ARM_TOKEN
from vts.pit.clock import AsOfClock
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

    def prompt_for(self, ticker, clock):
        return f"buy:{ticker}@{clock.as_of.isoformat()}"

    def decide(self, ticker, clock):
        return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                        rating=Rating.BUY, confidence=0.9)


def _live_trader(store, broker, tmp_path, *, risk=None):
    return LiveTrader(
        store, _Buy(),
        risk or RiskEngine(RiskLimits(daily_loss_limit=0.03, max_weight_per_symbol=1.0,
                                      max_drawdown_limit=None)),
        broker, venue=US_EQUITY,
        live_config=LiveConfig(allocated_capital=1_000.0, dry_run=False),
        cost_model=FREE, config=BacktestConfig(n_samples=1),
        state_path=tmp_path / "live.json",
    )


# === HIGH: live daily-loss halt must latch from genuine daily marks ==========
def test_live_daily_loss_halt_latches_and_liquidates(monkeypatch, tmp_path):
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    store = PointInTimeStore()
    # Day1 100 -> buy; day2 crashes -10% (breaches the 3% daily limit).
    dates = _daily(store, "A", utc(2025, 1, 2), [100.0, 90.0])
    broker = SimulatedBroker(cash=1_000_000.0, prices={"A": 100.0})

    t1 = _live_trader(store, broker, tmp_path)
    r1 = t1.run_cycle(["A"], dates[0])
    assert r1.submitted and r1.halted is False        # entered the position

    # NEW PROCESS (fresh engine): cycle 2 must latch from the -10% daily mark,
    # liquidate the sleeve, and persist the latch.
    t2 = _live_trader(store, broker, tmp_path)
    r2 = t2.run_cycle(["A"], dates[1])
    assert r2.halted is True
    assert r2.liquidation is not None
    st = LiveState.load_or_new(tmp_path / "live.json")
    assert st.halted is True
    assert any("daily_loss_limit" in x for x in st.halt_reasons)


def test_live_drawdown_peak_persists_across_processes(monkeypatch, tmp_path):
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    store = PointInTimeStore()
    # Slow bleed: -2%/day, each under the 3% daily limit, drawdown crosses 10%.
    closes = [100.0 * (0.98 ** i) for i in range(8)]
    dates = _daily(store, "A", utc(2025, 1, 2), closes)
    broker = SimulatedBroker(cash=1_000_000.0, prices={"A": 100.0})

    halted_at = None
    for i, d in enumerate(dates):
        trader = _live_trader(
            store, broker, tmp_path,
            risk=RiskEngine(RiskLimits(daily_loss_limit=0.5,   # daily stop out of the way
                                       max_weight_per_symbol=1.0,
                                       max_drawdown_limit=0.10)),
        )
        result = trader.run_cycle(["A"], d)
        if result.halted:
            halted_at = i
            break
    # Peak/equity persisted across fresh engines -> cumulative stop fires.
    assert halted_at is not None and halted_at >= 2


# === MED: decision calendar is the union across the universe =================
def test_decision_dates_union_and_warning(tmp_path, capsys, monkeypatch):
    store = PointInTimeStore()
    _daily(store, "SPARSE", utc(2025, 1, 2), [100.0] * 3)          # 3 days only
    _daily(store, "FULL", utc(2025, 1, 2), [100.0] * 10)           # 10 days
    dates = cli._decision_dates(store, ["SPARSE", "FULL"],
                                utc(2025, 1, 1), utc(2025, 1, 20))
    assert len(dates) == 10                                        # union, not anchor
    err = capsys.readouterr().err
    assert "SPARSE covers 3/10" in err                             # loud gap warning


# === MED: re-ingested data busts the decision cache ==========================
def test_backfill_busts_momentum_cache():
    store = PointInTimeStore()
    dates = _daily(store, "A", utc(2025, 1, 2), [100.0 + i for i in range(25)])
    model = FakeMomentumModel(store)
    clock = AsOfClock.at(dates[-1])
    key_before = decision_key("A", clock.as_of.isoformat(), model.model_id,
                              model.prompt_for("A", clock), 0)
    # Backfill an earlier missing bar (append-only store, past knowledge_time is
    # not required for the fingerprint to change — bar COUNT changes).
    t = utc(2024, 12, 20, 21)
    store.append(OHLCVBar(symbol="A", event_time=t, knowledge_time=t, source="t",
                          open=90, high=90, low=90, close=90, volume=1))
    key_after = decision_key("A", clock.as_of.isoformat(), model.model_id,
                             model.prompt_for("A", clock), 0)
    assert key_before != key_after                                 # stale cache busted


# === CLI: --no-cache and cache-clear ==========================================
def test_cli_cache_clear(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("VTS_DATA_DIR", str(tmp_path))
    cache_path = tmp_path / "decision_cache.sqlite"
    DecisionCache(cache_path).close()
    assert cache_path.exists()
    assert cli.main(["cache-clear"]) == 0
    assert not cache_path.exists()
    assert cli.main(["cache-clear"]) == 0                          # idempotent
