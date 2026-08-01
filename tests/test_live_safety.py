"""Step 6 safety envelope: capital cap, arming, kill-switch liquidation, dry run."""

from __future__ import annotations

import pytest
from conftest import utc

from vts.backtest.costs import CostModel, CostParams
from vts.backtest.engine import BacktestConfig
from vts.decision import Decision, Rating
from vts.live import (
    LIVE_ARM_ENV,
    LIVE_ARM_TOKEN,
    MAX_INITIAL_CAPITAL_FRACTION,
    CapitalCapExceeded,
    LiveConfig,
    LiveNotArmed,
    LiveTrader,
    SimulatedBroker,
    assert_capital_within_cap,
    live_trading_armed,
    liquidate_all,
)
from vts.pit.schema import OHLCVBar
from vts.pit.store import PointInTimeStore
from vts.risk import RiskEngine, RiskLimits
from vts.risk.killswitch import KILL_SWITCH_ENV
from vts.risk.limits import US_EQUITY

FREE = CostModel(CostParams(0, 0, 0, 0))


def _bar(store, sym, day, close):
    t = utc(2024, 1, day, 21)
    store.append(OHLCVBar(symbol=sym, event_time=t, knowledge_time=t, source="t",
                          open=close, high=close, low=close, close=close, volume=1_000_000))
    return t


class _Buy:
    model_id = "buy"

    def prompt_for(self, ticker, clock):
        return f"buy:{ticker}@{clock.as_of.isoformat()}"

    def decide(self, ticker, clock):
        return Decision(ticker=ticker, as_of=clock.as_of.isoformat(),
                        rating=Rating.BUY, confidence=0.9)


def _trader(store, broker, *, live_config, risk=None, audit=None, state_path=None):
    return LiveTrader(
        store, _Buy(),
        risk or RiskEngine(RiskLimits(max_weight_per_symbol=1.0, max_drawdown_limit=None)),
        broker, venue=US_EQUITY, live_config=live_config, cost_model=FREE,
        config=BacktestConfig(n_samples=1), audit_path=audit, state_path=state_path,
    )


# --- hardcoded 1% capital cap -------------------------------------------------
def test_capital_cap_constant_is_one_percent():
    assert MAX_INITIAL_CAPITAL_FRACTION == 0.01


def test_capital_cap_boundary_and_breach():
    assert_capital_within_cap(1_000.0, 100_000.0)          # exactly 1% is allowed
    with pytest.raises(CapitalCapExceeded):
        assert_capital_within_cap(1_000.01, 100_000.0)     # a cent over is not


def test_capital_cap_rejects_nonpositive_inputs():
    with pytest.raises(CapitalCapExceeded):
        assert_capital_within_cap(0.0, 100_000.0)
    with pytest.raises(CapitalCapExceeded):
        assert_capital_within_cap(100.0, 0.0)               # no divide-into-permissive


def test_capital_cap_has_no_env_override(monkeypatch):
    """The cap is a module constant: no VTS_* var can raise it."""
    for var in ("VTS_MAX_INITIAL_CAPITAL_FRACTION", "VTS_LIVE_CAPITAL_FRACTION",
                "MAX_INITIAL_CAPITAL_FRACTION"):
        monkeypatch.setenv(var, "0.99")
    with pytest.raises(CapitalCapExceeded):
        assert_capital_within_cap(50_000.0, 100_000.0)


# --- arming gate ---------------------------------------------------------------
def test_dry_run_defaults_true():
    assert LiveConfig(allocated_capital=1_000.0).dry_run is True


def test_arming_requires_exact_token(monkeypatch):
    monkeypatch.setenv(LIVE_ARM_ENV, "1")
    assert live_trading_armed() is False          # a boolean is not enough
    monkeypatch.setenv(LIVE_ARM_ENV, "true")
    assert live_trading_armed() is False
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    assert live_trading_armed() is True


def test_assert_armed_requires_all_three_conditions(monkeypatch):
    monkeypatch.delenv(LIVE_ARM_ENV, raising=False)
    # (1) dry_run must be explicitly False
    with pytest.raises(LiveNotArmed, match="dry_run"):
        LiveConfig(allocated_capital=1_000.0).assert_armed_for_live(100_000.0)
    # (2) arming token must be present
    cfg = LiveConfig(allocated_capital=1_000.0, dry_run=False)
    with pytest.raises(LiveNotArmed, match=LIVE_ARM_ENV):
        cfg.assert_armed_for_live(100_000.0)
    # (3) capital must be within the cap
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    with pytest.raises(CapitalCapExceeded):
        LiveConfig(allocated_capital=5_000.0, dry_run=False).assert_armed_for_live(100_000.0)
    cfg.assert_armed_for_live(100_000.0)          # all three satisfied -> no raise


# --- kill-switch liquidation (전량 청산) ---------------------------------------
def test_liquidate_closes_longs_and_shorts_after_cancelling():
    broker = SimulatedBroker(positions={"A": 10.0, "B": -5.0},
                             prices={"A": 100.0, "B": 50.0}, open_orders=3)
    report = liquidate_all(broker, reason="test", dry_run=False)
    assert report.cancelled_orders == 3
    # Orders cancelled BEFORE closes, and a short is closed by BUYING.
    sides = {o.symbol: o.side for o in broker.submitted}
    assert sides == {"A": "sell", "B": "buy"}
    assert report.closed == {"A": 10.0, "B": 5.0}
    assert report.complete is True
    assert broker.get_positions() == []           # flat


def test_liquidate_dry_run_touches_nothing():
    broker = SimulatedBroker(positions={"A": 10.0}, prices={"A": 100.0}, open_orders=2)
    report = liquidate_all(broker, reason="test", dry_run=True)
    assert report.dry_run is True
    assert broker.submitted == []                   # nothing sent
    assert len(broker.get_positions()) == 1         # still held
    assert report.would_close == {"A": 10.0}        # a PLAN, kept distinct from `closed`
    assert report.closed == {}
    assert report.complete is False                 # a plan is never "complete"


def test_liquidate_is_idempotent_on_flat_account():
    broker = SimulatedBroker()
    report = liquidate_all(broker, reason="test", dry_run=False)
    assert report.closed == {} and report.complete is True


def test_liquidate_continues_past_a_failing_symbol():
    class _Flaky(SimulatedBroker):
        def submit_order(self, symbol, side, qty, *, limit_price):
            if symbol == "BAD":
                raise RuntimeError("venue rejected")
            return super().submit_order(symbol, side, qty, limit_price=limit_price)

    broker = _Flaky(positions={"BAD": 5.0, "GOOD": 7.0},
                    prices={"BAD": 10.0, "GOOD": 20.0})
    report = liquidate_all(broker, reason="test", dry_run=False)
    assert "GOOD" in report.closed                  # one bad symbol does not strand the rest
    assert "BAD" in report.failures
    assert report.complete is False


# --- LiveTrader: kill switch and halts ----------------------------------------
def test_cycle_kill_switch_liquidates_and_refuses_to_trade(monkeypatch, tmp_path):
    from vts.live import LiveState

    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    broker = SimulatedBroker(positions={"A": 10.0}, prices={"A": 100.0}, open_orders=1)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    # Liquidation is sleeve-scoped, so the ledger must show the position as ours.
    sp = tmp_path / "live.json"
    LiveState(sleeve_positions={"A": 10.0}).save(sp)
    trader = _trader(store, broker,
                     live_config=LiveConfig(allocated_capital=1_000.0, dry_run=False),
                     audit=tmp_path / "audit.log", state_path=sp)
    result = trader.run_cycle(["A"], date)
    assert result.halted is True
    assert result.liquidation is not None and result.liquidation.closed == {"A": 10.0}
    assert result.liquidation.complete is True      # verified flat, not assumed
    assert result.submitted == []                   # no strategy orders under the switch
    assert broker.get_positions() == []             # actually flat
    audit = (tmp_path / "audit.log").read_text()
    assert "LIQUIDATE_INTENT" in audit and "LIQUIDATE_RESULT" in audit


def test_cycle_latched_risk_halt_liquidates(monkeypatch, tmp_path):
    from vts.live import LiveState

    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    broker = SimulatedBroker(positions={"A": 10.0}, prices={"A": 100.0})
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    risk = RiskEngine(RiskLimits(daily_loss_limit=0.03, max_drawdown_limit=None))
    risk.observe_daily_return(-0.5)                 # latch the daily-loss halt
    sp = tmp_path / "live.json"
    LiveState(sleeve_positions={"A": 10.0}).save(sp)
    trader = _trader(store, broker,
                     live_config=LiveConfig(allocated_capital=1_000.0, dry_run=False),
                     risk=risk, state_path=sp)
    result = trader.run_cycle(["A"], date)
    assert result.halted is True
    assert broker.get_positions() == []


# --- LiveTrader: dry run vs armed live ----------------------------------------
def test_dry_run_cycle_sends_nothing(monkeypatch):
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    broker = SimulatedBroker(cash=1_000_000.0, prices={"A": 100.0})
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    trader = _trader(store, broker, live_config=LiveConfig(allocated_capital=10_000.0))
    result = trader.run_cycle(["A"], date)
    assert result.dry_run is True
    assert result.would_submit and not result.submitted
    assert broker.submitted == []                   # broker untouched


def test_armed_live_cycle_routes_within_allocated_capital(monkeypatch):
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    broker = SimulatedBroker(cash=1_000_000.0, prices={"A": 100.0})
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    trader = _trader(store, broker,
                     live_config=LiveConfig(allocated_capital=10_000.0, dry_run=False))
    result = trader.run_cycle(["A"], date)
    assert result.submitted and not result.would_submit
    # Sized against ALLOCATED capital (10k), never the 1M account.
    assert result.submitted[0].qty == 100           # 10_000 / 100
    assert broker.submitted[0].symbol == "A"


def test_live_cycle_without_arming_token_raises(monkeypatch):
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    broker = SimulatedBroker(cash=1_000_000.0, prices={"A": 100.0})
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.delenv(LIVE_ARM_ENV, raising=False)
    trader = _trader(store, broker,
                     live_config=LiveConfig(allocated_capital=10_000.0, dry_run=False))
    with pytest.raises(LiveNotArmed):
        trader.run_cycle(["A"], date)
    assert broker.submitted == []


def test_capital_cap_enforced_every_cycle(monkeypatch):
    """An account that shrinks below 100x the allocation stops trading."""
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    # allocation 10k but total assets only 500k -> cap is 5k -> breach.
    broker = SimulatedBroker(cash=500_000.0, total_assets=500_000.0, prices={"A": 100.0})
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    trader = _trader(store, broker, live_config=LiveConfig(allocated_capital=10_000.0))
    with pytest.raises(CapitalCapExceeded):
        trader.run_cycle(["A"], date)               # even in dry run the cap binds


def test_order_count_circuit_breaker(monkeypatch):
    store = PointInTimeStore()
    date = None
    for i, sym in enumerate(["A", "B", "C", "D"]):
        date = _bar(store, sym, 2, 100.0)
    broker = SimulatedBroker(cash=1_000_000.0,
                             prices={s: 100.0 for s in ["A", "B", "C", "D"]})
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    trader = _trader(store, broker, live_config=LiveConfig(
        allocated_capital=10_000.0, dry_run=False, max_orders_per_session=2))
    result = trader.run_cycle(["A", "B", "C", "D"], date)
    assert result.submitted == []
    assert any("max_orders_per_session" in n for n in result.notes)
    assert broker.submitted == []
