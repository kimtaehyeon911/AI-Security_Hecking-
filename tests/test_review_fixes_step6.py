"""Regression tests for the Step 6 adversarial-review findings (3 critical, 6 high)."""

from __future__ import annotations

import math

import pytest
from conftest import utc
from pydantic import ValidationError

from vts.backtest.costs import CostModel, CostParams
from vts.backtest.engine import BacktestConfig
from vts.decision import Decision, Rating
from vts.live import (
    LIVE_ARM_ENV,
    LIVE_ARM_TOKEN,
    Account,
    CapitalCapExceeded,
    LiveConfig,
    LiveState,
    LiveTrader,
    SimulatedBroker,
    assert_capital_within_cap,
    liquidate_all,
)
from vts.live.broker import BrokerOrder
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


def _trader(store, broker, live_config, *, risk=None, state_path=None, audit=None):
    return LiveTrader(
        store, _Buy(),
        risk or RiskEngine(RiskLimits(max_weight_per_symbol=1.0, max_drawdown_limit=None)),
        broker, venue=US_EQUITY, live_config=live_config, cost_model=FREE,
        config=BacktestConfig(n_samples=1), audit_path=audit, state_path=state_path,
    )


# === CRITICAL 1: non-finite total_assets must not disable the cap =============
def test_nan_total_assets_fails_closed():
    for total in (math.nan, math.inf, -math.inf):
        with pytest.raises(CapitalCapExceeded):
            assert_capital_within_cap(500_000.0, total)


def test_nan_allocated_fails_closed():
    with pytest.raises(CapitalCapExceeded):
        assert_capital_within_cap(math.nan, 1_000_000.0)


def test_money_models_reject_non_finite():
    with pytest.raises(ValidationError):
        Account(cash=1.0, total_assets=math.nan)
    with pytest.raises(ValidationError):
        LiveConfig(allocated_capital=math.inf)     # gt=0 alone would accept inf


def test_nan_account_cannot_route_a_live_order(monkeypatch):
    """End-to-end: the repro that deployed 50% of the account must now be refused."""
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    # A broker reporting NaN equity can no longer even construct an Account.
    broker = SimulatedBroker(cash=1_000_000.0, total_assets=math.nan, prices={"A": 100.0})
    trader = _trader(store, broker,
                     LiveConfig(allocated_capital=500_000.0, dry_run=False))
    with pytest.raises((CapitalCapExceeded, ValidationError)):
        trader.run_cycle(["A"], date)
    assert broker.submitted == []


# === CRITICAL 2: liquidation must verify flat ================================
class _NoFillBroker(SimulatedBroker):
    """Accepts closes but never actually reduces the position (partial-fill/reject)."""

    def submit_order(self, symbol, side, qty, *, limit_price):
        self._seq += 1
        return BrokerOrder(symbol=symbol.upper(), side=side, qty=qty,
                           limit_price=limit_price, order_id=f"x-{self._seq}")


def test_liquidation_incomplete_when_position_remains():
    broker = _NoFillBroker(positions={"A": 10.0}, prices={"A": 100.0})
    report = liquidate_all(broker, reason="test", dry_run=False)
    assert report.residual == {"A": 10.0}
    assert report.complete is False          # was True before: "no exception" != flat


def test_broker_rejected_status_is_not_success():
    class _Rejecting(SimulatedBroker):
        def submit_order(self, symbol, side, qty, *, limit_price):
            return BrokerOrder(symbol=symbol.upper(), side=side, qty=qty,
                               limit_price=limit_price, status="rejected")

    broker = _Rejecting(positions={"A": 5.0}, prices={"A": 10.0})
    report = liquidate_all(broker, reason="t", dry_run=False)
    assert "A" in report.failures and report.complete is False


def test_dry_run_liquidation_is_never_complete():
    broker = SimulatedBroker(positions={"A": 10.0}, prices={"A": 100.0})
    report = liquidate_all(broker, reason="t", dry_run=True)
    assert report.complete is False           # a plan is not an execution
    assert report.would_close == {"A": 10.0}
    assert report.closed == {}
    assert broker.submitted == []


def test_cancel_failure_does_not_abort_the_closes():
    class _BadCancel(SimulatedBroker):
        def cancel_all_orders(self):
            raise RuntimeError("venue down")

    broker = _BadCancel(positions={"A": 10.0}, prices={"A": 100.0})
    report = liquidate_all(broker, reason="t", dry_run=False)
    assert "__cancel_all_orders__" in report.failures
    assert report.closed == {"A": 10.0}       # closes still ran (the code matches its comment)
    assert broker.get_positions() == []


def test_liquidation_never_crosses_zero():
    broker = SimulatedBroker(positions={"A": 10.0}, prices={"A": 100.0})
    liquidate_all(broker, reason="t", dry_run=False)
    liquidate_all(broker, reason="t", dry_run=False)   # repeated call on a flat book
    assert broker.get_positions() == []               # not flipped short
    assert all(o.qty == 10.0 for o in broker.submitted)


# === CRITICAL 3: limit price must reach the venue ============================
def test_limit_price_is_sent_to_the_broker(monkeypatch):
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    broker = SimulatedBroker(cash=1_000_000.0, prices={"A": 100.0})
    trader = _trader(store, broker, LiveConfig(allocated_capital=10_000.0, dry_run=False))
    trader.run_cycle(["A"], date)
    assert broker.submitted[0].limit_price == 100.0   # not an implicit market order


# === CRITICAL 4: no duplicate exposure from unfilled orders ==================
def test_working_orders_are_netted_not_duplicated(monkeypatch):
    store = PointInTimeStore()
    d1 = _bar(store, "A", 2, 100.0)
    d2 = _bar(store, "A", 3, 100.0)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    broker = SimulatedBroker(cash=1_000_000.0, prices={"A": 100.0}, fill=False)  # never fills
    trader = _trader(store, broker, LiveConfig(allocated_capital=10_000.0, dry_run=False))
    trader.run_cycle(["A"], d1)
    assert len(broker.submitted) == 1
    trader.run_cycle(["A"], d2)               # order still working -> must NOT re-send
    assert len(broker.submitted) == 1


# === HIGH: never touch the operator's pre-existing book ======================
def test_stray_broker_positions_are_never_liquidated(monkeypatch, tmp_path):
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    # The account holds a big unrelated position the sleeve never bought.
    broker = SimulatedBroker(positions={"LEGACY": 5000.0}, prices={"LEGACY": 300.0})
    trader = _trader(store, broker,
                     LiveConfig(allocated_capital=1_000.0, dry_run=False,
                                allow_unarmed_liquidation=True),
                     state_path=tmp_path / "live.json")
    result = trader.run_cycle(["A"], date)
    assert result.halted is True
    assert broker.submitted == []                       # LEGACY untouched
    assert broker.get_positions()[0].symbol == "LEGACY"


# === HIGH: durable halt latch ================================================
def test_halt_latch_survives_restart(monkeypatch, tmp_path):
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    broker = SimulatedBroker(cash=1_000_000.0, prices={"A": 100.0})
    sp = tmp_path / "live.json"
    _trader(store, broker, LiveConfig(allocated_capital=1_000.0, dry_run=False,
                                      allow_unarmed_liquidation=True),
            state_path=sp).run_cycle(["A"], date)
    assert LiveState.load_or_new(sp).halted is True

    # Restart with the kill switch CLEARED — a fresh engine must stay halted.
    monkeypatch.delenv(KILL_SWITCH_ENV)
    revived = _trader(store, broker,
                      LiveConfig(allocated_capital=1_000.0, dry_run=False),
                      risk=RiskEngine(RiskLimits(max_drawdown_limit=None)),
                      state_path=sp)
    assert revived.risk.halt.halted is True
    result = revived.run_cycle(["A"], date)
    assert result.halted is True
    assert broker.submitted == []             # did not resume trading


# === HIGH: unarmed liquidation only reports unless explicitly allowed ========
def test_unarmed_liquidation_reports_but_does_not_act(monkeypatch, tmp_path):
    store = PointInTimeStore()
    date = _bar(store, "A", 2, 100.0)
    monkeypatch.delenv(LIVE_ARM_ENV, raising=False)     # NOT armed
    monkeypatch.setenv(KILL_SWITCH_ENV, "1")
    broker = SimulatedBroker(positions={"A": 10.0}, prices={"A": 100.0})
    state = LiveState(sleeve_positions={"A": 10.0})
    sp = tmp_path / "live.json"
    state.save(sp)
    trader = _trader(store, broker,
                     LiveConfig(allocated_capital=1_000.0, dry_run=False),
                     state_path=sp)
    result = trader.run_cycle(["A"], date)
    assert result.halted is True
    assert broker.submitted == []                        # reported only
    assert any("unarmed" in n for n in result.notes)


# === MEDIUM: aggregate notional cannot exceed the allocation =================
def test_aggregate_buy_notional_capped(monkeypatch):
    store = PointInTimeStore()
    date = None
    for sym in ("A", "B", "C"):
        date = _bar(store, sym, 2, 100.0)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)
    broker = SimulatedBroker(cash=10_000_000.0, prices={s: 100.0 for s in "ABC"})
    # max_gross 3.0 would let the risk engine ask for 3x the allocation.
    risk = RiskEngine(RiskLimits(max_weight_per_symbol=1.0, max_gross_exposure=2.0,
                                 max_drawdown_limit=None))
    trader = _trader(store, broker,
                     LiveConfig(allocated_capital=10_000.0, dry_run=False), risk=risk)
    result = trader.run_cycle(["A", "B", "C"], date)
    assert broker.submitted == []
    assert any("aggregate buy notional" in n for n in result.notes)


# === MEDIUM: submit failure preserves the record of what went live ===========
def test_submit_error_returns_partial_result(monkeypatch):
    store = PointInTimeStore()
    date = None
    for sym in ("A", "B"):
        date = _bar(store, sym, 2, 100.0)
    monkeypatch.delenv(KILL_SWITCH_ENV, raising=False)
    monkeypatch.setenv(LIVE_ARM_ENV, LIVE_ARM_TOKEN)

    class _FailsSecond(SimulatedBroker):
        def submit_order(self, symbol, side, qty, *, limit_price):
            if len(self.submitted) >= 1:
                raise RuntimeError("connection reset")
            return super().submit_order(symbol, side, qty, limit_price=limit_price)

    broker = _FailsSecond(cash=1_000_000.0, prices={"A": 100.0, "B": 100.0})
    risk = RiskEngine(RiskLimits(max_weight_per_symbol=0.5, max_drawdown_limit=None))
    trader = _trader(store, broker,
                     LiveConfig(allocated_capital=10_000.0, dry_run=False), risk=risk)
    result = trader.run_cycle(["A", "B"], date)     # must not raise
    assert len(result.submitted) == 1                # the one that reached the venue
    assert any("submit failed" in n for n in result.notes)
