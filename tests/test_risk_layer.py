"""Step 4 risk layer: gates, limits, halt latch, kill switch, order validation."""

from __future__ import annotations

import pytest
from conftest import utc
from pydantic import ValidationError

from vts.decision import AggregatedDecision, Rating
from vts.risk import (
    AccountState,
    Order,
    RiskEngine,
    RiskLimits,
    gate_decision,
    validate_order,
)
from vts.risk.gates import hold_fallback
from vts.risk.killswitch import KILL_SWITCH_ENV, HaltState, kill_switch_active
from vts.risk.limits import KRX_KOSPI, US_EQUITY, VenueRules


def _agg(rating=Rating.BUY, conf=0.9, agree=1.0, disp=0.0, tie=False) -> AggregatedDecision:
    return AggregatedDecision(
        ticker="AAPL", as_of="2024-06-03T21:00:00+00:00", rating=rating, n_samples=3,
        agreement=agree, dispersion=disp, mean_confidence=conf,
        votes={rating.value: 3}, tie_broken_to_hold=tie,
    )


# --- limits are frozen constants, never agent-writable ------------------------
def test_limits_frozen_and_reject_unknown_fields():
    limits = RiskLimits()
    with pytest.raises(ValidationError):
        limits.max_weight_per_symbol = 0.99  # frozen
    with pytest.raises(ValidationError):
        RiskLimits(llm_suggested_limit=5.0)  # extra=forbid


def test_limits_from_env(monkeypatch):
    monkeypatch.setenv("VTS_RISK_MAX_WEIGHT", "0.10")
    monkeypatch.setenv("VTS_RISK_DAILY_LOSS_LIMIT", "0.02")
    limits = RiskLimits.from_env()
    assert limits.max_weight_per_symbol == 0.10
    assert limits.daily_loss_limit == 0.02
    monkeypatch.setenv("VTS_RISK_MAX_GROSS", "abc")
    with pytest.raises(ValueError, match="VTS_RISK_MAX_GROSS"):
        RiskLimits.from_env()


# --- gate: weak decisions forced to Hold --------------------------------------
def test_low_confidence_forced_hold():
    g = gate_decision(_agg(rating=Rating.BUY, conf=0.1), RiskLimits(min_confidence=0.3))
    assert g.effective == Rating.HOLD and g.forced_hold
    assert any("low_confidence" in r for r in g.reasons)


def test_low_agreement_and_high_dispersion_forced_hold():
    limits = RiskLimits(min_agreement=0.6, max_dispersion=1.0)
    g1 = gate_decision(_agg(agree=0.4), limits)
    g2 = gate_decision(_agg(disp=1.8), limits)
    assert g1.effective == Rating.HOLD and g2.effective == Rating.HOLD


def test_strong_decision_passes_through():
    g = gate_decision(_agg(rating=Rating.OVERWEIGHT, conf=0.8, agree=1.0, disp=0.0), RiskLimits())
    assert g.effective == Rating.OVERWEIGHT and not g.forced_hold and g.reasons == ()


def test_schema_violation_fallback_is_hold():
    g = hold_fallback("AAPL", "unparseable model output")
    assert g.effective == Rating.HOLD and g.forced_hold
    assert "schema_violation" in g.reasons[0]


# --- clamps -------------------------------------------------------------------
def test_per_symbol_and_gross_clamps():
    limits = RiskLimits(max_weight_per_symbol=0.30, max_gross_exposure=0.50)
    eng = RiskEngine(limits)
    aggs = {
        t: AggregatedDecision(
            ticker=t, as_of="x", rating=Rating.BUY, n_samples=3, agreement=1.0,
            dispersion=0.0, mean_confidence=0.9, votes={"Buy": 3}, tie_broken_to_hold=False,
        )
        for t in ("A", "B", "C")
    }
    verdict = eng.apply(aggs)
    assert all(w <= 0.30 + 1e-12 for w in verdict.weights.values())      # symbol cap
    assert sum(abs(w) for w in verdict.weights.values()) <= 0.50 + 1e-9  # gross cap


# --- halt latch + kill switch --------------------------------------------------
def test_daily_loss_halt_latches_at_threshold():
    h = HaltState(daily_loss_limit=0.03)
    assert h.observe_period_return(-0.029) is False
    assert h.observe_period_return(-0.03) is True   # exactly at the limit halts
    assert h.observe_period_return(+0.10) is True   # latched: a good day does not unlatch


def test_halt_reset_requires_operator():
    h = HaltState(daily_loss_limit=0.03)
    h.observe_period_return(-0.05)
    h.reset(operator="human@ops")
    assert h.halted is False
    assert any("reset by human@ops" in r for r in h.halt_reasons)


def test_halted_engine_goes_flat():
    eng = RiskEngine(RiskLimits(daily_loss_limit=0.03))
    eng.observe_period_return(-0.05)
    verdict = eng.apply({"A": _agg(rating=Rating.BUY)})
    assert verdict.halted is True
    assert verdict.weights == {"A": 0.0}


def test_kill_switch_forces_flat(monkeypatch):
    monkeypatch.setenv(KILL_SWITCH_ENV, "true")
    assert kill_switch_active() is True
    eng = RiskEngine(RiskLimits())
    verdict = eng.apply({"A": _agg(rating=Rating.BUY)})
    assert verdict.halted is True and verdict.weights["A"] == 0.0
    monkeypatch.delenv(KILL_SWITCH_ENV)
    assert kill_switch_active() is False


# --- order validation ----------------------------------------------------------
def _acct(cash=10_000.0, **positions):
    return AccountState(cash=cash, positions={k.upper(): v for k, v in positions.items()})


def test_order_rejects_insufficient_cash_and_position():
    buy = Order(symbol="AAPL", side="buy", qty=100, limit_price=200.0)  # $20k > $10k cash
    rej = validate_order(buy, _acct(cash=10_000), US_EQUITY)
    assert rej and any("insufficient_cash" in v for v in rej.violations)

    sell = Order(symbol="AAPL", side="sell", qty=5, limit_price=100.0)
    rej = validate_order(sell, _acct(aapl=3), US_EQUITY)
    assert rej and any("insufficient_position" in v for v in rej.violations)


def test_order_rejects_off_tick_krx():
    # KRX 100,000 KRW band tick = 100; 123,450 is on-tick, 123,449 is not.
    ok = Order(symbol="005930", side="buy", qty=1, limit_price=123_400.0)
    bad = Order(symbol="005930", side="buy", qty=1, limit_price=123_449.0)
    assert validate_order(ok, _acct(cash=1e9), KRX_KOSPI) is None
    rej = validate_order(bad, _acct(cash=1e9), KRX_KOSPI)
    assert rej and any("off_tick" in v for v in rej.violations)


def test_order_rejects_below_min_notional_and_bad_lot():
    venue = VenueRules(name="x", min_notional=100.0, lot_size=10)
    small = Order(symbol="A", side="buy", qty=10, limit_price=5.0)  # $50 < $100
    rej = validate_order(small, _acct(cash=1e6), venue)
    assert rej and any("below_min_notional" in v for v in rej.violations)

    odd_lot = Order(symbol="A", side="buy", qty=15, limit_price=100.0)
    rej = validate_order(odd_lot, _acct(cash=1e6), venue)
    assert rej and any("lot_size" in v for v in rej.violations)


def test_valid_order_passes_and_collects_all_violations():
    ok = Order(symbol="AAPL", side="buy", qty=10, limit_price=150.00)
    assert validate_order(ok, _acct(cash=2_000), US_EQUITY) is None

    # One order violating cash + tick + notional at once reports all three.
    venue = VenueRules(name="x", min_notional=1_000_000.0)
    terrible = Order(symbol="A", side="buy", qty=1, limit_price=100.005)
    rej = validate_order(terrible, _acct(cash=1.0), venue)
    assert rej and len(rej.violations) == 3


def test_float_price_tick_check_no_fp_false_reject():
    # 0.1 + 0.2 style floats: 264.30 must be on a $0.01 tick despite fp repr.
    ok = Order(symbol="AAPL", side="buy", qty=1, limit_price=264.30)
    assert validate_order(ok, _acct(cash=1_000), US_EQUITY) is None
