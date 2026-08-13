"""Metric correctness against hand-computed values."""

from __future__ import annotations

import pytest
from conftest import utc

from vts.backtest.metrics import (
    avg_win_loss_ratio,
    cagr,
    compute_metrics,
    max_drawdown,
    sharpe,
    sortino,
    win_rate,
)


def _curve(*points):
    """points = (day, value) in Jan-Mar 2024."""
    out = []
    for day, value in points:
        month = 1 + (day - 1) // 28
        out.append((utc(2024, month, ((day - 1) % 28) + 1, 21), float(value)))
    return out


def test_max_drawdown_hand_computed():
    curve = _curve((1, 100), (2, 120), (3, 90), (4, 130))
    # peak 120 -> trough 90 = -25%
    assert max_drawdown(curve) == pytest.approx(0.25)


def test_win_rate_and_win_loss_ratio():
    curve = _curve((1, 100), (2, 120), (3, 90), (4, 130))
    # returns: +0.20, -0.25, +0.4444 -> 2/3 wins
    assert win_rate(curve) == pytest.approx(2 / 3)
    ratio = avg_win_loss_ratio(curve)
    expected = ((0.20 + 40 / 90) / 2) / 0.25
    assert ratio == pytest.approx(expected)


def test_avg_win_loss_none_when_no_losses():
    assert avg_win_loss_ratio(_curve((1, 100), (2, 110), (3, 121))) is None


def test_cagr_one_year_window():
    curve = [(utc(2023, 1, 1, 21), 100.0), (utc(2024, 1, 1, 21), 121.0)]
    # 365 elapsed days ~= 1 year -> CAGR ~ 21%
    assert cagr(curve) == pytest.approx(0.21, abs=0.005)


def test_sharpe_sign_and_flat_zero():
    rising = _curve((1, 100), (2, 101), (3, 103), (4, 104))
    falling = _curve((1, 100), (2, 99), (3, 97), (4, 96))
    flat = _curve((1, 100), (2, 100), (3, 100), (4, 100))
    assert sharpe(rising) > 0
    assert sharpe(falling) < 0
    assert sharpe(flat) == 0.0


def test_sortino_zero_downside_convention():
    # All-win curve: downside deviation 0 -> documented convention returns 0.0.
    assert sortino(_curve((1, 100), (2, 110), (3, 121))) == 0.0


def test_compute_metrics_annual_turnover():
    curve = [(utc(2023, 1, 1, 21), 100.0), (utc(2024, 1, 1, 21), 110.0)]
    m = compute_metrics(curve, turnover_per_period=[0.5, 0.3])
    # ~1 year elapsed -> annual turnover ~ 0.8
    assert m.annual_turnover == pytest.approx(0.8, abs=0.01)
    assert m.total_return == pytest.approx(0.10)
    assert m.n_periods == 1


def test_short_or_degenerate_curves_do_not_crash():
    single = [(utc(2024, 1, 1, 21), 100.0)]
    m = compute_metrics(single)
    assert m.total_return == 0.0
    assert m.cagr == 0.0
    assert m.sharpe == 0.0
    assert m.max_drawdown == 0.0
