"""Performance metrics computed from an equity curve.

All functions are pure: input is a time-ordered ``[(datetime, equity), ...]``
curve (plus per-rebalance turnover where relevant), output is a number or a
frozen :class:`PerformanceMetrics`. Conventions, stated once:

- Annualization uses actual elapsed calendar time (``days / 365.25``), not an
  assumed bar count, so weekly/irregular decision dates annualize correctly.
- ``max_drawdown`` is reported as a POSITIVE fraction (0.25 == a -25% drawdown).
- ``sharpe`` / ``sortino`` return 0.0 when volatility/downside is zero — a flat
  curve is "no evidence of skill", not infinite skill.
- ``avg_win_loss_ratio`` (평균손익비) is mean(win) / |mean(loss)|; ``None`` when
  there are no losing periods (undefined, not infinite).
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime

from pydantic import BaseModel, ConfigDict


class PerformanceMetrics(BaseModel):
    """The Step 3 mandated metric set for one curve."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_return: float
    cagr: float
    sharpe: float
    sortino: float
    max_drawdown: float
    win_rate: float
    avg_win_loss_ratio: float | None
    annual_turnover: float
    n_periods: int
    years: float


def period_returns(curve: list[tuple[datetime, float]]) -> list[float]:
    """Simple per-period returns between consecutive curve points."""
    rets: list[float] = []
    for (_, prev), (_, cur) in zip(curve, curve[1:]):
        if prev > 0:
            rets.append(cur / prev - 1.0)
    return rets


def elapsed_years(curve: list[tuple[datetime, float]]) -> float:
    if len(curve) < 2:
        return 0.0
    return (curve[-1][0] - curve[0][0]).total_seconds() / (365.25 * 24 * 3600)


def cagr(curve: list[tuple[datetime, float]]) -> float:
    years = elapsed_years(curve)
    if years <= 0 or curve[0][1] <= 0:
        return 0.0
    growth = curve[-1][1] / curve[0][1]
    if growth <= 0:
        return -1.0
    return growth ** (1.0 / years) - 1.0


def max_drawdown(curve: list[tuple[datetime, float]]) -> float:
    """Deepest peak-to-trough loss, as a positive fraction."""
    peak = -math.inf
    worst = 0.0
    for _, v in curve:
        peak = max(peak, v)
        if peak > 0:
            worst = min(worst, v / peak - 1.0)
    return -worst


def sharpe(curve: list[tuple[datetime, float]], rf_annual: float = 0.0) -> float:
    rets = period_returns(curve)
    years = elapsed_years(curve)
    if len(rets) < 2 or years <= 0:
        return 0.0
    ppy = len(rets) / years  # periods per year from actual spacing
    rf_period = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0 if ppy > 0 else 0.0
    excess = [r - rf_period for r in rets]
    vol = statistics.pstdev(excess)
    if vol == 0:
        return 0.0
    return statistics.mean(excess) / vol * math.sqrt(ppy)


def sortino(curve: list[tuple[datetime, float]], rf_annual: float = 0.0) -> float:
    rets = period_returns(curve)
    years = elapsed_years(curve)
    if len(rets) < 2 or years <= 0:
        return 0.0
    ppy = len(rets) / years
    rf_period = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0 if ppy > 0 else 0.0
    excess = [r - rf_period for r in rets]
    downside = math.sqrt(statistics.mean([min(r, 0.0) ** 2 for r in excess]))
    if downside == 0:
        return 0.0
    return statistics.mean(excess) / downside * math.sqrt(ppy)


def win_rate(curve: list[tuple[datetime, float]]) -> float:
    rets = period_returns(curve)
    if not rets:
        return 0.0
    return sum(1 for r in rets if r > 0) / len(rets)


def avg_win_loss_ratio(curve: list[tuple[datetime, float]]) -> float | None:
    rets = period_returns(curve)
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    if not wins:
        return 0.0
    if not losses:
        return None  # undefined: nothing was ever lost
    return statistics.mean(wins) / abs(statistics.mean(losses))


def compute_metrics(
    curve: list[tuple[datetime, float]],
    *,
    turnover_per_period: list[float] | None = None,
    rf_annual: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full mandated metric set for one equity curve."""
    years = elapsed_years(curve)
    total = curve[-1][1] / curve[0][1] - 1.0 if len(curve) >= 2 and curve[0][1] > 0 else 0.0
    turn = sum(turnover_per_period or [])
    return PerformanceMetrics(
        total_return=total,
        cagr=cagr(curve),
        sharpe=sharpe(curve, rf_annual),
        sortino=sortino(curve, rf_annual),
        max_drawdown=max_drawdown(curve),
        win_rate=win_rate(curve),
        avg_win_loss_ratio=avg_win_loss_ratio(curve),
        annual_turnover=turn / years if years > 0 else 0.0,
        n_periods=max(0, len(curve) - 1),
        years=years,
    )
