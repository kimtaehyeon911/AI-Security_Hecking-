"""Performance metrics computed from an equity curve.

All functions are pure: input is a time-ordered ``[(datetime, equity), ...]``
curve (plus per-rebalance turnover where relevant), output is a number or a
frozen :class:`PerformanceMetrics`. Conventions, stated once:

- Annualization uses actual elapsed calendar time (``days / 365.25``), not an
  assumed bar count, so weekly/irregular decision dates annualize correctly.
  Sub-year windows therefore produce large annualized figures (a +1% day is a
  ~3700% CAGR) — mathematically correct, but read CAGR on short windows with
  care. The computation is done in log space with a capped exponent so extreme
  short-window growth yields a huge finite number, never an ``OverflowError``.
- Every function first passes the curve through :func:`sanitize_curve`: the curve
  is truncated at the first non-positive equity point (a wiped-out account has no
  further returns), and the wipe-out period itself is clamped at -100%. This
  keeps ``n_periods``, annualization (periods-per-year), win rate, and volatility
  all computed over the SAME sample — no silently dropped periods.
- ``max_drawdown`` is reported as a POSITIVE fraction (0.25 == a -25% drawdown).
  It can exceed 1.0 when equity goes negative within the crash period; a curve
  with no positive peak at all reports 1.0 (wiped out), never 0.
- ``sharpe`` / ``sortino`` use POPULATION (n-denominator) deviations
  (``statistics.pstdev``), consistent with the dispersion metric in
  ``vts.decision``. Sample-stdev (n-1) Sharpe from other systems will read
  slightly lower on small n. Both return 0.0 when volatility/downside is zero —
  a flat curve is "no evidence of skill", not infinite skill.
- ``avg_win_loss_ratio`` (평균손익비) is mean(win) / |mean(loss)|; ``None`` when
  there are no losing periods — including an all-flat curve — (undefined, not
  infinite), and 0.0 only when losses occurred but no period ever won.
- ``win_rate`` counts strictly positive periods over all periods; flat (0.0)
  periods count in the denominator as non-wins.
"""

from __future__ import annotations

import math
import statistics
from datetime import datetime

from pydantic import BaseModel, ConfigDict

# exp() overflows just above 709; capping the annualization exponent here turns
# a pathological sub-hour growth spurt into a huge finite CAGR instead of a crash.
_MAX_LOG_GROWTH_PER_YEAR = 700.0


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


def sanitize_curve(curve: list[tuple[datetime, float]]) -> list[tuple[datetime, float]]:
    """Truncate the curve at the first non-positive equity point (inclusive).

    An account at or below zero has no meaningful further returns; keeping
    post-wipe-out points would silently desynchronize periods-per-year,
    ``n_periods`` and the ratio denominators from the actual return sample.
    """
    out: list[tuple[datetime, float]] = []
    for point in curve:
        out.append(point)
        if point[1] <= 0:
            break
    return out


def period_returns(curve: list[tuple[datetime, float]]) -> list[float]:
    """Simple per-period returns, clamped at -100% (compounded returns cannot lose
    more than everything). Operates on the sanitized curve."""
    curve = sanitize_curve(curve)
    rets: list[float] = []
    for (_, prev), (_, cur) in zip(curve, curve[1:]):
        if prev > 0:
            rets.append(max(cur / prev - 1.0, -1.0))
    return rets


def elapsed_years(curve: list[tuple[datetime, float]]) -> float:
    curve = sanitize_curve(curve)
    if len(curve) < 2:
        return 0.0
    return (curve[-1][0] - curve[0][0]).total_seconds() / (365.25 * 24 * 3600)


def cagr(curve: list[tuple[datetime, float]]) -> float:
    curve = sanitize_curve(curve)
    years = elapsed_years(curve)
    if years <= 0 or len(curve) < 2 or curve[0][1] <= 0:
        return 0.0
    growth = curve[-1][1] / curve[0][1]
    if growth <= 0:
        return -1.0  # wiped out
    exponent = math.log(growth) / years
    return math.exp(min(exponent, _MAX_LOG_GROWTH_PER_YEAR)) - 1.0


def max_drawdown(curve: list[tuple[datetime, float]]) -> float:
    """Deepest peak-to-trough loss as a positive fraction (may exceed 1.0 when
    the crash period drives equity negative). 1.0 for a curve with no positive
    peak (there is nothing but loss to report), 0.0 for an empty curve."""
    curve = sanitize_curve(curve)
    peak = -math.inf
    worst = 0.0
    saw_positive_peak = False
    for _, v in curve:
        peak = max(peak, v)
        if peak > 0:
            saw_positive_peak = True
            worst = min(worst, v / peak - 1.0)
    if not saw_positive_peak:
        return 1.0 if curve else 0.0
    return -worst


def _annualized_ratio(
    curve: list[tuple[datetime, float]], rf_annual: float, downside_only: bool
) -> float:
    rets = period_returns(curve)
    years = elapsed_years(curve)
    if len(rets) < 2 or years <= 0:
        return 0.0
    ppy = len(rets) / years  # periods per year from the sanitized sample itself
    rf_period = (1.0 + rf_annual) ** (1.0 / ppy) - 1.0 if ppy > 0 else 0.0
    excess = [r - rf_period for r in rets]
    if downside_only:
        denom = math.sqrt(statistics.mean([min(r, 0.0) ** 2 for r in excess]))
    else:
        denom = statistics.pstdev(excess)
    if denom == 0:
        return 0.0
    return statistics.mean(excess) / denom * math.sqrt(ppy)


def sharpe(curve: list[tuple[datetime, float]], rf_annual: float = 0.0) -> float:
    return _annualized_ratio(curve, rf_annual, downside_only=False)


def sortino(curve: list[tuple[datetime, float]], rf_annual: float = 0.0) -> float:
    return _annualized_ratio(curve, rf_annual, downside_only=True)


def win_rate(curve: list[tuple[datetime, float]]) -> float:
    rets = period_returns(curve)
    if not rets:
        return 0.0
    return sum(1 for r in rets if r > 0) / len(rets)


def avg_win_loss_ratio(curve: list[tuple[datetime, float]]) -> float | None:
    rets = period_returns(curve)
    wins = [r for r in rets if r > 0]
    losses = [r for r in rets if r < 0]
    if not losses:
        return None  # undefined: nothing was ever lost (covers the all-flat curve)
    if not wins:
        return 0.0  # losses occurred, never a winning period
    return statistics.mean(wins) / abs(statistics.mean(losses))


def compute_metrics(
    curve: list[tuple[datetime, float]],
    *,
    turnover_per_period: list[float] | None = None,
    rf_annual: float = 0.0,
) -> PerformanceMetrics:
    """Compute the full mandated metric set for one equity curve.

    ``turnover_per_period`` is summed as given (it is trade bookkeeping, not a
    curve property) and annualized over the sanitized curve's span.
    """
    clean = sanitize_curve(curve)
    years = elapsed_years(clean)
    total = clean[-1][1] / clean[0][1] - 1.0 if len(clean) >= 2 and clean[0][1] > 0 else 0.0
    turn = sum(turnover_per_period or [])
    return PerformanceMetrics(
        total_return=max(total, -1.0),
        cagr=cagr(clean),
        sharpe=sharpe(clean, rf_annual),
        sortino=sortino(clean, rf_annual),
        max_drawdown=max_drawdown(clean),
        win_rate=win_rate(clean),
        avg_win_loss_ratio=avg_win_loss_ratio(clean),
        annual_turnover=turn / years if years > 0 else 0.0,
        n_periods=max(0, len(clean) - 1),
        years=years,
    )
