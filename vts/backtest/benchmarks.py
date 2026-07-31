"""Benchmark curves computed from the point-in-time store.

Two mandated benchmarks, both share-based (weights drift; no hidden free
rebalancing — the Step 2 review lesson) and both charged a one-time entry cost so
the strategy-vs-benchmark comparison is after-cost on both sides:

- **buy&hold**: equal-weight basket of the universe bought at the first priced
  date and held.
- **60/40**: 60% the same equity basket + 40% cash accruing ``rf_annual``.
  Classic 60/40 holds bonds; with an equity-only PIT store we proxy the 40% as
  cash at a configurable risk-free rate (default 0 — we do not fabricate a bond
  return). Stated here once so nobody mistakes it for a bond sleeve; the rendered
  evaluation report also discloses it.

Conventions (aligned with the strategy engine so the gate is symmetric):

- Prices are read as-of each date's clock (no look-ahead), and only strictly
  positive closes count as "priced" — a schema-valid zero close never divides.
- The ENTRY point of a curve is pre-cost capital; the entry cost is subtracted
  from every LATER point. Folding the cost into the first point would cancel it
  out of last/first total_return and make the benchmark effectively pre-cost.
- Curves start AT the entry date. Padding earlier dates with flat capital would
  dilute the benchmark's win rate and Sharpe with fabricated 0% periods.
- Basket membership is frozen at entry (that is what buy&hold means). When some
  universe tickers list later, the strategy can trade names the benchmark never
  held — :func:`entry_basket` exposes membership so the evaluation report can
  disclose the mismatch instead of hiding it.
"""

from __future__ import annotations

from datetime import datetime

from vts.backtest.costs import CostModel
from vts.backtest.engine import _dollar_adv, _last_close
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore


def _priced_at(
    store: PointInTimeStore, universe: list[str], clock: AsOfClock
) -> dict[str, float]:
    """Tickers with a strictly positive last close as-of ``clock``."""
    out: dict[str, float] = {}
    for t in universe:
        p = _last_close(store, t, clock)
        if p is not None and p > 0:
            out[t] = p
    return out


def entry_basket(
    store: PointInTimeStore, universe: list[str], dates: list[datetime]
) -> tuple[list[str], datetime | None]:
    """The benchmark's frozen membership: tickers priced at the first priced date.

    Returns ``(sorted tickers, entry_date)`` — or ``([], None)`` when nothing in
    the universe is ever priced. Used by the evaluation report to disclose any
    gap between the benchmark basket and the strategy's evolving universe.
    """
    for d in sorted(dates):
        priced = _priced_at(store, universe, AsOfClock.at(d))
        if priced:
            return sorted(priced), d
    return [], None


def _entry_and_shares(
    store: PointInTimeStore,
    universe: list[str],
    dates: list[datetime],
    capital: float,
    cost_model: CostModel,
    adv_lookback: int,
) -> tuple[dict[str, float], float, int] | None:
    """Buy the equal-weight basket at the first date where any ticker is priced.

    Returns (shares, entry_cost, entry_index) or None if nothing is ever priced.
    """
    for i, d in enumerate(sorted(dates)):
        clock = AsOfClock.at(d)
        priced = _priced_at(store, universe, clock)
        if not priced:
            continue
        alloc = capital / len(priced)
        shares = {t: alloc / p for t, p in priced.items()}
        entry_cost = sum(
            cost_model.cost(alloc, _dollar_adv(store, t, clock, adv_lookback), "buy").total
            for t in priced
        )
        return shares, entry_cost, i
    return None


def _basket_value(store: PointInTimeStore, shares: dict[str, float], clock: AsOfClock) -> float:
    value = 0.0
    for t, sh in shares.items():
        p = _last_close(store, t, clock)
        if p is not None:
            value += sh * p
    return value


def buy_and_hold_curve(
    store: PointInTimeStore,
    universe: list[str],
    dates: list[datetime],
    *,
    capital: float = 100_000.0,
    cost_model: CostModel | None = None,
    adv_lookback: int = 20,
) -> list[tuple[datetime, float]]:
    """Equal-weight buy&hold: buy once, entry cost drags every later point."""
    cost_model = cost_model or CostModel()
    ordered = sorted(dates)
    entered = _entry_and_shares(store, universe, ordered, capital, cost_model, adv_lookback)
    if entered is None:
        return [(d, capital) for d in ordered]
    shares, entry_cost, entry_idx = entered
    curve: list[tuple[datetime, float]] = []
    for i, d in enumerate(ordered):
        if i < entry_idx:
            continue  # curve starts at entry; no fabricated flat periods
        if i == entry_idx:
            curve.append((d, capital))  # pre-cost entry point (strategy convention)
        else:
            curve.append((d, _basket_value(store, shares, AsOfClock.at(d)) - entry_cost))
    return curve


def sixty_forty_curve(
    store: PointInTimeStore,
    universe: list[str],
    dates: list[datetime],
    *,
    capital: float = 100_000.0,
    equity_fraction: float = 0.6,
    rf_annual: float = 0.0,
    cost_model: CostModel | None = None,
    adv_lookback: int = 20,
) -> list[tuple[datetime, float]]:
    """60/40: 60% equity basket (buy&hold, entry-costed) + 40% cash at ``rf_annual``.

    The cash sleeve compounds continuously between dates by actual elapsed days.
    No inter-sleeve rebalancing after entry (a drifting, honest 60/40 — matching
    the buy&hold convention; a periodically rebalanced variant can be added later
    with its own turnover costs).
    """
    cost_model = cost_model or CostModel()
    ordered = sorted(dates)
    eq_capital = capital * equity_fraction
    cash = capital * (1.0 - equity_fraction)
    entered = _entry_and_shares(store, universe, ordered, eq_capital, cost_model, adv_lookback)
    if entered is None:
        return [(d, capital) for d in ordered]
    shares, entry_cost, entry_idx = entered
    curve: list[tuple[datetime, float]] = []
    prev_date: datetime | None = None
    for i, d in enumerate(ordered):
        if i < entry_idx:
            continue
        if prev_date is not None and rf_annual != 0.0:
            dt_years = (d - prev_date).total_seconds() / (365.25 * 24 * 3600)
            cash *= (1.0 + rf_annual) ** dt_years
        if i == entry_idx:
            curve.append((d, capital))  # pre-cost entry point
        else:
            equity_value = _basket_value(store, shares, AsOfClock.at(d)) - entry_cost
            curve.append((d, equity_value + cash))
        prev_date = d
    return curve
