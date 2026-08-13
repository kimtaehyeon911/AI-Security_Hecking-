"""Point-in-time realized returns — the look-ahead-free replacement for
TradingAgents' ``_fetch_returns`` (which used live yfinance with no as-of cap).

Used both by the reflection gate (to score a matured past decision) and available
to the engine. Every price read is capped at an as-of clock, so a return can only
be computed from bars a real observer at that clock already holds.
"""

from __future__ import annotations

from datetime import date, datetime

from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore


def realized_return(
    store: PointInTimeStore,
    ticker: str,
    entry_date: date | datetime,
    holding_days: int,
    ceiling: AsOfClock,
) -> float | None:
    """Close-to-close return over ``holding_days`` **trading bars** from entry.

    Trading-bar (not calendar-day) counting: the exit is the bar ``holding_days``
    positions after the entry bar, so holidays/weekends don't distort the horizon.
    All bars are read as-of ``ceiling`` (the simulated 'now'), so the exit can never
    be a bar newer than the caller's knowledge horizon. Returns ``None`` if the
    entry bar is unknown or the holding window has not fully elapsed within the
    bars visible at ``ceiling``.
    """
    if holding_days < 0:
        raise ValueError("holding_days must be non-negative")

    visible = store.get_ohlcv(ticker, ceiling)  # already capped at the ceiling
    if not visible:
        return None

    ed = entry_date.date() if isinstance(entry_date, datetime) else entry_date
    entry_idx: int | None = None
    for idx, b in enumerate(visible):
        if b.event_time.date() <= ed:
            entry_idx = idx
        else:
            break
    if entry_idx is None:
        return None

    exit_idx = entry_idx + holding_days
    if exit_idx >= len(visible):
        return None  # window not yet elapsed within data knowable at the ceiling

    entry_price = visible[entry_idx].close
    exit_price = visible[exit_idx].close
    if entry_price <= 0:
        return None
    return exit_price / entry_price - 1.0
