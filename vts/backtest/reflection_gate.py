"""Neutralize TradingAgents' deferred-reflection look-ahead.

Step 0 found the dominant leak: ``propagate()`` calls ``_resolve_pending_entries``
which, for every prior decision on the ticker, fetches realized returns over
``[entry.date, entry.date + holding_days]`` via **live** yfinance and folds the
resulting lesson into the *current* decision as ``past_context``. Replaying dates
chronologically, resolving a decision made at ``D1`` while "now" is ``D2`` reads
returns realized *after* D2 — a future leak.

The gate is a pure predicate: a pending entry may be resolved at ``trade_date``
only once its full holding window has elapsed
(``entry_date + holding_days <= trade_date``). Everything else is deferred. The
realized-return fetch itself must additionally run against the point-in-time
store capped at the simulated clock (see ``vts/integration/ta_decision.py``), so
even a resolvable entry never reads a price newer than its own resolution date.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from datetime import date, datetime, timedelta


def _as_date(value: date | datetime | str) -> date:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return datetime.strptime(value[:10], "%Y-%m-%d").date()


@dataclass(frozen=True, slots=True)
class PendingEntry:
    """A prior decision awaiting outcome resolution."""

    ticker: str
    entry_date: date

    @property
    def key(self) -> tuple[str, date]:
        return (self.ticker, self.entry_date)


@dataclass(frozen=True, slots=True)
class ReflectionSplit:
    """Entries safe to resolve now, and entries that must wait."""

    resolvable: tuple[PendingEntry, ...]
    deferred: tuple[PendingEntry, ...]


def partition_pending(
    pending: Iterable[PendingEntry | dict],
    trade_date: date | datetime | str,
    holding_days: int,
) -> ReflectionSplit:
    """Split pending entries into resolvable-now vs deferred at ``trade_date``.

    An entry is resolvable iff its holding window has fully closed on or before the
    simulated ``trade_date``. Resolving it therefore reads only prices that a real
    observer at ``trade_date`` already has.
    """
    if holding_days < 0:
        raise ValueError("holding_days must be non-negative")
    td = _as_date(trade_date)

    resolvable: list[PendingEntry] = []
    deferred: list[PendingEntry] = []
    for raw in pending:
        entry = (
            raw
            if isinstance(raw, PendingEntry)
            else PendingEntry(ticker=raw["ticker"], entry_date=_as_date(raw["date"]))
        )
        outcome_known = entry.entry_date + timedelta(days=holding_days)
        if outcome_known <= td:
            resolvable.append(entry)
        else:
            deferred.append(entry)
    return ReflectionSplit(resolvable=tuple(resolvable), deferred=tuple(deferred))
