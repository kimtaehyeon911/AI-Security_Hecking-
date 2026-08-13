"""Ingestion: pull point-in-time records from a source into the store.

Ingestion writes the vendor's records verbatim (with their ``knowledge_time``
intact) — it never filters by wall clock. As-of correctness is a *read-time*
property enforced by :class:`~vts.pit.store.PointInTimeStore`, so ingesting more
data (even "future" data) can never leak into a past as-of view.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vts.pit.store import PointInTimeStore
from vts.sources.base import DataSource


@dataclass(frozen=True, slots=True)
class IngestReport:
    """Counts written per record kind."""

    ohlcv: int = 0
    corporate_actions: int = 0
    news: int = 0
    fundamentals: int = 0

    @property
    def total(self) -> int:
        return self.ohlcv + self.corporate_actions + self.news + self.fundamentals


def ingest_symbol(
    store: PointInTimeStore,
    source: DataSource,
    symbol: str,
    start: datetime,
    end: datetime,
) -> IngestReport:
    """Fetch all record kinds for ``symbol`` over ``[start, end]`` and store them."""
    ohlcv = source.fetch_ohlcv(symbol, start, end)
    actions = source.fetch_corporate_actions(symbol, start, end)
    news = source.fetch_news(symbol, start, end)
    fundamentals = source.fetch_fundamentals(symbol)

    return IngestReport(
        ohlcv=store.append_many(ohlcv),
        corporate_actions=store.append_many(actions),
        news=store.append_many(news),
        fundamentals=store.append_many(fundamentals),
    )
