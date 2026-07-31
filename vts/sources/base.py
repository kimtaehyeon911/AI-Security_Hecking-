"""The data-source protocol every vendor adapter implements."""

from __future__ import annotations

from datetime import datetime
from typing import Protocol, runtime_checkable

from vts.pit.schema import CorporateAction, FundamentalFact, NewsItem, OHLCVBar


@runtime_checkable
class DataSource(Protocol):
    """A vendor adapter that returns point-in-time records.

    Every returned record must carry a correct ``knowledge_time``. Implementations
    fetch a superset and let the :class:`~vts.pit.store.PointInTimeStore` handle
    as-of filtering; they must never pre-filter by wall clock.
    """

    name: str

    def fetch_ohlcv(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[OHLCVBar]:
        """Raw (unadjusted) daily bars for ``symbol`` over ``[start, end]``."""
        ...

    def fetch_corporate_actions(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[CorporateAction]:
        """Splits/dividends for ``symbol`` over ``[start, end]``."""
        ...

    def fetch_news(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[NewsItem]:
        """News/sentiment items published in ``[start, end]``."""
        ...

    def fetch_fundamentals(self, symbol: str) -> list[FundamentalFact]:
        """All available fundamental facts for ``symbol`` (as-of filtering done downstream)."""
        ...
