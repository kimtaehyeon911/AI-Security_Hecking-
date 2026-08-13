"""A deterministic in-memory data source for tests and offline development.

Holds pre-built records and returns the subset in the requested window. It never
pre-filters by knowledge_time — that is the store's job — so it is also useful
for exercising look-ahead guards (feed it "future" records and assert they are
correctly hidden).
"""

from __future__ import annotations

from datetime import datetime

from vts.pit.schema import CorporateAction, FundamentalFact, NewsItem, OHLCVBar


class InMemorySource:
    """A :class:`~vts.sources.base.DataSource` backed by in-memory lists."""

    name = "fake"

    def __init__(
        self,
        *,
        ohlcv: list[OHLCVBar] | None = None,
        corporate_actions: list[CorporateAction] | None = None,
        news: list[NewsItem] | None = None,
        fundamentals: list[FundamentalFact] | None = None,
    ) -> None:
        self._ohlcv = list(ohlcv or [])
        self._actions = list(corporate_actions or [])
        self._news = list(news or [])
        self._fundamentals = list(fundamentals or [])

    @staticmethod
    def _in_window(t: datetime, start: datetime, end: datetime) -> bool:
        return start <= t <= end

    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime) -> list[OHLCVBar]:
        sym = symbol.strip().upper()
        return [
            b for b in self._ohlcv if b.symbol == sym and self._in_window(b.event_time, start, end)
        ]

    def fetch_corporate_actions(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[CorporateAction]:
        sym = symbol.strip().upper()
        return [
            a for a in self._actions if a.symbol == sym and self._in_window(a.event_time, start, end)
        ]

    def fetch_news(self, symbol: str, start: datetime, end: datetime) -> list[NewsItem]:
        sym = symbol.strip().upper()
        return [
            n for n in self._news if n.symbol == sym and self._in_window(n.knowledge_time, start, end)
        ]

    def fetch_fundamentals(self, symbol: str) -> list[FundamentalFact]:
        sym = symbol.strip().upper()
        return [f for f in self._fundamentals if f.symbol == sym]
