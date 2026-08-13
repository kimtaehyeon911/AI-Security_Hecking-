"""PITDataProvider: the string surface TradingAgents consumes must be leak-free."""

from __future__ import annotations

from conftest import utc

from vts.integration.tradingagents_vendor import PITDataProvider
from vts.pit.clock import AsOfClock
from vts.pit.schema import NewsItem, OHLCVBar
from vts.pit.store import PointInTimeStore


def _bar(day: int, close: float) -> OHLCVBar:
    t = utc(2024, 1, day, 21)
    return OHLCVBar(
        symbol="AAPL", event_time=t, knowledge_time=t, source="t",
        open=close, high=close, low=close, close=close, volume=1000,
    )


def test_provider_stock_data_excludes_future():
    store = PointInTimeStore()
    store.append_many([_bar(2, 100), _bar(3, 101), _bar(4, 102)])
    provider = PITDataProvider(store)
    clock = AsOfClock.at(utc(2024, 1, 3, 21))

    out = provider.get_stock_data("AAPL", "2024-01-01", "2024-01-31", clock)
    assert "2024-01-03" in out
    assert "2024-01-04" not in out  # future bar not rendered


def test_provider_news_publish_time_only():
    store = PointInTimeStore()
    store.append_many([
        NewsItem(symbol="AAPL", event_time=utc(2024, 1, 2, 9), knowledge_time=utc(2024, 1, 2, 9),
                 source="t", title="known", url="a"),
        NewsItem(symbol="AAPL", event_time=utc(2024, 1, 9, 9), knowledge_time=utc(2024, 1, 9, 9),
                 source="t", title="future scoop", url="b"),
    ])
    provider = PITDataProvider(store)
    clock = AsOfClock.at(utc(2024, 1, 3, 21))
    out = provider.get_news("AAPL", "2024-01-01", "2024-01-31", clock)
    assert "known" in out
    assert "future scoop" not in out
