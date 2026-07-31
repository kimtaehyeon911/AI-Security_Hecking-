"""End-to-end: ingest from a DataSource into the store, then read as-of."""

from __future__ import annotations

from conftest import utc

from vts.config import Settings
from vts.ingest import ingest_symbol
from vts.pit.clock import AsOfClock
from vts.pit.schema import FundamentalFact, NewsItem, OHLCVBar
from vts.pit.store import PointInTimeStore
from vts.sources.fake import InMemorySource


def _bar(day: int, close: float) -> OHLCVBar:
    t = utc(2024, 1, day, 21)
    return OHLCVBar(
        symbol="AAPL", event_time=t, knowledge_time=t, source="fake",
        open=close, high=close, low=close, close=close, volume=1000,
    )


def test_ingest_then_asof_read():
    source = InMemorySource(
        ohlcv=[_bar(2, 100), _bar(3, 101), _bar(4, 102)],
        news=[
            NewsItem(symbol="AAPL", event_time=utc(2024, 1, 3, 9), knowledge_time=utc(2024, 1, 3, 9),
                     source="fake", title="n", url="u"),
        ],
        fundamentals=[
            FundamentalFact(symbol="AAPL", event_time=utc(2023, 12, 31),
                            knowledge_time=utc(2024, 1, 25, 21), source="fake",
                            metric="eps", value=2.0),
        ],
    )
    store = PointInTimeStore()
    report = ingest_symbol(store, source, "AAPL", utc(2024, 1, 1), utc(2024, 1, 31))
    assert report.ohlcv == 3 and report.news == 1 and report.fundamentals == 1
    assert report.total == 5

    # As-of Jan 3: two bars + one news; fundamentals not yet reported.
    clock = AsOfClock.at(utc(2024, 1, 3, 21))
    assert len(store.get_ohlcv("AAPL", clock)) == 2
    assert len(store.get_news("AAPL", clock)) == 1
    assert store.get_fundamentals("AAPL", clock) == []


def test_settings_from_env_defaults_and_override(monkeypatch):
    s = Settings.from_env()
    assert s.asset_class == "us_equity"
    assert s.deep_think_llm == "deepseek-v4-pro"
    assert s.temperature == 0.0
    assert s.store_path.name == "pit_store.sqlite"

    monkeypatch.setenv("VTS_ASSET_CLASS", "crypto_spot")
    monkeypatch.setenv("VTS_UNIVERSE", "btc-usd, eth-usd")
    monkeypatch.setenv("VTS_TEMPERATURE", "0.2")
    s2 = Settings.from_env()
    assert s2.asset_class == "crypto_spot"
    assert s2.universe == ["BTC-USD", "ETH-USD"]
    assert s2.temperature == 0.2
