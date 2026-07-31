"""As-of querying and restatement collapse in the point-in-time store."""

from __future__ import annotations

from conftest import utc

from vts.pit.clock import AsOfClock
from vts.pit.schema import FundamentalFact, NewsItem, OHLCVBar
from vts.pit.store import PointInTimeStore


def _bar(day: int, close: float) -> OHLCVBar:
    t = utc(2024, 1, day, 21)
    return OHLCVBar(
        symbol="AAPL", event_time=t, knowledge_time=t, source="t",
        open=close, high=close, low=close, close=close, volume=1000,
    )


def test_asof_hides_future_bars():
    store = PointInTimeStore()
    store.append_many([_bar(2, 100), _bar(3, 101), _bar(4, 102)])

    clock = AsOfClock.at(utc(2024, 1, 3, 21))
    bars = store.get_ohlcv("AAPL", clock)
    assert [b.event_time.day for b in bars] == [2, 3]  # day 4 not yet knowable


def test_restatement_collapse_returns_latest_known():
    """Same fiscal period reported, then restated — as-of view flips at the restatement."""
    store = PointInTimeStore()
    fiscal = utc(2024, 3, 31)
    first = FundamentalFact(
        symbol="AAPL", event_time=fiscal, knowledge_time=utc(2024, 4, 25, 21),
        source="t", metric="netIncome", value=100.0, revision=0,
    )
    restated = FundamentalFact(
        symbol="AAPL", event_time=fiscal, knowledge_time=utc(2024, 7, 10, 21),
        source="t", metric="netIncome", value=95.0, revision=1,
    )
    store.append_many([first, restated])

    # Before the first report: nothing known.
    assert store.get_fundamentals("AAPL", AsOfClock.at(utc(2024, 4, 1)), metric="netIncome") == []

    # Between first report and restatement: original value.
    mid = store.get_fundamentals("AAPL", AsOfClock.at(utc(2024, 5, 1)), metric="netIncome")
    assert len(mid) == 1 and mid[0].value == 100.0

    # After restatement: latest-known value only (collapsed, not both).
    after = store.get_fundamentals("AAPL", AsOfClock.at(utc(2024, 8, 1)), metric="netIncome")
    assert len(after) == 1 and after[0].value == 95.0


def test_news_not_collapsed():
    store = PointInTimeStore()
    store.append_many([
        NewsItem(symbol="AAPL", event_time=utc(2024, 1, 2, 9), knowledge_time=utc(2024, 1, 2, 9),
                 source="t", title="a", url="u1"),
        NewsItem(symbol="AAPL", event_time=utc(2024, 1, 2, 15), knowledge_time=utc(2024, 1, 2, 15),
                 source="t", title="b", url="u2"),
    ])
    got = store.get_news("AAPL", AsOfClock.at(utc(2024, 1, 3)))
    assert {n.title for n in got} == {"a", "b"}


def test_persistence_roundtrip(tmp_path):
    path = tmp_path / "pit.sqlite"
    store = PointInTimeStore(path)
    store.append(_bar(2, 100))
    store.close()

    reopened = PointInTimeStore(path)
    bars = reopened.get_ohlcv("AAPL", AsOfClock.at(utc(2024, 1, 3)))
    assert len(bars) == 1 and bars[0].close == 100
