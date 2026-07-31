"""Schema invariants: these are look-ahead correctness guarantees, not niceties."""

from __future__ import annotations

from datetime import datetime

import pytest
from conftest import utc
from pydantic import ValidationError

from vts.pit.schema import FundamentalFact, NewsItem, OHLCVBar


def test_naive_datetime_rejected():
    with pytest.raises(ValidationError):
        OHLCVBar(
            symbol="AAPL",
            event_time=datetime(2024, 1, 2, 21, 0, 0),  # naive!
            knowledge_time=utc(2024, 1, 2, 21),
            source="test",
            open=1, high=2, low=1, close=1.5, volume=100,
        )


def test_knowledge_before_event_rejected():
    with pytest.raises(ValidationError):
        NewsItem(
            symbol="AAPL",
            event_time=utc(2024, 1, 3),
            knowledge_time=utc(2024, 1, 2),  # known before it happened
            source="test",
            title="impossible",
        )


def test_ohlc_consistency_enforced():
    with pytest.raises(ValidationError):
        OHLCVBar(
            symbol="AAPL", event_time=utc(2024, 1, 2, 21), knowledge_time=utc(2024, 1, 2, 21),
            source="test", open=10, high=5, low=1, close=3, volume=100,  # high < open
        )


def test_symbol_normalized_and_frozen():
    bar = OHLCVBar(
        symbol="  aapl ", event_time=utc(2024, 1, 2, 21), knowledge_time=utc(2024, 1, 2, 21),
        source="test", open=1, high=2, low=1, close=1.5, volume=100,
    )
    assert bar.symbol == "AAPL"
    with pytest.raises(ValidationError):
        bar.close = 2.0  # frozen


def test_timestamps_normalized_to_utc():
    from datetime import timezone, timedelta

    kst = timezone(timedelta(hours=9))
    fact = FundamentalFact(
        symbol="AAPL",
        event_time=datetime(2024, 3, 31, tzinfo=kst),
        knowledge_time=datetime(2024, 5, 1, 9, 0, 0, tzinfo=kst),
        source="test", metric="eps", value=1.5,
    )
    assert fact.event_time.tzinfo is not None
    assert fact.knowledge_time.utcoffset().total_seconds() == 0
