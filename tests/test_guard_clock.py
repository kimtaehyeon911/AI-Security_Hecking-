"""The look-ahead guard and the as-of clock."""

from __future__ import annotations

import pytest
from conftest import utc

from vts.pit.clock import AsOfClock
from vts.pit.guard import LookaheadError, assert_no_lookahead, filter_visible
from vts.pit.schema import NewsItem


def _news(kt) -> NewsItem:
    return NewsItem(symbol="AAPL", event_time=kt, knowledge_time=kt, source="t", title="x")


def test_assert_no_lookahead_passes_and_raises():
    clock = AsOfClock.at(utc(2024, 6, 1, 12))
    assert_no_lookahead(utc(2024, 6, 1, 11), clock)  # ok
    assert_no_lookahead(utc(2024, 6, 1, 12), clock)  # boundary ok (<=)
    with pytest.raises(LookaheadError):
        assert_no_lookahead(utc(2024, 6, 1, 12, 0, 1), clock)


def test_filter_visible_strict_raises_on_future():
    clock = AsOfClock.at(utc(2024, 6, 1))
    recs = [_news(utc(2024, 5, 30)), _news(utc(2024, 6, 2))]
    with pytest.raises(LookaheadError):
        filter_visible(recs, clock, strict=True)


def test_filter_visible_nonstrict_drops_future():
    clock = AsOfClock.at(utc(2024, 6, 1))
    recs = [_news(utc(2024, 5, 30)), _news(utc(2024, 6, 2))]
    kept = filter_visible(recs, clock, strict=False)
    assert len(kept) == 1


def test_clock_cannot_move_backwards():
    clock = AsOfClock.at(utc(2024, 6, 1))
    assert clock.advance_to(utc(2024, 6, 2)).as_of == utc(2024, 6, 2)
    with pytest.raises(ValueError):
        clock.advance_to(utc(2024, 5, 31))


def test_clock_requires_tz_aware():
    from datetime import datetime

    with pytest.raises(ValueError):
        AsOfClock.at(datetime(2024, 6, 1))
