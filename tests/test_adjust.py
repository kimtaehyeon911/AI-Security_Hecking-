"""Point-in-time split adjustment must never use a split not yet known."""

from __future__ import annotations

from conftest import utc

from vts.pit.adjust import split_adjusted_closes
from vts.pit.clock import AsOfClock
from vts.pit.schema import CorporateAction, OHLCVBar


def _bar(day: int, close: float) -> OHLCVBar:
    t = utc(2024, 1, day, 21)
    return OHLCVBar(
        symbol="AAPL", event_time=t, knowledge_time=t, source="t",
        open=close, high=close, low=close, close=close, volume=1000,
    )


def test_split_not_yet_known_leaves_prices_unadjusted():
    bars = [_bar(2, 100), _bar(3, 100)]
    # 2:1 split with ex-date/knowledge on Jan 10.
    split = CorporateAction(
        symbol="AAPL", event_time=utc(2024, 1, 10, 14), knowledge_time=utc(2024, 1, 10, 14),
        source="t", action_type="split", value=2.0,
    )

    # As-of Jan 5 the split is NOT known -> closes are raw (no restatement leak).
    before = split_adjusted_closes(bars, [split], AsOfClock.at(utc(2024, 1, 5)))
    assert [round(b.adj_close, 4) for b in before] == [100.0, 100.0]


def test_known_split_back_adjusts_prior_closes():
    bars = [_bar(2, 100), _bar(3, 100), _bar(11, 50)]
    split = CorporateAction(
        symbol="AAPL", event_time=utc(2024, 1, 10, 14), knowledge_time=utc(2024, 1, 10, 14),
        source="t", action_type="split", value=2.0,
    )
    # As-of Jan 15 the split IS known -> pre-split closes halved, post-split unchanged.
    after = split_adjusted_closes(bars, [split], AsOfClock.at(utc(2024, 1, 15)))
    adj = {b.event_time.day: round(b.adj_close, 4) for b in after}
    assert adj[2] == 50.0 and adj[3] == 50.0 and adj[11] == 50.0
