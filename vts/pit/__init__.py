"""Point-in-time (PIT) data layer.

Every datum carries two timestamps:

- ``event_time``      — the instant the datum *describes* (a bar's session close,
                        a fiscal period end, an article's subject moment).
- ``knowledge_time``  — the instant the datum first *became knowable* to a
                        real-time observer (an article's publish time, a
                        filing's report/release date, a bar's close time).

A backtest reads the world "as of" a simulated clock ``T`` and may only ever see
records whose ``knowledge_time <= T``. That single rule kills the two classic
leaks found in Step 0:

1. **Restatement leak** — auto-adjusted prices bake in future splits/dividends.
   We store *raw* OHLCV plus separately-timed :class:`CorporateAction` records,
   so adjustment can only use actions already known at ``T``.
2. **Future-news leak** — news keyed by publish time is invisible until published.
"""

from __future__ import annotations

from vts.pit.clock import AsOfClock
from vts.pit.guard import LookaheadError, assert_no_lookahead, filter_visible
from vts.pit.schema import (
    CorporateAction,
    FundamentalFact,
    KnowledgeTimedRecord,
    NewsItem,
    OHLCVBar,
    RecordKind,
)
from vts.pit.store import PointInTimeStore

__all__ = [
    "AsOfClock",
    "LookaheadError",
    "assert_no_lookahead",
    "filter_visible",
    "CorporateAction",
    "FundamentalFact",
    "KnowledgeTimedRecord",
    "NewsItem",
    "OHLCVBar",
    "RecordKind",
    "PointInTimeStore",
]
