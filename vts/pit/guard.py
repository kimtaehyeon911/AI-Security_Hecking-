"""The look-ahead guard — the enforced choke point of the data-access layer.

Step 1 mandate: *"미래 데이터 접근 시 예외를 던지는 가드(assert_no_lookahead)를
데이터 접근 계층에 강제로 넣어라."* Any record that leaves the PIT store is run
through :func:`assert_no_lookahead`; a record whose ``knowledge_time`` exceeds the
requested as-of clock raises :class:`LookaheadError` instead of being returned.

This is deliberately redundant with the store's SQL ``WHERE knowledge_time <= T``
filter: the query prevents leaks, and the guard proves the query did its job.
Belt and suspenders, because a single leaked future row silently inflates every
backtest metric.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime, timezone
from typing import TypeVar

from vts.pit.clock import AsOfClock
from vts.pit.schema import KnowledgeTimedRecord

R = TypeVar("R", bound=KnowledgeTimedRecord)


class LookaheadError(RuntimeError):
    """Raised when code attempts to read data not yet knowable at the as-of clock."""

    def __init__(self, *, knowledge_time: datetime, as_of: datetime, context: str) -> None:
        self.knowledge_time = knowledge_time
        self.as_of = as_of
        self.context = context
        delta = knowledge_time - as_of
        super().__init__(
            f"look-ahead detected in {context!r}: record knowledge_time "
            f"{knowledge_time.isoformat()} is {delta} after the as-of clock "
            f"{as_of.isoformat()} — future data must never be visible"
        )


def _as_utc(dt: datetime) -> datetime:
    return dt.astimezone(timezone.utc) if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def assert_no_lookahead(
    knowledge_time: datetime,
    clock: AsOfClock,
    *,
    context: str = "data access",
) -> None:
    """Raise :class:`LookaheadError` if ``knowledge_time`` is after ``clock.as_of``."""
    kt = _as_utc(knowledge_time)
    if kt > clock.as_of:
        raise LookaheadError(knowledge_time=kt, as_of=clock.as_of, context=context)


def filter_visible(
    records: Iterable[R],
    clock: AsOfClock,
    *,
    context: str = "data access",
    strict: bool = True,
) -> list[R]:
    """Return records visible at ``clock``.

    With ``strict=True`` (the default and the mandated behavior for the data
    layer) a record from the future is a *bug in the caller* — it raises
    :class:`LookaheadError`. With ``strict=False`` future records are dropped
    silently; use that only for opportunistic live-feed ingestion, never on a
    backtest read path.
    """
    out: list[R] = []
    for rec in records:
        if clock.is_visible(rec.knowledge_time):
            out.append(rec)
        elif strict:
            raise LookaheadError(
                knowledge_time=_as_utc(rec.knowledge_time),
                as_of=clock.as_of,
                context=context,
            )
    return out
