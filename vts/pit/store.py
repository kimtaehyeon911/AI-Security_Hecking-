"""SQLite-backed point-in-time store.

Design choices:

- **stdlib ``sqlite3`` only** — no pandas/pyarrow. One table per
  :class:`RecordKind`; timestamps stored as integer microseconds-since-epoch
  (UTC) so ``knowledge_time <= T`` is an exact, index-friendly integer compare.
- **Append-only.** A correction is a new row with a later ``knowledge_time`` and
  higher ``revision``; nothing is ever updated in place. This is what lets an
  as-of query reconstruct exactly what was knowable at any past instant.
- **Guarded egress.** Every public read filters ``knowledge_time <= clock.as_of``
  in SQL *and* re-checks each returned record through
  :func:`vts.pit.guard.filter_visible` (``strict=True``). A leaked future row is
  a raised exception, never a silent metric inflation.
- **Restatement collapse.** Reads return, per logical series key, the revision
  with the greatest ``knowledge_time`` that is still ``<= clock.as_of`` — i.e.
  the value a real observer would have believed at ``T``.
"""

from __future__ import annotations

import sqlite3
from collections.abc import Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path

from vts.pit.clock import AsOfClock
from vts.pit.guard import filter_visible
from vts.pit.schema import (
    AnyRecord,
    CorporateAction,
    FundamentalFact,
    KnowledgeTimedRecord,
    NewsItem,
    OHLCVBar,
    RecordKind,
    model_for_kind,
)

_US = 1_000_000


def _to_us(dt: datetime) -> int:
    """Microseconds since epoch (UTC). Assumes tz-aware (schema guarantees it)."""
    return int(dt.astimezone(timezone.utc).timestamp() * _US)


def _series_key(rec: KnowledgeTimedRecord) -> tuple:
    """Logical identity used to collapse restatements to the latest-known revision.

    Two records sharing a series key describe *the same fact at the same event
    time*; only the one with the greatest visible ``knowledge_time`` survives a
    point-in-time read.
    """
    if isinstance(rec, OHLCVBar):
        return (rec.symbol, rec.kind, rec.interval, _to_us(rec.event_time))
    if isinstance(rec, FundamentalFact):
        return (rec.symbol, rec.kind, rec.metric, rec.period, _to_us(rec.event_time))
    if isinstance(rec, CorporateAction):
        return (rec.symbol, rec.kind, rec.action_type, _to_us(rec.event_time))
    if isinstance(rec, NewsItem):
        # News is not restated; identity is the article itself.
        return (rec.symbol, rec.kind, rec.url or rec.title, _to_us(rec.event_time))
    return (rec.symbol, rec.kind, _to_us(rec.event_time))  # pragma: no cover


class PointInTimeStore:
    """An append-only, as-of-queryable store for :class:`KnowledgeTimedRecord`."""

    def __init__(self, path: str | Path = ":memory:") -> None:
        self._path = str(path)
        self._conn = sqlite3.connect(self._path)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL")
        self._create_schema()

    # ------------------------------------------------------------------ schema
    def _create_schema(self) -> None:
        for kind in RecordKind:
            table = self._table(kind)
            self._conn.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table} (
                    id           INTEGER PRIMARY KEY AUTOINCREMENT,
                    symbol       TEXT    NOT NULL,
                    event_us     INTEGER NOT NULL,
                    knowledge_us INTEGER NOT NULL,
                    revision     INTEGER NOT NULL DEFAULT 0,
                    payload      TEXT    NOT NULL
                )
                """
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS ix_{table}_sym_know "
                f"ON {table}(symbol, knowledge_us)"
            )
            self._conn.execute(
                f"CREATE INDEX IF NOT EXISTS ix_{table}_sym_event "
                f"ON {table}(symbol, event_us)"
            )
        self._conn.commit()

    @staticmethod
    def _table(kind: RecordKind) -> str:
        return f"rec_{kind.value}"

    # ------------------------------------------------------------------ writes
    def append(self, record: AnyRecord) -> None:
        """Insert one record (append-only)."""
        self.append_many([record])

    def append_many(self, records: Iterable[AnyRecord]) -> int:
        """Insert many records; returns the count written."""
        n = 0
        for rec in records:
            table = self._table(rec.kind)
            self._conn.execute(
                f"INSERT INTO {table} (symbol, event_us, knowledge_us, revision, payload) "
                f"VALUES (?, ?, ?, ?, ?)",
                (
                    rec.symbol,
                    _to_us(rec.event_time),
                    _to_us(rec.knowledge_time),
                    rec.revision,
                    rec.model_dump_json(),
                ),
            )
            n += 1
        self._conn.commit()
        return n

    # ------------------------------------------------------------------- reads
    def query(
        self,
        kind: RecordKind,
        symbol: str,
        clock: AsOfClock,
        *,
        event_start: datetime | None = None,
        event_end: datetime | None = None,
        latest_per_series: bool = True,
        context: str | None = None,
    ) -> list[KnowledgeTimedRecord]:
        """Return records of ``kind`` for ``symbol`` knowable at ``clock``.

        Only rows with ``knowledge_time <= clock.as_of`` are considered (enforced
        in SQL). With ``latest_per_series`` (default) restatements collapse to the
        latest-known revision per logical series key. Results are re-verified by
        the look-ahead guard before returning.
        """
        table = self._table(kind)
        model = model_for_kind(kind)
        ctx = context or f"query({kind.value}, {symbol})"

        sql = f"SELECT payload FROM {table} WHERE symbol = ? AND knowledge_us <= ?"
        params: list[object] = [symbol.strip().upper(), _to_us(clock.as_of)]
        if event_start is not None:
            sql += " AND event_us >= ?"
            params.append(_to_us(event_start))
        if event_end is not None:
            sql += " AND event_us <= ?"
            params.append(_to_us(event_end))
        sql += " ORDER BY event_us ASC, knowledge_us ASC, revision ASC"

        rows: Sequence[sqlite3.Row] = self._conn.execute(sql, params).fetchall()
        records = [model.model_validate_json(r["payload"]) for r in rows]

        if latest_per_series:
            records = self._collapse_latest(records)

        # Belt-and-suspenders: prove the SQL filter did its job.
        return filter_visible(records, clock, context=ctx, strict=True)

    @staticmethod
    def _collapse_latest(records: list[KnowledgeTimedRecord]) -> list[KnowledgeTimedRecord]:
        """Keep, per series key, the record with the greatest (knowledge_time, revision)."""
        best: dict[tuple, KnowledgeTimedRecord] = {}
        for rec in records:
            key = _series_key(rec)
            cur = best.get(key)
            if cur is None or (rec.knowledge_time, rec.revision) >= (
                cur.knowledge_time,
                cur.revision,
            ):
                best[key] = rec
        return sorted(best.values(), key=lambda r: (r.event_time, r.knowledge_time))

    # --------------------------------------------------------- typed accessors
    def get_ohlcv(
        self,
        symbol: str,
        clock: AsOfClock,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[OHLCVBar]:
        """Raw (unadjusted) bars in ``[start, end]`` knowable at ``clock``."""
        return [
            r  # type: ignore[misc]
            for r in self.query(
                RecordKind.OHLCV, symbol, clock, event_start=start, event_end=end
            )
        ]

    def get_news(
        self,
        symbol: str,
        clock: AsOfClock,
        *,
        since: datetime | None = None,
    ) -> list[NewsItem]:
        """News published on/before ``clock`` (and after ``since`` if given).

        News is keyed by publish time (== knowledge_time), so this is inherently
        publish-time-only retrieval — the Step 1 mandate for news/sentiment.
        """
        return [
            r  # type: ignore[misc]
            for r in self.query(
                RecordKind.NEWS, symbol, clock, event_start=since, latest_per_series=False
            )
        ]

    def get_fundamentals(
        self,
        symbol: str,
        clock: AsOfClock,
        *,
        metric: str | None = None,
    ) -> list[FundamentalFact]:
        """Latest-known fundamental facts per fiscal period, as of ``clock``."""
        facts = [r for r in self.query(RecordKind.FUNDAMENTAL, symbol, clock)]
        if metric is not None:
            facts = [f for f in facts if isinstance(f, FundamentalFact) and f.metric == metric]
        return facts  # type: ignore[return-value]

    def get_corporate_actions(
        self,
        symbol: str,
        clock: AsOfClock,
        *,
        start: datetime | None = None,
        end: datetime | None = None,
    ) -> list[CorporateAction]:
        """Splits/dividends known at ``clock`` — the only inputs allowed to adjust prices."""
        return [
            r  # type: ignore[misc]
            for r in self.query(
                RecordKind.CORPORATE_ACTION, symbol, clock, event_start=start, event_end=end
            )
        ]

    # ------------------------------------------------------------------ misc
    def close(self) -> None:
        self._conn.close()

    def __enter__(self) -> PointInTimeStore:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
