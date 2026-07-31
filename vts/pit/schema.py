"""Pydantic schemas for point-in-time records.

The invariants enforced here are the backbone of no-look-ahead correctness, so
they are validated at construction time rather than trusted from callers:

- all timestamps are timezone-aware and normalized to UTC;
- ``knowledge_time >= event_time`` (you cannot know a fact before it occurs);
- OHLCV is stored **raw / unadjusted** — adjustment is a *view* computed from
  separately-timed :class:`CorporateAction` records, never baked into storage.
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class RecordKind(str, Enum):
    """Discriminator used by :class:`~vts.pit.store.PointInTimeStore` tables."""

    OHLCV = "ohlcv"
    NEWS = "news"
    FUNDAMENTAL = "fundamental"
    CORPORATE_ACTION = "corporate_action"


def _to_utc(value: datetime) -> datetime:
    """Return ``value`` as a timezone-aware UTC datetime.

    A naive datetime is *rejected* rather than silently assumed-UTC: silent
    assumptions are exactly how host-timezone look-ahead bugs (Step 0, #1126)
    creep in. Callers must be explicit about the zone of every timestamp.
    """
    if value.tzinfo is None:
        raise ValueError(
            "naive datetime is not allowed; attach an explicit tzinfo "
            "(all PIT timestamps must be timezone-aware)"
        )
    return value.astimezone(timezone.utc)


class KnowledgeTimedRecord(BaseModel):
    """Base for every stored datum.

    ``model_config`` freezes instances so a record cannot be mutated after the
    invariants are checked — a stored fact is immutable history; a correction is
    a *new* record with a later ``knowledge_time`` and higher ``revision``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1, description="Canonical instrument symbol, e.g. 'AAPL'.")
    event_time: datetime = Field(description="When the datum's subject occurred (UTC).")
    knowledge_time: datetime = Field(
        description="When the datum first became knowable to a live observer (UTC)."
    )
    source: str = Field(min_length=1, description="Vendor/source identifier, e.g. 'alpha_vantage'.")
    revision: int = Field(
        default=0,
        ge=0,
        description="Monotonic correction counter; a restatement is a new record, revision+1.",
    )
    knowledge_time_estimated: bool = Field(
        default=False,
        description=(
            "True when knowledge_time was approximated (e.g. filing date inferred "
            "from fiscal period + reporting lag) rather than sourced exactly. "
            "Backtests may choose to treat estimated knowledge conservatively."
        ),
    )

    @field_validator("event_time", "knowledge_time")
    @classmethod
    def _tz_aware_utc(cls, v: datetime) -> datetime:
        return _to_utc(v)

    @field_validator("symbol")
    @classmethod
    def _normalize_symbol(cls, v: str) -> str:
        return v.strip().upper()

    @model_validator(mode="after")
    def _knowledge_not_before_event(self) -> KnowledgeTimedRecord:
        if self.knowledge_time < self.event_time:
            raise ValueError(
                f"knowledge_time ({self.knowledge_time.isoformat()}) is before "
                f"event_time ({self.event_time.isoformat()}): a fact cannot be "
                f"known before it happens"
            )
        return self

    @property
    def kind(self) -> RecordKind:  # pragma: no cover - overridden by subclasses
        raise NotImplementedError


NonNegFloat = Annotated[float, Field(ge=0.0)]


class OHLCVBar(KnowledgeTimedRecord):
    """One raw (unadjusted) OHLCV bar.

    ``event_time`` is the session/interval close; ``knowledge_time`` defaults to
    the same instant (the close is known at the close). Prices are the vendor's
    *raw* prints — never split/dividend adjusted — so no future corporate action
    can retroactively change a stored bar.
    """

    interval: Literal["1d", "1h", "4h"] = "1d"
    open: NonNegFloat
    high: NonNegFloat
    low: NonNegFloat
    close: NonNegFloat
    volume: NonNegFloat

    @model_validator(mode="after")
    def _ohlc_consistent(self) -> OHLCVBar:
        if self.high < self.low:
            raise ValueError(f"high ({self.high}) < low ({self.low})")
        if not (self.low <= self.open <= self.high):
            raise ValueError(f"open ({self.open}) outside [low, high]")
        if not (self.low <= self.close <= self.high):
            raise ValueError(f"close ({self.close}) outside [low, high]")
        return self

    @property
    def kind(self) -> RecordKind:
        return RecordKind.OHLCV


class CorporateAction(KnowledgeTimedRecord):
    """A split or cash dividend, timed by when it became known (announcement/ex-date).

    Kept separate from OHLCV so adjustment factors are applied using only the
    actions visible at the backtest's as-of clock.
    """

    action_type: Literal["split", "dividend"]
    # For a split: shares-multiplier (2.0 == 2:1). For a dividend: cash per share.
    value: float = Field(gt=0.0)

    @property
    def kind(self) -> RecordKind:
        return RecordKind.CORPORATE_ACTION


class NewsItem(KnowledgeTimedRecord):
    """A news / sentiment item, keyed by publish time (== knowledge_time).

    ``event_time`` and ``knowledge_time`` are both the publish instant: an
    article is knowable exactly when it is published.
    """

    title: str = Field(min_length=1)
    url: str = ""
    publisher: str = ""
    summary: str = ""
    # Optional pre-computed sentiment (e.g. Alpha Vantage NEWS_SENTIMENT).
    sentiment_score: float | None = Field(default=None, ge=-1.0, le=1.0)
    relevance: float | None = Field(default=None, ge=0.0, le=1.0)

    @property
    def kind(self) -> RecordKind:
        return RecordKind.NEWS


class FundamentalFact(KnowledgeTimedRecord):
    """One fundamental datapoint keyed by fiscal period (event) and release date (knowledge).

    ``event_time`` is the fiscal period end; ``knowledge_time`` is the report /
    filing release date. When the release date is unknown it is approximated as
    ``fiscal_period_end + reporting_lag`` and ``knowledge_time_estimated`` is set.
    """

    metric: str = Field(min_length=1, description="e.g. 'eps', 'totalRevenue', 'netIncome'.")
    value: float | None = None
    period: Literal["quarterly", "annual"] = "quarterly"
    currency: str = "USD"

    @property
    def kind(self) -> RecordKind:
        return RecordKind.FUNDAMENTAL


# Convenience union for typed store operations.
AnyRecord = OHLCVBar | CorporateAction | NewsItem | FundamentalFact

_KIND_TO_MODEL: dict[RecordKind, type[KnowledgeTimedRecord]] = {
    RecordKind.OHLCV: OHLCVBar,
    RecordKind.CORPORATE_ACTION: CorporateAction,
    RecordKind.NEWS: NewsItem,
    RecordKind.FUNDAMENTAL: FundamentalFact,
}


def model_for_kind(kind: RecordKind) -> type[KnowledgeTimedRecord]:
    """Return the concrete model class for a :class:`RecordKind`."""
    return _KIND_TO_MODEL[kind]
