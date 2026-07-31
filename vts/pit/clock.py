"""The simulated 'now' for point-in-time reads.

Every read path in a backtest is parameterized by an :class:`AsOfClock`. It is
the single source of truth for "how much of the world is visible right now",
so a backtest can never accidentally read live/wall-clock data.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone


@dataclass(frozen=True, slots=True)
class AsOfClock:
    """An immutable knowledge ceiling.

    A record is visible iff ``record.knowledge_time <= as_of``. The clock is
    frozen; advancing time means constructing a new clock (``clock.advance_to``),
    which makes accidental in-place mutation impossible.
    """

    as_of: datetime

    def __post_init__(self) -> None:
        if self.as_of.tzinfo is None:
            raise ValueError("AsOfClock.as_of must be timezone-aware")
        # Normalize to UTC without mutating the frozen field in place.
        object.__setattr__(self, "as_of", self.as_of.astimezone(timezone.utc))

    @classmethod
    def at(cls, when: datetime) -> AsOfClock:
        """Construct a clock at ``when`` (must be tz-aware)."""
        return cls(as_of=when)

    def is_visible(self, knowledge_time: datetime) -> bool:
        """Whether a datum with ``knowledge_time`` is knowable at this clock."""
        kt = knowledge_time if knowledge_time.tzinfo else knowledge_time.replace(tzinfo=timezone.utc)
        return kt.astimezone(timezone.utc) <= self.as_of

    def advance_to(self, when: datetime) -> AsOfClock:
        """Return a new clock at ``when``; refuses to move backwards."""
        new = AsOfClock.at(when)
        if new.as_of < self.as_of:
            raise ValueError(
                f"AsOfClock cannot move backwards: {self.as_of.isoformat()} -> "
                f"{new.as_of.isoformat()}"
            )
        return new

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        return f"AsOfClock({self.as_of.isoformat()})"
