"""Point-in-time price adjustment.

Step 0 found the yfinance path uses ``auto_adjust=True`` — prices back-adjusted
for *all* splits/dividends, including ones that happen after the date being
viewed. That is a restatement leak: the "history" you see on date D depends on
the future.

Here, adjustment is a **view** computed from raw bars plus only the corporate
actions knowable at the as-of clock. Ask for the adjusted series as-of T and you
get exactly what an observer at T could have computed — no future action leaks in.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from vts.pit.clock import AsOfClock
from vts.pit.schema import CorporateAction, OHLCVBar


@dataclass(frozen=True, slots=True)
class AdjustedBar:
    """A bar with a split-adjusted close, computed as-of a specific clock."""

    event_time: datetime
    raw_close: float
    adj_close: float
    split_factor: float


def split_adjusted_closes(
    bars: list[OHLCVBar],
    actions: list[CorporateAction],
    clock: AsOfClock,
) -> list[AdjustedBar]:
    """Return split-adjusted closes using only splits known at ``clock``.

    Back-adjustment convention: a split with coefficient ``c`` and ex-date ``e``
    divides the close of every bar *strictly before* ``e`` by the cumulative
    product of later splits — but only splits with ``knowledge_time <= clock`` are
    included. Dividends are intentionally not folded into the price here (total-
    return adjustment is a separate, explicit choice); this function isolates the
    split-restatement hazard.
    """
    known_splits = sorted(
        (a for a in actions if a.action_type == "split" and clock.is_visible(a.knowledge_time)),
        key=lambda a: a.event_time,
    )
    out: list[AdjustedBar] = []
    for bar in sorted(bars, key=lambda b: b.event_time):
        # Cumulative factor from splits that are ex-dated strictly after this bar.
        factor = 1.0
        for sp in known_splits:
            if sp.event_time > bar.event_time:
                factor *= sp.value
        out.append(
            AdjustedBar(
                event_time=bar.event_time,
                raw_close=bar.close,
                adj_close=bar.close / factor,
                split_factor=factor,
            )
        )
    return out
