"""Walk-forward splitting into train/test folds.

Step 2 mandate: *"워크포워드 방식. train/test 구간 분리, 구간별 리포트."* For an
agent that carries a memory log rather than fitted parameters, the "train" window
is the reflection/warm-up period and "test" is strictly out-of-sample evaluation.
Test windows never overlap, so a segment report is a clean OOS measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime


@dataclass(frozen=True, slots=True)
class Fold:
    """One walk-forward fold over an ordered list of decision dates."""

    index: int
    train: tuple[datetime, ...]
    test: tuple[datetime, ...]

    @property
    def test_start(self) -> datetime:
        return self.test[0]

    @property
    def test_end(self) -> datetime:
        return self.test[-1]


def walk_forward(
    dates: Sequence[datetime],
    train_size: int,
    test_size: int,
    *,
    step: int | None = None,
    anchored: bool = False,
) -> list[Fold]:
    """Produce non-overlapping-test walk-forward folds.

    ``anchored=True`` grows the train window from the start (expanding window);
    otherwise the train window rolls forward with the test window. ``step`` defaults
    to ``test_size`` so successive test windows tile the timeline without overlap.
    """
    if train_size <= 0 or test_size <= 0:
        raise ValueError("train_size and test_size must be positive")
    ordered = list(dates)
    if any(ordered[i] > ordered[i + 1] for i in range(len(ordered) - 1)):
        raise ValueError("dates must be sorted ascending")
    step = step or test_size

    folds: list[Fold] = []
    start = 0
    idx = 0
    while True:
        train_end = start + train_size
        test_end = train_end + test_size
        if test_end > len(ordered):
            break
        train_lo = 0 if anchored else start
        folds.append(
            Fold(
                index=idx,
                train=tuple(ordered[train_lo:train_end]),
                test=tuple(ordered[train_end:test_end]),
            )
        )
        idx += 1
        start += step
    return folds
