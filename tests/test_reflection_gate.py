"""Falsification tests for the deferred-reflection look-ahead gate (Step 0's #1 leak).

Written before the engine: the property to protect is that a chronological replay
never resolves a decision whose outcome is not yet realized at the simulated date.
"""

from __future__ import annotations

from datetime import date

import pytest

from vts.backtest.reflection_gate import PendingEntry, partition_pending


def test_entry_inside_holding_window_is_deferred():
    """The core anti-leak property: an unfinished holding window cannot be resolved."""
    pending = [PendingEntry("AAPL", date(2024, 1, 10))]
    # holding=5d -> outcome known 2024-01-15. At 2024-01-12 it is NOT yet known.
    split = partition_pending(pending, date(2024, 1, 12), holding_days=5)
    assert split.resolvable == ()
    assert split.deferred == tuple(pending)


def test_entry_resolves_exactly_when_window_closes():
    pending = [PendingEntry("AAPL", date(2024, 1, 10))]
    split = partition_pending(pending, date(2024, 1, 15), holding_days=5)  # boundary
    assert split.resolvable == tuple(pending)
    assert split.deferred == ()


def test_mixed_batch_only_matured_entries_resolve():
    pending = [
        PendingEntry("AAPL", date(2024, 1, 1)),   # matured
        PendingEntry("AAPL", date(2024, 1, 9)),   # matured (14 >= 6? entry+5=14)
        PendingEntry("AAPL", date(2024, 1, 12)),  # not matured (17 > 14)
    ]
    split = partition_pending(pending, date(2024, 1, 14), holding_days=5)
    assert {e.entry_date for e in split.resolvable} == {date(2024, 1, 1), date(2024, 1, 9)}
    assert {e.entry_date for e in split.deferred} == {date(2024, 1, 12)}


def test_accepts_dict_entries_and_iso_dates():
    split = partition_pending(
        [{"ticker": "MSFT", "date": "2024-03-01"}], "2024-03-20", holding_days=5
    )
    assert len(split.resolvable) == 1 and split.resolvable[0].ticker == "MSFT"


def test_negative_holding_rejected():
    with pytest.raises(ValueError):
        partition_pending([], date(2024, 1, 1), holding_days=-1)


def test_no_future_entry_ever_leaks_across_a_sweep():
    """Sweep 'now' forward day by day; a decision is resolvable only strictly after
    its holding window — never before. This is the property a naive replay violates."""
    entry = PendingEntry("AAPL", date(2024, 1, 10))
    holding = 5
    for day in range(10, 20):
        td = date(2024, 1, day)
        split = partition_pending([entry], td, holding_days=holding)
        resolved = bool(split.resolvable)
        should_be_resolved = day >= 15  # entry_date(10) + holding(5)
        assert resolved == should_be_resolved, f"leak/lag at day {day}"
