"""Pre-flight gate: refuse a backtest window that is NOT cutoff-clean.

Encodes the mandate "백테스트 구간은 사용 모델의 knowledge cutoff(effective) 이후로만" as a
hard, runnable check the runbook calls BEFORE spending money on a backtest. It reads
the SAME registry the engine uses (``vts/backtest/model_cutoffs.json``) so there is one
source of truth.

Usage:
    python scripts/gate/check_window.py <model_id> <start YYYY-MM-DD> [<end YYYY-MM-DD>]

Exit codes:
    0  every date in [start, end] is CLEAN (or the model is contamination-exempt)
    3  the window is CONTAMINATED (starts on/before the effective cutoff)
    4  the window is UNKNOWN (model cutoff not filled/verified) — fill it first
    2  bad arguments
"""

from __future__ import annotations

import sys
from datetime import date, datetime, timedelta

from vts.backtest.cutoff import Contamination, CutoffRegistry


def _d(s: str) -> date:
    return datetime.strptime(s, "%Y-%m-%d").date()


def main(argv: list[str]) -> int:
    if not (2 <= len(argv) <= 3):
        print(__doc__, file=sys.stderr)
        return 2
    model_id = argv[0]
    try:
        start = _d(argv[1])
        end = _d(argv[2]) if len(argv) == 3 else start
    except ValueError as exc:
        print(f"bad date: {exc}", file=sys.stderr)
        return 2
    if end < start:
        print(f"end {end} is before start {start}", file=sys.stderr)
        return 2

    reg = CutoffRegistry.load()
    # Classify the window boundaries AND every day between — a single contaminated
    # day taints the segment, exactly like the engine's segment_status.
    dates: list[date] = []
    d = start
    while d <= end:
        dates.append(d)
        d += timedelta(days=1)
    status = reg.segment_status(model_id, dates)

    eff = reg.cutoff_for(model_id)
    eff_s = eff.isoformat() if eff else "(none / unverified)"
    exempt = reg.is_exempt(model_id)

    if status == Contamination.CLEAN:
        note = "contamination-exempt (deterministic non-LLM)" if exempt else \
               f"entirely after effective cutoff {eff_s}"
        print(f"OK  [{model_id}] {start}..{end} is CLEAN — {note}")
        return 0
    if status == Contamination.CONTAMINATED:
        print(
            f"FAIL [{model_id}] {start}..{end} is CONTAMINATED — effective cutoff is "
            f"{eff_s}; move the window to start AFTER it (contamination lies BEFORE the "
            f"cutoff). Results in this window are reference-only, never certifiable.",
            file=sys.stderr,
        )
        return 3
    print(
        f"FAIL [{model_id}] cutoff is UNKNOWN — no verified entry in model_cutoffs.json. "
        f"Fill it (max(sources)+buffer_days) before backtesting; the gate never guesses "
        f"a window clean.",
        file=sys.stderr,
    )
    return 4


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
