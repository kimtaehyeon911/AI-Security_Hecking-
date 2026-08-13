"""Implementation shortfall — the daily gap between paper and backtest.

The backtest (Step 2) models weight-based fills at the decision close with no
share granularity and no order rejections. The paper loop executes integer-share
orders that pass real validation and pay real costs. The difference is
implementation shortfall (백테스트 성과와의 괴리): logged every day so drift
between the model and realizable trading is visible as it accrues.

Definitions (both marks measured POST-cost so an identically-trading book reads ~0):

- ``gap`` = ``backtest_equity - paper_equity`` — the TOTAL accumulated drift to
  date. POSITIVE means paper underperformed the model. Equity is already a
  cumulative quantity, so the current gap IS the cumulative shortfall — it must
  not be re-summed each day (that would re-count a standing gap N times).
- ``daily_shortfall`` = today's gap − yesterday's gap — the drift ADDED today
  (costs + share rounding + rejections for the day, or a halt divergence).
- ``paper_halted`` / ``reference_halted`` — surfaced so days where the two risk
  paths disagree are visibly attributable, not silently folded into "execution".
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ShortfallRecord(BaseModel):
    """One day's implementation-shortfall observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: str
    paper_equity: float
    backtest_equity: float | None            # None when the reference has no point (e.g. kill switch)
    gap: float | None                        # total drift to date = backtest - paper
    daily_shortfall: float | None            # gap - prior_gap (drift added today)
    gap_bps_of_capital: float | None         # total gap in bps of initial capital
    paper_halted: bool = False
    reference_halted: bool = False


def make_record(
    date_iso: str,
    paper_equity: float,
    backtest_equity: float | None,
    initial_capital: float,
    prior_gap: float,
    *,
    paper_halted: bool = False,
    reference_halted: bool = False,
) -> ShortfallRecord:
    if backtest_equity is None:
        return ShortfallRecord(
            date=date_iso, paper_equity=paper_equity, backtest_equity=None,
            gap=None, daily_shortfall=None, gap_bps_of_capital=None,
            paper_halted=paper_halted, reference_halted=reference_halted,
        )
    gap = backtest_equity - paper_equity
    bps = (gap / initial_capital) * 1e4 if initial_capital > 0 else 0.0
    return ShortfallRecord(
        date=date_iso, paper_equity=paper_equity, backtest_equity=backtest_equity,
        gap=gap, daily_shortfall=gap - prior_gap, gap_bps_of_capital=bps,
        paper_halted=paper_halted, reference_halted=reference_halted,
    )
