"""Implementation shortfall — the daily gap between paper and backtest.

The backtest (Step 2) models weight-based fills at the decision close with no
share granularity and no order rejections. The paper loop executes integer-share
orders that pass real validation and pay real costs. The difference is
implementation shortfall (백테스트 성과와의 괴리): logged every day so drift
between the model and realizable trading is visible as it accrues, not only at
the end.

Sign convention: ``shortfall = backtest_equity - paper_equity`` — POSITIVE means
the paper portfolio underperformed the idealized backtest (the usual case: costs
+ share rounding + rejections drag realized results below the model).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class ShortfallRecord(BaseModel):
    """One day's implementation-shortfall observation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    date: str
    paper_equity: float
    backtest_equity: float | None            # None when the reference has no point for this date
    shortfall: float | None                  # backtest - paper (None if no reference)
    shortfall_bps_of_capital: float | None   # shortfall / initial_capital, in bps
    cumulative_shortfall: float


def make_record(
    date_iso: str,
    paper_equity: float,
    backtest_equity: float | None,
    initial_capital: float,
    prior_cumulative: float,
) -> ShortfallRecord:
    if backtest_equity is None:
        return ShortfallRecord(
            date=date_iso, paper_equity=paper_equity, backtest_equity=None,
            shortfall=None, shortfall_bps_of_capital=None,
            cumulative_shortfall=prior_cumulative,
        )
    shortfall = backtest_equity - paper_equity
    bps = (shortfall / initial_capital) * 1e4 if initial_capital > 0 else 0.0
    return ShortfallRecord(
        date=date_iso, paper_equity=paper_equity, backtest_equity=backtest_equity,
        shortfall=shortfall, shortfall_bps_of_capital=bps,
        cumulative_shortfall=prior_cumulative + shortfall,
    )
