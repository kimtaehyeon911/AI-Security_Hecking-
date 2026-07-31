"""The decision schema — Trading-R1's evidence-based thesis → 5-tier rating.

Mirrors TradingAgents' ``PortfolioRating`` (Buy/Overweight/Hold/Underweight/Sell)
so parsed graph output maps 1:1, and adds the two fields the paper's structure
and our risk layer (Step 4) need but upstream lacks: a numeric ``confidence`` and
a structured ``evidence`` list. Aggregation across N stochastic samples
(majority-vote rating + dispersion) also lives here.
"""

from __future__ import annotations

import statistics
from collections import Counter
from enum import Enum

from pydantic import BaseModel, ConfigDict, Field


class Rating(str, Enum):
    """5-tier rating, most bullish to most bearish (== TradingAgents PortfolioRating)."""

    BUY = "Buy"
    OVERWEIGHT = "Overweight"
    HOLD = "Hold"
    UNDERWEIGHT = "Underweight"
    SELL = "Sell"


# Ordinal scale for dispersion math and signed target weights.
_ORDINAL: dict[Rating, int] = {
    Rating.BUY: 2,
    Rating.OVERWEIGHT: 1,
    Rating.HOLD: 0,
    Rating.UNDERWEIGHT: -1,
    Rating.SELL: -2,
}


def rating_to_signed_weight(rating: Rating) -> float:
    """Map a rating to a signed target weight in [-1, 1] (before risk limits)."""
    return _ORDINAL[rating] / 2.0


class Evidence(BaseModel):
    """One evidence item backing the thesis (Trading-R1 structure)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    claim: str = Field(min_length=1)
    source: str = ""
    # Direction this evidence pushes the thesis.
    stance: str = Field(default="neutral", description="bullish | bearish | neutral")


class Decision(BaseModel):
    """A single model decision for one (ticker, date)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str = Field(min_length=1)
    as_of: str = Field(description="ISO timestamp of the as-of clock this decision was made under.")
    rating: Rating
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Model confidence in [0,1]; Step 4 forces Hold below a threshold.",
    )
    thesis: str = ""
    evidence: tuple[Evidence, ...] = ()
    model_id: str = ""

    @property
    def signed_weight(self) -> float:
        return rating_to_signed_weight(self.rating)


class AggregatedDecision(BaseModel):
    """Majority-vote result over N samples, with a dispersion metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    as_of: str
    rating: Rating
    n_samples: int = Field(ge=1)
    agreement: float = Field(ge=0.0, le=1.0, description="fraction of samples that chose the majority rating")
    dispersion: float = Field(ge=0.0, description="stdev of sample ratings on the ordinal scale")
    mean_confidence: float = Field(ge=0.0, le=1.0)
    votes: dict[str, int] = Field(default_factory=dict)
    tie_broken_to_hold: bool = False


def aggregate_decisions(samples: list[Decision]) -> AggregatedDecision:
    """Combine N samples into a majority-vote rating plus dispersion.

    Ties (no strict plurality winner, or a tie for the top count) resolve to
    ``Hold`` — the conservative default consistent with the Step 4 risk policy —
    and the fact is recorded in ``tie_broken_to_hold``. ``dispersion`` is the
    population stdev of the samples' ordinal rating values, so an all-agree batch
    scores 0 and a Buy/Sell split scores high.
    """
    if not samples:
        raise ValueError("aggregate_decisions requires at least one sample")

    counts = Counter(s.rating for s in samples)
    top = counts.most_common()
    best_count = top[0][1]
    winners = [r for r, c in top if c == best_count]

    tie = len(winners) > 1
    rating = Rating.HOLD if tie else winners[0]

    ordinals = [_ORDINAL[s.rating] for s in samples]
    dispersion = statistics.pstdev(ordinals) if len(ordinals) > 1 else 0.0

    return AggregatedDecision(
        ticker=samples[0].ticker,
        as_of=samples[0].as_of,
        rating=rating,
        n_samples=len(samples),
        agreement=best_count / len(samples),
        dispersion=dispersion,
        mean_confidence=sum(s.confidence for s in samples) / len(samples),
        votes={r.value: c for r, c in counts.items()},
        tie_broken_to_hold=tie,
    )
