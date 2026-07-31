"""Decision-model protocol and a deterministic offline model.

The backtest engine depends only on :class:`DecisionModel`, so it runs identically
against the real TradingAgents graph (``vts/integration/ta_decision.py``) or the
deterministic :class:`FakeMomentumModel` used to test the harness offline (no LLM,
no network, no API key).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from vts.decision import Decision, Rating
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore


@runtime_checkable
class DecisionModel(Protocol):
    """Produces a :class:`Decision` for one (ticker, as-of clock)."""

    model_id: str

    def prompt_for(self, ticker: str, clock: AsOfClock) -> str:
        """A stable string identifying the inputs, hashed into the cache key."""
        ...

    def decide(self, ticker: str, clock: AsOfClock) -> Decision:
        """Return a decision using only data knowable at ``clock``."""
        ...


@dataclass(frozen=True, slots=True)
class MomentumParams:
    lookback_bars: int = 20
    strong_threshold: float = 0.10   # |return| above this -> Buy/Sell
    mild_threshold: float = 0.02     # |return| above this -> Overweight/Underweight
    confidence_scale: float = 0.20   # |return| mapped to confidence, saturating at 1


class FakeMomentumModel:
    """A deterministic trailing-momentum model reading only the point-in-time store.

    Purely a test/reference harness: it maps the trailing return (as-of the clock,
    so inherently look-ahead-free) onto the 5-tier scale. Deterministic, so N
    samples agree (dispersion 0) — exactly what we want when validating the engine
    itself; a real stochastic LLM produces the dispersion the vote is designed for.
    """

    model_id = "fake-momentum"

    def __init__(self, store: PointInTimeStore, params: MomentumParams | None = None) -> None:
        self._store = store
        self.p = params or MomentumParams()

    def prompt_for(self, ticker: str, clock: AsOfClock) -> str:
        # Encode EVERY param that affects decide(): the cache key hashes this prompt,
        # so a param change must alter it or a persistent cache would serve stale
        # ratings computed under a different configuration.
        p = self.p
        return (
            f"momentum(lb={p.lookback_bars},strong={p.strong_threshold},"
            f"mild={p.mild_threshold},cscale={p.confidence_scale}):"
            f"{ticker.upper()}@{clock.as_of.isoformat()}"
        )

    def _momentum(self, ticker: str, clock: AsOfClock) -> float | None:
        bars = self._store.get_ohlcv(ticker, clock)
        if len(bars) <= self.p.lookback_bars:
            return None
        recent = bars[-1].close
        past = bars[-1 - self.p.lookback_bars].close
        if past <= 0:
            return None
        return recent / past - 1.0

    def decide(self, ticker: str, clock: AsOfClock) -> Decision:
        mom = self._momentum(ticker, clock)
        if mom is None:
            return Decision(
                ticker=ticker, as_of=clock.as_of.isoformat(), rating=Rating.HOLD,
                confidence=0.0, thesis="insufficient price history", model_id=self.model_id,
            )
        if mom >= self.p.strong_threshold:
            rating = Rating.BUY
        elif mom >= self.p.mild_threshold:
            rating = Rating.OVERWEIGHT
        elif mom <= -self.p.strong_threshold:
            rating = Rating.SELL
        elif mom <= -self.p.mild_threshold:
            rating = Rating.UNDERWEIGHT
        else:
            rating = Rating.HOLD
        confidence = min(1.0, abs(mom) / self.p.confidence_scale)
        return Decision(
            ticker=ticker, as_of=clock.as_of.isoformat(), rating=rating,
            confidence=confidence, thesis=f"trailing {self.p.lookback_bars}-bar return {mom:+.2%}",
            model_id=self.model_id,
        )
