"""Adapter: drive the real TradingAgents graph as a :class:`DecisionModel`.

Three concerns, separated so the look-ahead-critical logic is unit-tested without
the (heavy, network-bound) agent graph installed:

1. ``rating_from_text`` / ``build_decision`` — pure parsing of the graph's 5-tier
   output into our :class:`~vts.decision.Decision`. Fully tested offline.
2. ``resolvable_reflections`` — the gated reflection resolver: partition pending
   entries (only matured ones) and score them via the point-in-time store, not
   live yfinance. Fully tested offline.
3. ``TradingAgentsDecisionModel`` — the live wrapper that sets the as-of clock,
   installs the reflection gate, calls ``propagate`` and parses the result. Runs
   only with the fork installed (``pragma: no cover``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable

from vts.backtest.reflection_gate import PendingEntry, partition_pending
from vts.backtest.returns import realized_return
from vts.decision import Decision, Rating
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore

_RATING_BY_VALUE = {r.value.lower(): r for r in Rating}


def rating_from_text(text: str, default: Rating = Rating.HOLD) -> Rating:
    """Tolerantly extract a 5-tier rating from prose (mirrors TA's parse_rating).

    Looks for an explicit 'Rating: X' first, then the first rating word anywhere.
    Falls back to ``default`` (Hold) — the same conservative default the risk layer
    uses for unparseable output.
    """
    for line in text.splitlines():
        low = line.lower()
        if "rating" in low:
            for token in low.replace(":", " ").replace("*", " ").replace("-", " ").split():
                if token in _RATING_BY_VALUE:
                    return _RATING_BY_VALUE[token]
    for token in text.lower().replace("*", " ").split():
        clean = token.strip(":.,")
        if clean in _RATING_BY_VALUE:
            return _RATING_BY_VALUE[clean]
    return default


def build_decision(
    ticker: str,
    clock: AsOfClock,
    rating_text: str,
    *,
    thesis: str = "",
    confidence: float | None = None,
    model_id: str = "tradingagents",
    default_confidence: float = 0.5,
) -> Decision:
    """Build a :class:`Decision` from the graph's rating text.

    TradingAgents' ``PortfolioDecision`` has no confidence field, so when the graph
    does not supply one we use ``default_confidence`` and rely on the engine's
    N-sample dispersion/agreement as the confidence signal (Step 4 will thread a
    real per-decision confidence once the schema is extended).
    """
    conf = default_confidence if confidence is None else max(0.0, min(1.0, confidence))
    return Decision(
        ticker=ticker,
        as_of=clock.as_of.isoformat(),
        rating=rating_from_text(rating_text),
        confidence=conf,
        thesis=thesis,
        model_id=model_id,
    )


def resolvable_reflections(
    store: PointInTimeStore,
    pending: list[PendingEntry | dict],
    trade_date: datetime,
    holding_days: int,
) -> list[tuple[PendingEntry, float]]:
    """Return (matured entry, realized PIT return) pairs safe to reflect on now.

    Combines the reflection gate (only matured entries) with PIT-capped returns, so
    replacing TradingAgents' ``_resolve_pending_entries`` with this eliminates the
    deferred-reflection look-ahead entirely.
    """
    ceiling = AsOfClock.at(trade_date)
    split = partition_pending(pending, trade_date, holding_days)
    out: list[tuple[PendingEntry, float]] = []
    for entry in split.resolvable:
        ret = realized_return(
            store, entry.ticker, datetime(
                entry.entry_date.year, entry.entry_date.month, entry.entry_date.day,
                tzinfo=ceiling.as_of.tzinfo,
            ), holding_days, ceiling,
        )
        if ret is not None:
            out.append((entry, ret))
    return out


class TradingAgentsDecisionModel:  # pragma: no cover - requires the fork + network
    """Live wrapper around ``TradingAgentsGraph`` implementing :class:`DecisionModel`.

    Not exercised offline (no fork/API here). It sets the PIT vendor clock, installs
    the reflection gate, calls ``propagate(ticker, date)`` and parses the decision.
    """

    def __init__(
        self,
        store: PointInTimeStore,
        *,
        holding_days: int = 5,
        model_id: str = "tradingagents",
        graph_factory: Callable[[], object] | None = None,
    ) -> None:
        self.store = store
        self.holding_days = holding_days
        self.model_id = model_id
        self._graph_factory = graph_factory
        self._graph = None

    def _ensure_graph(self):
        if self._graph is None:
            if self._graph_factory is None:
                from tradingagents.graph.trading_graph import TradingAgentsGraph  # type: ignore

                self._graph = TradingAgentsGraph()
            else:
                self._graph = self._graph_factory()
            from vts.integration.tradingagents_vendor import register_pit_vendor

            register_pit_vendor(self.store)
        return self._graph

    def prompt_for(self, ticker: str, clock: AsOfClock) -> str:
        return f"tradingagents:{ticker.upper()}@{clock.as_of.date()}"

    def decide(self, ticker: str, clock: AsOfClock) -> Decision:
        from vts.integration.tradingagents_vendor import CURRENT_CLOCK

        graph = self._ensure_graph()
        token = CURRENT_CLOCK.set(clock)
        try:
            # Neutralize the deferred-reflection look-ahead before the pipeline runs.
            graph._resolve_pending_entries = lambda _t: None  # type: ignore[attr-defined]
            _final_state, signal = graph.propagate(ticker, clock.as_of.strftime("%Y-%m-%d"))
        finally:
            CURRENT_CLOCK.reset(token)
        return build_decision(ticker, clock, str(signal), model_id=self.model_id)
