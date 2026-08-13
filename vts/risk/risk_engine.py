"""RiskEngine: the deterministic pipeline between agent output and any execution.

    aggregated decisions
        → gate (schema/confidence/agreement/dispersion → Hold)
        → target weights (rating → signed weight)
        → clamps (per-symbol ceiling, gross ceiling)
        → halt check (daily loss latch, env kill switch) → flat when halted

Limits enter at construction time from frozen :class:`RiskLimits`; nothing on the
per-decision path can change them. The same engine object serves the backtest
(Step 2), paper trading (Step 5) and live (Step 6), so the risk code path is
identical everywhere — the mandate's reason for keeping it out of the agents.
"""

from __future__ import annotations

from dataclasses import dataclass

from vts.decision import AggregatedDecision, rating_to_signed_weight
from vts.risk.gates import GatedDecision, gate_decision
from vts.risk.killswitch import HaltState
from vts.risk.limits import RiskLimits


@dataclass(frozen=True, slots=True)
class RiskVerdict:
    """One batch's output: final weights plus the full audit trail."""

    weights: dict[str, float]
    gated: dict[str, GatedDecision]
    halted: bool
    halt_reasons: tuple[str, ...]


class RiskEngine:
    """Owns the halt latch and applies gate + clamps to every decision batch."""

    def __init__(self, limits: RiskLimits | None = None, *, long_only: bool = True) -> None:
        self.limits = limits or RiskLimits()
        self.long_only = long_only
        self.halt = HaltState(
            daily_loss_limit=self.limits.daily_loss_limit,
            max_drawdown_limit=self.limits.max_drawdown_limit,
        )

    # ------------------------------------------------------------------ inputs
    def observe_daily_return(self, daily_return: float) -> bool:
        """Feed one SINGLE-DAY mark-to-market return; returns halted state."""
        return self.halt.observe_daily_return(daily_return)

    def observe_equity(self, equity: float) -> bool:
        """Feed running equity for the cumulative drawdown-from-peak stop."""
        return self.halt.observe_equity(equity)

    # ------------------------------------------------------------------- apply
    def apply(self, aggs: dict[str, AggregatedDecision]) -> RiskVerdict:
        """Gate, size and clamp one decision batch (flat if halted/killed)."""
        self.halt.check_kill_switch()

        gated = {t: gate_decision(a, self.limits) for t, a in aggs.items()}

        if self.halt.halted:
            return RiskVerdict(
                weights={t: 0.0 for t in aggs},
                gated=gated,
                halted=True,
                halt_reasons=tuple(self.halt.halt_reasons),
            )

        weights = {t: rating_to_signed_weight(g.effective) for t, g in gated.items()}
        if self.long_only:
            weights = {t: max(0.0, w) for t, w in weights.items()}

        # Per-symbol ceiling first, then gross ceiling on what remains.
        cap = self.limits.max_weight_per_symbol
        weights = {t: max(min(w, cap), -cap) for t, w in weights.items()}
        gross = sum(abs(w) for w in weights.values())
        if gross > self.limits.max_gross_exposure and gross > 0:
            scale = self.limits.max_gross_exposure / gross
            weights = {t: w * scale for t, w in weights.items()}

        return RiskVerdict(
            weights=weights,
            gated=gated,
            halted=False,
            halt_reasons=tuple(self.halt.halt_reasons),
        )
