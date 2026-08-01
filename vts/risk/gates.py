"""The decision gate: schema violations and weak decisions are forced to Hold.

Step 4 mandate: *"에이전트 출력이 스키마를 위반하거나 확신도 낮으면 → 무조건
Hold."* The gate is a pure function from (aggregated decision, limits) to a
:class:`GatedDecision` carrying the effective rating and an audit trail of why it
was (or wasn't) overridden. Callers that fail to even *parse* agent output never
reach the gate — they call :func:`hold_fallback` and get the same auditable Hold.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from vts.decision import AggregatedDecision, Rating
from vts.risk.limits import RiskLimits


class GatedDecision(BaseModel):
    """An agent decision after the deterministic gate."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    ticker: str
    proposed: Rating              # what the agents wanted
    effective: Rating             # what the risk layer allows
    forced_hold: bool
    reasons: tuple[str, ...]      # audit trail; empty when passed through


def hold_fallback(ticker: str, reason: str) -> GatedDecision:
    """The Hold every unparseable / schema-violating agent output collapses to."""
    return GatedDecision(
        ticker=ticker,
        proposed=Rating.HOLD,
        effective=Rating.HOLD,
        forced_hold=True,
        reasons=(f"schema_violation: {reason}",),
    )


def gate_decision(agg: AggregatedDecision, limits: RiskLimits) -> GatedDecision:
    """Force Hold when the decision is weak; pass it through otherwise.

    Weakness tests (all deterministic, all from frozen limits):
    - mean confidence below ``min_confidence``
    - vote agreement below ``min_agreement``
    - ordinal rating dispersion above ``max_dispersion``
    A vote tie already collapsed to Hold upstream; it is recorded here for audit.
    """
    reasons: list[str] = []
    if agg.mean_confidence < limits.min_confidence:
        reasons.append(
            f"low_confidence: {agg.mean_confidence:.2f} < {limits.min_confidence:.2f}"
        )
    if agg.agreement < limits.min_agreement:
        reasons.append(f"low_agreement: {agg.agreement:.2f} < {limits.min_agreement:.2f}")
    if agg.dispersion > limits.max_dispersion:
        reasons.append(f"high_dispersion: {agg.dispersion:.2f} > {limits.max_dispersion:.2f}")
    if agg.tie_broken_to_hold:
        reasons.append("vote_tie")
    # A single sample makes agreement (always 1.0) and dispersion (always 0.0)
    # vacuous, so those gates cannot fire. An operator who relies on them can set
    # hold_on_single_sample so an un-assessable single-sample decision is a
    # conservative Hold rather than a free pass. Off by default (deterministic
    # research runs legitimately use n=1).
    if limits.hold_on_single_sample and agg.n_samples < 2:
        reasons.append(f"single_sample: n={agg.n_samples} cannot assess agreement/dispersion")

    forced = bool(reasons) and agg.rating != Rating.HOLD
    return GatedDecision(
        ticker=agg.ticker,
        proposed=agg.rating,
        effective=Rating.HOLD if reasons else agg.rating,
        forced_hold=forced,
        reasons=tuple(reasons),
    )
