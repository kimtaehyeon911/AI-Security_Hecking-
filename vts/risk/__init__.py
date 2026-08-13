"""Deterministic risk layer — separate from, and downstream of, the agents.

Design constraint (Step 4 mandate, 금지사항): **no limit in this package is ever
decided by an LLM.** Limits are frozen configuration constants loaded from code
or environment at startup; agent output enters this layer only as a *candidate
decision* to be gated, clamped, halted, or rejected. Nothing an agent emits can
loosen a limit — the data flow is one-way.
"""

from __future__ import annotations

from vts.risk.gates import GatedDecision, gate_decision
from vts.risk.killswitch import HaltState, kill_switch_active
from vts.risk.limits import RiskLimits, VenueRules
from vts.risk.orders import AccountState, Order, OrderRejection, validate_order
from vts.risk.risk_engine import RiskEngine

__all__ = [
    "AccountState",
    "GatedDecision",
    "HaltState",
    "Order",
    "OrderRejection",
    "RiskEngine",
    "RiskLimits",
    "VenueRules",
    "gate_decision",
    "kill_switch_active",
    "validate_order",
]
