"""Full liquidation (전량 청산) — what the kill switch actually does live.

Step 4's kill switch flattens *target weights*; that is enough in simulation but
not on a real account, where flat targets merely mean "stop opening" while the
existing book stays exposed. Here the switch means what the mandate says: every
open order cancelled, every position closed.

Order of operations is the safety-critical part:

1. **Cancel open orders first.** A resting buy that fills mid-liquidation would
   re-open exposure behind the liquidator.
2. **Then close every position**, long or short (a short is closed by buying).
3. Report per-symbol outcomes; a failure on one symbol must not abandon the rest.

The function is idempotent: with nothing open and nothing held it is a no-op, so
it is safe to call on every cycle and safe to retry after a partial failure.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from vts.live.broker import BrokerClient


class LiquidationReport(BaseModel):
    """Outcome of one liquidation attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str
    dry_run: bool
    cancelled_orders: int = 0
    closed: dict[str, float] = Field(
        default_factory=dict, description="symbol -> qty closed (absolute)"
    )
    failures: dict[str, str] = Field(
        default_factory=dict, description="symbol -> error message"
    )

    @property
    def complete(self) -> bool:
        """True when nothing failed — the account is flat as far as we can tell."""
        return not self.failures


def liquidate_all(
    broker: BrokerClient, *, reason: str, dry_run: bool = True
) -> LiquidationReport:
    """Cancel all open orders, then close every position.

    ``dry_run=True`` (the default) reports exactly what *would* be sent without
    touching the account.
    """
    if dry_run:
        positions = broker.get_positions()
        return LiquidationReport(
            reason=reason, dry_run=True, cancelled_orders=0,
            closed={p.symbol: abs(p.qty) for p in positions if p.qty != 0.0},
        )

    cancelled = 0
    try:
        cancelled = broker.cancel_all_orders()
    except Exception as exc:  # a cancel failure must not stop us closing positions
        return LiquidationReport(
            reason=reason, dry_run=False,
            failures={"__cancel_all_orders__": str(exc)},
        )

    closed: dict[str, float] = {}
    failures: dict[str, str] = {}
    for pos in broker.get_positions():
        if pos.qty == 0.0:
            continue
        side = "sell" if pos.qty > 0 else "buy"   # a short is closed by buying
        try:
            broker.submit_order(pos.symbol, side, abs(pos.qty))
            closed[pos.symbol] = abs(pos.qty)
        except Exception as exc:  # keep going: one bad symbol must not strand the book
            failures[pos.symbol] = str(exc)

    return LiquidationReport(
        reason=reason, dry_run=False, cancelled_orders=cancelled,
        closed=closed, failures=failures,
    )
