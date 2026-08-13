"""Full liquidation (전량 청산) — what the kill switch actually does live.

Step 4's kill switch flattens *target weights*; enough in simulation, not on a
real account where flat targets merely mean "stop opening" while the book stays
exposed. Here the switch means what the mandate says: open orders cancelled,
positions closed — and then **verified closed**.

Safety properties, each of which cost a confirmed review finding to learn:

1. **Cancel first, but never gate on it.** A resting buy filling mid-liquidation
   re-opens exposure, so cancellation comes first; if cancelling *fails* the
   closes still run — the closes are the higher-value action.
2. **Verify, don't assume.** After submitting closes the positions are re-fetched;
   anything still non-zero is reported as ``residual``. ``complete`` is derived
   from observed flatness, not from "no exception was raised" — a broker that
   returns a rejected-status order instead of raising must not read as success.
3. **Never cross zero.** Each close is clamped to the position's own sign and
   size, so a repeated liquidation can never flip a long into a short.
4. **Scope.** ``symbols=None`` closes everything; the live trader passes its own
   sleeve symbols so an emergency stop cannot liquidate an operator's unrelated
   holdings.
5. **Dry run is never "complete"** — a plan is not an execution.
"""

from __future__ import annotations

from collections.abc import Iterable

from pydantic import BaseModel, ConfigDict, Field

from vts.live.broker import BrokerClient


class LiquidationReport(BaseModel):
    """Outcome of one liquidation attempt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reason: str
    dry_run: bool
    cancelled_orders: int = 0
    closed: dict[str, float] = Field(
        default_factory=dict, description="symbol -> qty actually submitted for close"
    )
    would_close: dict[str, float] = Field(
        default_factory=dict, description="dry-run plan only; nothing was sent"
    )
    residual: dict[str, float] = Field(
        default_factory=dict,
        description="symbol -> qty STILL held after closing (verified by re-fetch)",
    )
    failures: dict[str, str] = Field(
        default_factory=dict, description="symbol -> error message"
    )

    @property
    def complete(self) -> bool:
        """True only when the account was OBSERVED flat and nothing failed.

        A dry run is never complete: it changed nothing.
        """
        return not self.dry_run and not self.failures and not self.residual


def liquidate_all(
    broker: BrokerClient,
    *,
    reason: str,
    dry_run: bool,
    symbols: Iterable[str] | None = None,
    verify_attempts: int = 2,
) -> LiquidationReport:
    """Cancel open orders, close every (in-scope) position, then verify flat.

    ``dry_run`` is a required keyword: the caller must state intent explicitly on
    a function that can move an entire book.
    """
    scope = {s.strip().upper() for s in symbols} if symbols is not None else None

    def in_scope(sym: str) -> bool:
        return scope is None or sym.strip().upper() in scope

    if dry_run:
        try:
            positions = broker.get_positions()
        except Exception as exc:
            return LiquidationReport(reason=reason, dry_run=True,
                                     failures={"__get_positions__": str(exc)})
        return LiquidationReport(
            reason=reason, dry_run=True,
            would_close={p.symbol: abs(p.qty)
                         for p in positions if p.qty != 0.0 and in_scope(p.symbol)},
        )

    failures: dict[str, str] = {}
    cancelled = 0
    try:
        cancelled = broker.cancel_all_orders()
    except Exception as exc:
        # Record and CONTINUE — closing the book matters more than cancelling.
        failures["__cancel_all_orders__"] = str(exc)

    closed: dict[str, float] = {}
    for attempt in range(max(1, verify_attempts)):
        try:
            positions = broker.get_positions()
        except Exception as exc:
            failures["__get_positions__"] = str(exc)
            return LiquidationReport(reason=reason, dry_run=False,
                                     cancelled_orders=cancelled, closed=closed,
                                     failures=failures)

        outstanding = [p for p in positions if p.qty != 0.0 and in_scope(p.symbol)]
        if not outstanding:
            break
        if attempt and not closed:
            break  # nothing we can do is changing the book; stop hammering it

        for pos in outstanding:
            side = "sell" if pos.qty > 0 else "buy"   # a short is closed by buying
            qty = abs(pos.qty)                         # clamped: never crosses zero
            price = pos.last_price
            try:
                order = broker.submit_order(pos.symbol, side, qty, limit_price=price)
                if not order.accepted:
                    failures[pos.symbol] = f"broker returned status={order.status!r}"
                    continue
                closed[pos.symbol] = closed.get(pos.symbol, 0.0) + qty
            except Exception as exc:  # one bad symbol must not strand the rest
                failures[pos.symbol] = str(exc)

    # Verify: re-fetch and report anything still held in scope.
    residual: dict[str, float] = {}
    try:
        for pos in broker.get_positions():
            if pos.qty != 0.0 and in_scope(pos.symbol):
                residual[pos.symbol] = pos.qty
    except Exception as exc:
        failures["__verify__"] = str(exc)

    return LiquidationReport(
        reason=reason, dry_run=False, cancelled_orders=cancelled,
        closed=closed, residual=residual, failures=failures,
    )
