"""Simulated broker for paper trading.

Fills orders against :class:`~vts.paper.state.PaperState` at a supplied price,
charging the same :class:`~vts.backtest.costs.CostModel` the backtest uses. The
``dry_run`` guard is the safety spine of the plan's priority order: this broker
only ever simulates. A ``dry_run=False`` construction raises
:class:`LiveTradingNotEnabled` — real-money routing is Step 6, gated separately.
"""

from __future__ import annotations

from dataclasses import dataclass

from vts.backtest.costs import CostModel
from vts.paper.state import PaperState
from vts.risk.orders import Order


class LiveTradingNotEnabled(RuntimeError):
    """Raised when a non-dry-run broker is requested before Step 6 enablement."""


@dataclass(frozen=True, slots=True)
class FillResult:
    order: Order
    fill_price: float
    cost: float
    cash_after: float


class PaperBroker:
    """Executes validated orders against paper state (simulation only)."""

    def __init__(self, cost_model: CostModel | None = None, *, dry_run: bool = True) -> None:
        if not dry_run:
            raise LiveTradingNotEnabled(
                "PaperBroker is simulation-only; live routing is Step 6 and must be "
                "enabled explicitly with the kill switch and capital cap in place."
            )
        self.cost_model = cost_model or CostModel()
        self.dry_run = dry_run

    def execute(
        self, state: PaperState, order: Order, fill_price: float, dollar_adv: float
    ) -> FillResult:
        """Apply a fill to ``state``. Assumes the order already passed validation.

        Fills at ``fill_price`` (the caller's execution price) and charges cost on
        that notional. Buys reduce cash by notional+cost; sells increase cash by
        notional-cost. Positions are updated by signed quantity.
        """
        notional = order.qty * fill_price
        cost = self.cost_model.cost(notional, dollar_adv, order.side).total
        sym = order.symbol.strip().upper()
        held = state.positions.get(sym, 0.0)
        if order.side == "buy":
            state.cash -= notional + cost
            state.positions[sym] = held + order.qty
        else:
            state.cash += notional - cost
            state.positions[sym] = held - order.qty
        if abs(state.positions[sym]) < 1e-9:
            state.positions.pop(sym, None)
        return FillResult(order=order, fill_price=fill_price, cost=cost, cash_after=state.cash)
