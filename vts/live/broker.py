"""Broker interface and an in-memory simulator.

:class:`BrokerClient` is the contract a real adapter (Alpaca, KIS, 키움, Binance)
must satisfy. **No real adapter ships in this package**: money-moving HTTP code
needs its own review and a live smoke test against a paper endpoint before any
production key is involved, so shipping an untested one here would be exactly the
wrong kind of convenience.

:class:`SimulatedBroker` implements the same contract in memory and is what the
tests — and every dry-run session — trade against.
"""

from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field


class Account(BaseModel):
    """Account snapshot as reported by the broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cash: float
    total_assets: float = Field(
        description="Total account equity — the denominator of the 1% capital cap."
    )


class Position(BaseModel):
    """One open position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    qty: float
    last_price: float = Field(default=0.0, ge=0.0)


class BrokerOrder(BaseModel):
    """An order as accepted by the broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0.0)
    order_id: str = ""


@runtime_checkable
class BrokerClient(Protocol):
    """The minimum surface the live trader needs.

    ``cancel_all_orders`` must be callable before liquidation so a resting buy
    cannot fill while positions are being closed.
    """

    def get_account(self) -> Account: ...

    def get_positions(self) -> list[Position]: ...

    def cancel_all_orders(self) -> int:
        """Cancel every open order; returns how many were cancelled."""
        ...

    def submit_order(self, symbol: str, side: str, qty: float) -> BrokerOrder: ...


class SimulatedBroker:
    """In-memory :class:`BrokerClient` for dry runs and tests."""

    def __init__(
        self,
        *,
        cash: float = 1_000_000.0,
        total_assets: float | None = None,
        positions: dict[str, float] | None = None,
        prices: dict[str, float] | None = None,
        open_orders: int = 0,
    ) -> None:
        self._cash = cash
        self._total_assets = cash if total_assets is None else total_assets
        self._positions = {k.strip().upper(): v for k, v in (positions or {}).items()}
        self._prices = {k.strip().upper(): v for k, v in (prices or {}).items()}
        self._open_orders = open_orders
        self.submitted: list[BrokerOrder] = []
        self._seq = 0

    def get_account(self) -> Account:
        return Account(cash=self._cash, total_assets=self._total_assets)

    def get_positions(self) -> list[Position]:
        return [
            Position(symbol=s, qty=q, last_price=self._prices.get(s, 0.0))
            for s, q in self._positions.items()
            if q != 0.0
        ]

    def cancel_all_orders(self) -> int:
        cancelled, self._open_orders = self._open_orders, 0
        return cancelled

    def submit_order(self, symbol: str, side: str, qty: float) -> BrokerOrder:
        sym = symbol.strip().upper()
        self._seq += 1
        order = BrokerOrder(symbol=sym, side=side, qty=qty, order_id=f"sim-{self._seq}")
        self.submitted.append(order)
        # Fill immediately at the last known price.
        price = self._prices.get(sym, 0.0)
        signed = qty if side == "buy" else -qty
        self._positions[sym] = self._positions.get(sym, 0.0) + signed
        self._cash -= signed * price
        if abs(self._positions[sym]) < 1e-9:
            self._positions.pop(sym, None)
        return order
