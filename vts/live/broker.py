"""Broker interface and an in-memory simulator.

:class:`BrokerClient` is the contract a real adapter (Alpaca, KIS, 키움, Binance)
must satisfy. **No real adapter ships in this package**: money-moving HTTP code
needs its own review and a live smoke test against a paper endpoint before any
production key is involved, so shipping an untested one here would be exactly the
wrong kind of convenience.

Contract notes that exist because getting them wrong loses money:

- ``submit_order`` takes a REQUIRED ``limit_price``. An earlier revision dropped
  it, so orders validated as limits were sent to the venue unpriced (i.e. market)
  with unbounded slippage against the validated notional.
- ``get_open_orders`` exists so the caller can net still-working quantity into its
  position view; without it an unfilled order is re-sent every cycle and exposure
  multiplies.
- Money fields reject nan/inf. ``Account.total_assets`` is the denominator of the
  hardcoded 1% capital cap, and a single NaN there silently removes the cap.
"""

from __future__ import annotations

from typing import Annotated, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

FiniteMoney = Annotated[float, Field(allow_inf_nan=False)]


class Account(BaseModel):
    """Account snapshot as reported by the broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cash: FiniteMoney
    total_assets: FiniteMoney = Field(
        description="Total account equity — the denominator of the 1% capital cap."
    )


class Position(BaseModel):
    """One open position."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    qty: FiniteMoney
    last_price: FiniteMoney = Field(default=0.0, ge=0.0)


class BrokerOrder(BaseModel):
    """An order as accepted (or rejected) by the broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str
    side: Literal["buy", "sell"]
    qty: FiniteMoney = Field(gt=0.0)
    limit_price: FiniteMoney | None = None
    order_id: str = ""
    status: str = Field(
        default="accepted",
        description=(
            "Adapters must NORMALIZE any not-working venue status into the "
            "denylist vocabulary ('rejected'/'canceled'/'expired') — the accepted "
            "check is a denylist and cannot know every venue's spellings."
        ),
    )
    raw_status: str = Field(
        default="", description="The venue's verbatim status string, for the audit trail."
    )

    @property
    def accepted(self) -> bool:
        return self.status.lower() not in {"rejected", "canceled", "cancelled", "expired"}


@runtime_checkable
class BrokerClient(Protocol):
    """The minimum surface the live trader needs."""

    def get_account(self) -> Account: ...

    def get_positions(self) -> list[Position]: ...

    def get_open_orders(self) -> list[BrokerOrder]:
        """Still-working orders, so the caller can avoid duplicating them."""
        ...

    def cancel_all_orders(self) -> int:
        """Cancel every open order; returns how many were cancelled."""
        ...

    def submit_order(
        self, symbol: str, side: str, qty: float, *, limit_price: float
    ) -> BrokerOrder: ...


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
        fill: bool = True,
    ) -> None:
        self._cash = cash
        self._total_assets = cash if total_assets is None else total_assets
        self._positions = {k.strip().upper(): v for k, v in (positions or {}).items()}
        self._prices = {k.strip().upper(): v for k, v in (prices or {}).items()}
        self._open: list[BrokerOrder] = []
        self._pending_cancel = open_orders
        self._fill = fill  # False leaves submitted orders working (unfilled)
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

    def get_open_orders(self) -> list[BrokerOrder]:
        return list(self._open)

    def cancel_all_orders(self) -> int:
        cancelled = self._pending_cancel + len(self._open)
        self._pending_cancel = 0
        self._open.clear()
        return cancelled

    def submit_order(
        self, symbol: str, side: str, qty: float, *, limit_price: float
    ) -> BrokerOrder:
        sym = symbol.strip().upper()
        self._seq += 1
        order = BrokerOrder(symbol=sym, side=side, qty=qty, limit_price=limit_price,
                            order_id=f"sim-{self._seq}")
        self.submitted.append(order)
        if not self._fill:
            self._open.append(order)
            return order
        price = limit_price if limit_price else self._prices.get(sym, 0.0)
        signed = qty if side == "buy" else -qty
        self._positions[sym] = self._positions.get(sym, 0.0) + signed
        self._cash -= signed * price
        if abs(self._positions[sym]) < 1e-9:
            self._positions.pop(sym, None)
        return order
