"""Order pre-validation: no order reaches a broker without passing every check.

Step 4 mandate: *"모든 주문은 사전 검증(잔고, 호가단위, 최소주문금액) 통과해야
전송."* :func:`validate_order` collects EVERY violation (not just the first) so a
rejected order's audit line shows the full picture. Tick arithmetic uses Decimal
on stringified inputs — float modulo would false-reject valid prices.
"""

from __future__ import annotations

from decimal import Decimal
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from vts.risk.limits import VenueRules


class Order(BaseModel):
    """One candidate order, as it would be sent to a broker."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    symbol: str = Field(min_length=1)
    side: Literal["buy", "sell"]
    qty: float = Field(gt=0.0)
    limit_price: float = Field(gt=0.0)

    @property
    def notional(self) -> float:
        return self.qty * self.limit_price


class AccountState(BaseModel):
    """Cash and positions the validation checks against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    cash: float = Field(ge=0.0)
    positions: dict[str, float] = Field(default_factory=dict)  # symbol -> qty held

    @field_validator("positions")
    @classmethod
    def _normalize_keys(cls, v: dict[str, float]) -> dict[str, float]:
        """Upper-case position keys so lookup (also upper-cased) is symmetric.

        Without this, a broker feed with lowercase keys ('brk.b') would strand
        the position — ``position()`` upper-cases the query and would miss it.
        """
        out: dict[str, float] = {}
        for k, qty in v.items():
            key = k.strip().upper()
            if key in out:
                raise ValueError(f"duplicate position symbol after normalization: {key}")
            out[key] = qty
        return out

    def position(self, symbol: str) -> float:
        return self.positions.get(symbol.strip().upper(), 0.0)


class OrderRejection(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    order: Order
    violations: tuple[str, ...]


def _off_tick(price: float, tick: Decimal) -> bool:
    """Exact tick check via Decimal on the string form (no float-modulo noise)."""
    return Decimal(str(price)) % tick != 0


def validate_order(
    order: Order,
    account: AccountState,
    venue: VenueRules,
    *,
    cash_buffer: float = 0.0,
) -> OrderRejection | None:
    """Return ``None`` when the order passes every check, else the full rejection.

    Checks (all mandated):
    - 잔고: buys must fit in ``cash - cash_buffer``; sells must not exceed the held
      position (no naked shorts through this layer). NOTE: the default
      ``cash_buffer=0.0`` is fee-blind — a buy may consume 100% of cash leaving
      nothing for commission/slippage. Live/paper callers MUST pass a
      ``cash_buffer`` covering estimated fees (e.g. from ``CostModel``) so a
      certified order can actually settle.
    - 호가단위: the limit price must sit on the venue's tick for that price band.
    - 최소주문금액 / lot: notional >= venue.min_notional; qty a positive multiple
      of ``lot_size`` (and an integer when lot_size is 1 — fractional shares are
      not assumed unless a venue explicitly models them).
    """
    v: list[str] = []

    if order.side == "buy":
        available = account.cash - cash_buffer
        if order.notional > available:
            v.append(
                f"insufficient_cash: need {order.notional:.2f}, "
                f"available {available:.2f}"
            )
    else:
        held = account.position(order.symbol)
        if order.qty > held:
            v.append(f"insufficient_position: selling {order.qty:g}, hold {held:g}")

    tick = venue.tick_for(order.limit_price)
    if _off_tick(order.limit_price, tick):
        v.append(f"off_tick: price {order.limit_price} not on tick {tick} ({venue.name})")

    if order.notional < venue.min_notional:
        v.append(f"below_min_notional: {order.notional:.2f} < {venue.min_notional:.2f}")

    qty_dec = Decimal(str(order.qty))
    if qty_dec % Decimal(venue.lot_size) != 0:
        v.append(f"lot_size: qty {order.qty:g} not a multiple of {venue.lot_size}")

    return None if not v else OrderRejection(order=order, violations=tuple(v))
