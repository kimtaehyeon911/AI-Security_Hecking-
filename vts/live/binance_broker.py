"""Binance spot broker adapter implementing :class:`~vts.live.broker.BrokerClient`.

Structure mirrors every other adapter in this repo: **pure mappers** (JSON → our
models) carry all the correctness-critical logic and are tested offline with
fixtures; the HTTP class is a thin signed transport. No test here ever needs a
real key or the network (httpx.MockTransport in tests).

Safety posture:

- **Testnet by default.** ``base_url`` defaults to ``https://testnet.binance.vision``;
  routing to production requires explicitly passing :data:`PROD_BASE_URL`. The
  arming token, capital cap and kill switch (vts.live.config / trader) still
  apply on top — this default is one more layer, not a replacement.
- Keys come from ``BINANCE_API_KEY`` / ``BINANCE_API_SECRET`` (never committed;
  see .env.example). Signed requests use the standard HMAC-SHA256 signature over
  the urlencoded query per Binance's API docs.
- Spot "positions" are asset balances mapped to ``{ASSET}{QUOTE}`` symbols.
  Assets with no quote pair are still REPORTED (last_price=0) — an unpriceable
  balance must be visible, not hidden — and they contribute 0 to total_assets,
  which UNDERSTATES the 1% cap denominator: the conservative direction.
"""

from __future__ import annotations

import hashlib
import hmac
import math
import os
import time
from decimal import Decimal
from typing import Callable
from urllib.parse import urlencode

import httpx

from vts.live.broker import Account, BrokerOrder, Position
from vts.risk.limits import VenueRules

TESTNET_BASE_URL = "https://testnet.binance.vision"
PROD_BASE_URL = "https://api.binance.com"

_ACCEPTED_STATUSES = {"NEW", "PARTIALLY_FILLED", "FILLED"}


def _num(value: object) -> float:
    """Parse a Binance numeric string; non-finite/garbage collapses to 0.0.

    Balances/prices feed the capital-cap denominator, where a NaN would silently
    disable the cap (a confirmed Step 6 critical) — 0.0 is the fail-closed value.
    """
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    return f if math.isfinite(f) else 0.0


def _fmt(value: float) -> str:
    """Exact, non-scientific decimal string (Binance rejects '1e-05')."""
    return format(Decimal(str(value)), "f")


def sign_params(params: dict[str, object], secret: str) -> str:
    """HMAC-SHA256 signature (hex) over the urlencoded query, per Binance docs."""
    query = urlencode(params)
    return hmac.new(secret.encode(), query.encode(), hashlib.sha256).hexdigest()


# --------------------------------------------------------------------- mappers
def map_account(
    account_json: dict, prices: dict[str, float], quote: str
) -> tuple[Account, list[Position]]:
    """Map /api/v3/account balances to (Account, positions).

    - ``cash`` = FREE quote balance (deployable now; locked quote counts toward
      total assets but not cash).
    - ``total_assets`` = free+locked quote + Σ (free+locked asset) × price. An
      asset without a ``{ASSET}{QUOTE}`` price contributes 0 — understating the
      cap denominator, the conservative direction.
    """
    q = quote.strip().upper()
    cash = 0.0
    total = 0.0
    positions: list[Position] = []
    for bal in account_json.get("balances", []):
        asset = str(bal.get("asset", "")).strip().upper()
        qty = _num(bal.get("free")) + _num(bal.get("locked"))
        if not asset or qty == 0.0:
            continue
        if asset == q:
            cash = _num(bal.get("free"))
            total += qty
            continue
        symbol = f"{asset}{q}"
        price = _num(prices.get(symbol, 0.0))
        total += qty * price
        positions.append(Position(symbol=symbol, qty=qty, last_price=price))
    return Account(cash=cash, total_assets=total), positions


def map_order_response(row: dict) -> BrokerOrder:
    """Map an /api/v3/order (or openOrders) row to a BrokerOrder.

    The raw Binance status string is preserved so ``BrokerOrder.accepted`` (which
    treats rejected/canceled/expired as not-accepted) stays truthful.
    """
    qty = _num(row.get("origQty"))
    price = _num(row.get("price"))
    return BrokerOrder(
        symbol=str(row.get("symbol", "")).strip().upper(),
        side="buy" if str(row.get("side", "")).upper() == "BUY" else "sell",
        qty=qty if qty > 0 else 1e-12,  # schema requires >0; a 0-qty row is broker garbage
        limit_price=price if price > 0 else None,
        order_id=str(row.get("orderId", "")),
        status=str(row.get("status", "accepted")),
    )


def venue_from_exchange_filters(symbol_info: dict) -> VenueRules:
    """Build per-symbol VenueRules from an /api/v3/exchangeInfo symbol entry.

    PRICE_FILTER.tickSize → flat tick ladder; LOT_SIZE.stepSize → lot;
    NOTIONAL/MIN_NOTIONAL.minNotional → min order value. Raises on a missing
    tick/step — trading a symbol with unknown filters is how orders bounce.
    """
    tick = step = None
    min_notional = 0.0
    for f in symbol_info.get("filters", []):
        ftype = f.get("filterType")
        if ftype == "PRICE_FILTER":
            tick = str(f.get("tickSize"))
        elif ftype == "LOT_SIZE":
            step = str(f.get("stepSize"))
        elif ftype in ("NOTIONAL", "MIN_NOTIONAL"):
            min_notional = _num(f.get("minNotional"))
    if tick is None or step is None:
        raise ValueError(
            f"exchangeInfo for {symbol_info.get('symbol')!r} lacks "
            f"PRICE_FILTER/LOT_SIZE filters"
        )
    # Binance encodes ticks like "0.01000000" — normalize the exact value.
    return VenueRules(
        name=f"binance:{symbol_info.get('symbol', '?')}",
        tick_ladder=((float("inf"), format(Decimal(tick), "f")),),
        lot_size=step,
        min_notional=min_notional,
    )


# ------------------------------------------------------------------ the client
class BinanceBroker:
    """Signed Binance spot client implementing the BrokerClient protocol."""

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
        *,
        base_url: str = TESTNET_BASE_URL,
        quote: str = "USDT",
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        recv_window_ms: int = 5000,
        time_fn: Callable[[], float] = time.time,
    ) -> None:
        self._key = api_key or os.environ.get("BINANCE_API_KEY", "")
        self._secret = api_secret or os.environ.get("BINANCE_API_SECRET", "")
        self._base = base_url.rstrip("/")
        self._quote = quote.strip().upper()
        self._client = client or httpx.Client(timeout=timeout)
        self._recv_window = recv_window_ms
        self._time_fn = time_fn

    @property
    def is_testnet(self) -> bool:
        return self._base == TESTNET_BASE_URL

    # ---------------------------------------------------------------- plumbing
    def _require_keys(self) -> None:
        if not self._key or not self._secret:
            raise RuntimeError(
                "BINANCE_API_KEY / BINANCE_API_SECRET are not set; add them to "
                "your .env (never commit them)."
            )

    def _signed(self, method: str, path: str, params: dict[str, object]) -> dict | list:
        self._require_keys()
        p = dict(params)
        p["timestamp"] = int(self._time_fn() * 1000)
        p["recvWindow"] = self._recv_window
        p["signature"] = sign_params(p, self._secret)
        resp = self._client.request(
            method, f"{self._base}{path}", params=p,
            headers={"X-MBX-APIKEY": self._key},
        )
        resp.raise_for_status()
        return resp.json()

    def _public(self, path: str, params: dict[str, object] | None = None) -> dict | list:
        resp = self._client.get(f"{self._base}{path}", params=params or {})
        resp.raise_for_status()
        return resp.json()

    def _all_prices(self) -> dict[str, float]:
        rows = self._public("/api/v3/ticker/price")
        return {str(r["symbol"]).upper(): _num(r["price"]) for r in rows}

    # ---------------------------------------------------------------- contract
    def get_account(self) -> Account:
        data = self._signed("GET", "/api/v3/account", {})
        account, _ = map_account(data, self._all_prices(), self._quote)
        return account

    def get_positions(self) -> list[Position]:
        data = self._signed("GET", "/api/v3/account", {})
        _, positions = map_account(data, self._all_prices(), self._quote)
        return positions

    def get_open_orders(self) -> list[BrokerOrder]:
        rows = self._signed("GET", "/api/v3/openOrders", {})
        return [map_order_response(r) for r in rows]

    def cancel_all_orders(self) -> int:
        """Binance cancels per symbol: enumerate open orders, cancel each symbol."""
        open_orders = self.get_open_orders()
        symbols = {o.symbol for o in open_orders}
        cancelled = 0
        for sym in sorted(symbols):
            rows = self._signed("DELETE", "/api/v3/openOrders", {"symbol": sym})
            cancelled += len(rows) if isinstance(rows, list) else 1
        return cancelled

    def submit_order(
        self, symbol: str, side: str, qty: float, *, limit_price: float
    ) -> BrokerOrder:
        row = self._signed(
            "POST", "/api/v3/order",
            {
                "symbol": symbol.strip().upper(),
                "side": side.strip().upper(),
                "type": "LIMIT",
                "timeInForce": "GTC",
                "quantity": _fmt(qty),
                "price": _fmt(limit_price),
            },
        )
        order = map_order_response(row)  # preserves the venue's status verbatim
        # A response with no/zero price still carries OUR limit for the audit trail.
        if order.limit_price is None:
            order = order.model_copy(update={"limit_price": limit_price})
        return order

    def close(self) -> None:
        self._client.close()
