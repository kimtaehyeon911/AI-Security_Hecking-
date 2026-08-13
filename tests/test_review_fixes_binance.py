"""Regression tests for the Binance-adapter adversarial-review findings (round 7)."""

from __future__ import annotations

from datetime import timedelta

import httpx
import pytest
from conftest import utc

from vts.live.binance_broker import (
    BinanceBroker,
    map_account,
    map_open_order,
    map_order_response,
)
from vts.risk.limits import VenueRules
from vts.risk.orders import AccountState, Order, validate_order
from vts.sources.binance_data import BinanceSource

_DAY_MS = 86_400_000


def _kline(open_dt, o, h, l, c, v):
    open_ms = int(open_dt.timestamp() * 1000)
    return [open_ms, str(o), str(h), str(l), str(c), str(v), open_ms + _DAY_MS - 1,
            "0", 100, "0", "0", "0"]


# --- HIGH: future end must not admit the still-forming candle -----------------
def test_forming_candle_dropped_even_with_future_end():
    # Wall clock: Jan-3 14:00. The Jan-3 candle (closes Jan-4 00:00) is FORMING.
    rows = [
        _kline(utc(2024, 1, 2), 100, 110, 95, 105, 10),   # closed (Jan-3 00:00)
        _kline(utc(2024, 1, 3), 105, 120, 100, 118, 10),  # forming
    ]
    src = BinanceSource(
        client=httpx.Client(transport=httpx.MockTransport(
            lambda r: httpx.Response(200, json=rows))),
        now_fn=lambda: utc(2024, 1, 3, 14),
    )
    # Caller passes a FUTURE end ("through tomorrow") — the forming bar must
    # still be dropped because the knowledge ceiling clamps to the wall clock.
    bars = src.fetch_ohlcv("BTCUSDT", utc(2024, 1, 1), utc(2024, 1, 5))
    assert [b.event_time for b in bars] == [utc(2024, 1, 3)]


# --- LOW: leading-edge bar (closing exactly at start) is fetched --------------
def test_leading_edge_bar_included():
    # Bar opening Dec-31 closes exactly at Jan-1 00:00 == start.
    all_rows = [_kline(utc(2023, 12, 31) + timedelta(days=i), 100 + i, 101 + i,
                       99 + i, 100 + i, 10) for i in range(4)]

    def handler(request: httpx.Request) -> httpx.Response:
        start_ms = int(request.url.params["startTime"])
        return httpx.Response(200, json=[r for r in all_rows if r[0] >= start_ms])

    src = BinanceSource(
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        now_fn=lambda: utc(2024, 2, 1),
    )
    bars = src.fetch_ohlcv("BTCUSDT", utc(2024, 1, 1), utc(2024, 1, 3))
    assert bars[0].event_time == utc(2024, 1, 1)   # was silently missing before


# --- HIGH: corrupt balances refuse the snapshot (no vanished positions) -------
def test_corrupt_balance_raises_instead_of_vanishing():
    bad = {"balances": [{"asset": "BTC", "free": "not-a-number", "locked": "0"}]}
    with pytest.raises(ValueError, match="BTC free balance"):
        map_account(bad, {}, "USDT")
    nan = {"balances": [{"asset": "BTC", "free": "nan", "locked": "0"}]}
    with pytest.raises(ValueError, match="non-finite"):
        map_account(nan, {}, "USDT")


def test_duplicate_quote_rows_accumulate_cash():
    dup = {"balances": [
        {"asset": "USDT", "free": "100", "locked": "0"},
        {"asset": "USDT", "free": "50", "locked": "0"},
    ]}
    account, _ = map_account(dup, {}, "USDT")
    assert account.cash == 150.0                    # += not overwrite


# --- HIGH: open-order netting uses REMAINING qty ------------------------------
def test_open_order_nets_executed_qty():
    row = {"symbol": "BTCUSDT", "side": "BUY", "origQty": "1.0",
           "executedQty": "0.6", "price": "100", "status": "PARTIALLY_FILLED"}
    order = map_open_order(row)
    assert order is not None and order.qty == pytest.approx(0.4)  # not 1.0

    filled = {"symbol": "BTCUSDT", "side": "BUY", "origQty": "1.0",
              "executedQty": "1.0", "price": "100", "status": "FILLED"}
    assert map_open_order(filled) is None            # nothing left to net


def test_zero_qty_response_uses_fallback_never_epsilon():
    row = {"symbol": "BTCUSDT", "side": "BUY", "origQty": "0", "price": "100",
           "status": "NEW"}
    assert map_order_response(row) is None                       # no fabricated 1e-12
    order = map_order_response(row, fallback_qty=0.5)
    assert order is not None and order.qty == 0.5                # our submitted qty


# --- LOW: unknown statuses never read as accepted -----------------------------
def test_unknown_binance_status_normalized_to_rejected():
    for raw in ("EXPIRED_IN_MATCH", "PENDING_CANCEL", "SOME_FUTURE_STATUS"):
        order = map_order_response({"symbol": "BTCUSDT", "side": "BUY",
                                    "origQty": "1", "price": "1", "status": raw})
        assert order is not None
        assert order.accepted is False
        assert order.raw_status == raw               # verbatim kept for audit


# --- MED: cancel sweep is best-effort across symbols --------------------------
def test_cancel_all_attempts_every_symbol_before_raising():
    deletes = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.method == "GET" and request.url.path == "/api/v3/openOrders":
            return httpx.Response(200, json=[
                {"symbol": "AAAUSDT", "side": "BUY", "origQty": "1",
                 "executedQty": "0", "price": "1", "status": "NEW"},
                {"symbol": "BBBUSDT", "side": "BUY", "origQty": "1",
                 "executedQty": "0", "price": "1", "status": "NEW"},
            ])
        if request.method == "DELETE":
            sym = request.url.params["symbol"]
            deletes.append(sym)
            if sym == "AAAUSDT":
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json=[{}])
        raise AssertionError(request.url.path)

    broker = BinanceBroker(api_key="k", api_secret="s",
                           client=httpx.Client(transport=httpx.MockTransport(handler)),
                           time_fn=lambda: 1700000000.0)
    with pytest.raises(RuntimeError, match="AAAUSDT"):
        broker.cancel_all_orders()
    assert deletes == ["AAAUSDT", "BBBUSDT"]         # BBB attempted despite AAA failing


# --- LOW: one snapshot serves both account and positions ----------------------
def test_account_snapshot_is_single_signed_call():
    account_calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/account":
            account_calls.append(1)
            return httpx.Response(200, json={"balances": [
                {"asset": "USDT", "free": "100", "locked": "0"},
                {"asset": "BTC", "free": "0.01", "locked": "0"},
            ]})
        if request.url.path == "/api/v3/ticker/price":
            return httpx.Response(200, json=[{"symbol": "BTCUSDT", "price": "100000"}])
        raise AssertionError(request.url.path)

    broker = BinanceBroker(api_key="k", api_secret="s",
                           client=httpx.Client(transport=httpx.MockTransport(handler)),
                           time_fn=lambda: 1700000000.0)
    account, positions = broker.account_snapshot()
    assert len(account_calls) == 1                   # one consistent view
    assert account.cash == 100.0 and positions[0].symbol == "BTCUSDT"


# --- HIGH/MED: unit-space deltas and quantized exits --------------------------
def test_delta_qty_immune_to_float_drift():
    venue = VenueRules(name="c", lot_size="0.1")
    drifted_held = 0.1 + 0.1 + 0.1                   # 0.30000000000000004
    # Target is the same 3 lots: the trade delta must be exactly zero, not -1 lot.
    assert venue.delta_qty(0.3, drifted_held) == 0.0
    # And a genuine one-lot increase survives the drift.
    assert venue.delta_qty(0.4, drifted_held) == pytest.approx(0.1)


def test_quantizer_output_always_passes_validator():
    venue = VenueRules(name="c", lot_size="0.00001", min_notional=0.0)
    acct = AccountState(cash=1e12)
    for raw in (0.000157, 1234.5678, 0.30000000000000004, 7.000000000000001):
        q = venue.quantize_qty(raw)
        if q <= 0:
            continue
        order = Order(symbol="BTCUSDT", side="buy", qty=q, limit_price=100.0)
        assert validate_order(order, acct, venue) is None, f"self-rejected {raw} -> {q}"
