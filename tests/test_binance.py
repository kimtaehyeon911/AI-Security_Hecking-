"""Binance adapters: kline PIT mapping, signing, account mapping, venue filters,
fractional lots, and the signed client against a mock transport (no network, no keys)."""

from __future__ import annotations

import json
from datetime import timedelta

import httpx
import pytest
from conftest import utc

from vts.live.binance_broker import (
    PROD_BASE_URL,
    TESTNET_BASE_URL,
    BinanceBroker,
    map_account,
    map_order_response,
    sign_params,
    venue_from_exchange_filters,
)
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore
from vts.risk.limits import BINANCE_SPOT_DEFAULT, VenueRules
from vts.risk.orders import AccountState, Order, validate_order
from vts.sources.binance_data import BinanceSource, map_klines

_DAY_MS = 86_400_000


def _kline(open_dt, o, h, l, c, v):
    open_ms = int(open_dt.timestamp() * 1000)
    return [open_ms, str(o), str(h), str(l), str(c), str(v), open_ms + _DAY_MS - 1,
            "0", 100, "0", "0", "0"]


# --- kline mapping -------------------------------------------------------------
def test_map_klines_stamps_bar_at_close_boundary():
    rows = [_kline(utc(2024, 1, 2), 100, 110, 95, 105, 1234.5)]
    bars = map_klines("btcusdt", rows, as_of=utc(2024, 1, 10))
    assert len(bars) == 1
    b = bars[0]
    assert b.symbol == "BTCUSDT"
    assert b.event_time == utc(2024, 1, 3)          # closeTime+1ms = next UTC midnight
    assert b.knowledge_time == b.event_time          # knowable exactly at close
    assert (b.open, b.high, b.low, b.close, b.volume) == (100, 110, 95, 105, 1234.5)


def test_map_klines_drops_still_forming_candle():
    rows = [
        _kline(utc(2024, 1, 2), 100, 110, 95, 105, 10),
        _kline(utc(2024, 1, 3), 105, 120, 100, 118, 10),   # closes at Jan 4 00:00
    ]
    bars = map_klines("BTCUSDT", rows, as_of=utc(2024, 1, 3, 12))  # mid-candle
    assert [b.event_time for b in bars] == [utc(2024, 1, 3)]        # forming bar dropped


def test_map_klines_skips_garbage_rows():
    rows = [
        _kline(utc(2024, 1, 2), 100, 110, 95, 105, 10),
        _kline(utc(2024, 1, 3), "nan", 1, 1, 1, 10),       # non-finite
        _kline(utc(2024, 1, 4), 0, 0, 0, 0, 10),           # zero close
        [123],                                              # short row
    ]
    bars = map_klines("BTCUSDT", rows, as_of=utc(2024, 2, 1))
    assert len(bars) == 1


# --- BinanceSource over a mock transport ---------------------------------------
def test_fetch_ohlcv_paginates_and_dedupes():
    pages = [
        [_kline(utc(2024, 1, 1) + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100 + i, 10)
         for i in range(0, 3)],
        [_kline(utc(2024, 1, 1) + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100 + i, 10)
         for i in range(2, 5)],  # overlapping row 2 -> dedupe must handle it
    ]
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(dict(request.url.params))
        idx = min(len(calls) - 1, len(pages) - 1)
        return httpx.Response(200, json=pages[idx])

    src = BinanceSource(client=httpx.Client(transport=httpx.MockTransport(handler)))
    # First page returns exactly 3 rows < limit? Force pagination by patching limit.
    src._MAX_LIMIT = 3
    bars = src.fetch_ohlcv("BTCUSDT", utc(2024, 1, 1), utc(2024, 1, 10))
    assert calls[0]["symbol"] == "BTCUSDT"
    times = [b.event_time for b in bars]
    assert times == sorted(set(times))               # deduped, ordered
    assert len(bars) == 5


def test_source_returns_honest_empties():
    src = BinanceSource(client=httpx.Client(transport=httpx.MockTransport(
        lambda r: httpx.Response(200, json=[]))))
    assert src.fetch_corporate_actions("BTCUSDT", utc(2024, 1, 1), utc(2024, 2, 1)) == []
    assert src.fetch_news("BTCUSDT", utc(2024, 1, 1), utc(2024, 2, 1)) == []
    assert src.fetch_fundamentals("BTCUSDT") == []


# --- signing: the exact vector from Binance's public API docs -------------------
def test_hmac_signature_matches_binance_docs_vector():
    secret = "NhqPtmdSJYdKjVHjA7PZj4Mge3R5YNiP1e3UZjInClVN65XAbvqqM6A7H5fATj0j"
    params = {
        "symbol": "LTCBTC", "side": "BUY", "type": "LIMIT", "timeInForce": "GTC",
        "quantity": 1, "price": "0.1", "recvWindow": 5000, "timestamp": 1499827319559,
    }
    assert sign_params(params, secret) == (
        "c8db56825ae71d6d79447849e617115f4a920fa2acdcab2b053c4b2838bd6b71"
    )


# --- account mapping ------------------------------------------------------------
def test_map_account_cash_positions_and_conservative_total():
    account_json = {"balances": [
        {"asset": "USDT", "free": "1000.0", "locked": "50.0"},
        {"asset": "BTC", "free": "0.5", "locked": "0.0"},
        {"asset": "XYZ", "free": "10.0", "locked": "0.0"},   # no XYZUSDT price
        {"asset": "ETH", "free": "0", "locked": "0"},        # zero -> skipped
    ]}
    prices = {"BTCUSDT": 100_000.0}
    account, positions = map_account(account_json, prices, "USDT")
    assert account.cash == 1000.0                            # free quote only
    assert account.total_assets == pytest.approx(1050.0 + 0.5 * 100_000.0)  # XYZ adds 0
    by_symbol = {p.symbol: p for p in positions}
    assert by_symbol["BTCUSDT"].qty == 0.5
    assert "XYZUSDT" in by_symbol                            # visible, not hidden
    assert by_symbol["XYZUSDT"].last_price == 0.0


def test_map_account_nan_price_fails_closed():
    account_json = {"balances": [
        {"asset": "USDT", "free": "100", "locked": "0"},
        {"asset": "BTC", "free": "1", "locked": "0"},
    ]}
    account, _ = map_account(account_json, {"BTCUSDT": float("nan")}, "USDT")
    assert account.total_assets == pytest.approx(100.0)      # nan price -> 0, not nan


def test_map_order_response_status():
    assert map_order_response({"symbol": "btcusdt", "side": "BUY", "origQty": "0.5",
                               "price": "100", "status": "NEW"}).accepted is True
    assert map_order_response({"symbol": "BTCUSDT", "side": "SELL", "origQty": "0.5",
                               "price": "100", "status": "REJECTED"}).accepted is False
    assert map_order_response({"symbol": "BTCUSDT", "side": "SELL", "origQty": "0.5",
                               "price": "100", "status": "EXPIRED"}).accepted is False


# --- venue filters --------------------------------------------------------------
def test_venue_from_exchange_filters():
    info = {"symbol": "BTCUSDT", "filters": [
        {"filterType": "PRICE_FILTER", "tickSize": "0.01000000"},
        {"filterType": "LOT_SIZE", "stepSize": "0.00001000"},
        {"filterType": "NOTIONAL", "minNotional": "5.00000000"},
    ]}
    venue = venue_from_exchange_filters(info)
    assert venue.lot_size == "0.00001000"
    assert venue.min_notional == 5.0
    from decimal import Decimal

    assert venue.tick_for(50_000.0) == Decimal("0.01")


def test_venue_from_exchange_filters_missing_raises():
    with pytest.raises(ValueError, match="lacks"):
        venue_from_exchange_filters({"symbol": "X", "filters": []})


# --- fractional lots end to end -------------------------------------------------
def test_quantize_qty_fractional_and_integer():
    frac = VenueRules(name="c", lot_size="0.00001")
    assert frac.quantize_qty(0.000157) == pytest.approx(0.00015)
    assert frac.quantize_qty(-0.000157) == pytest.approx(-0.00015)  # toward zero
    whole = VenueRules(name="e", lot_size=1)                        # int accepted
    assert whole.quantize_qty(123.9) == 123.0


def test_validate_order_fractional_lot_grid():
    venue = VenueRules(name="c", lot_size="0.00001", min_notional=0.0)
    acct = AccountState(cash=1e9)
    on_grid = Order(symbol="BTCUSDT", side="buy", qty=0.00015, limit_price=100_000.0)
    off_grid = Order(symbol="BTCUSDT", side="buy", qty=0.000015, limit_price=100_000.0)
    assert validate_order(on_grid, acct, venue) is None
    rej = validate_order(off_grid, acct, venue)
    assert rej and any("lot_size" in v for v in rej.violations)


# --- the signed client over a mock transport ------------------------------------
def _mock_broker(handler, **kwargs):
    return BinanceBroker(
        api_key="k", api_secret="s",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
        time_fn=lambda: 1700000000.0,
        **kwargs,
    )


def test_broker_defaults_to_testnet():
    broker = BinanceBroker(api_key="k", api_secret="s")
    assert broker.is_testnet is True
    assert TESTNET_BASE_URL != PROD_BASE_URL


def test_broker_requires_keys():
    broker = BinanceBroker(api_key="", api_secret="",
                           client=httpx.Client(transport=httpx.MockTransport(
                               lambda r: httpx.Response(200, json={}))))
    with pytest.raises(RuntimeError, match="BINANCE_API_KEY"):
        broker.get_account()


def test_submit_order_signs_and_formats_decimal():
    seen = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["path"] = request.url.path
        seen["params"] = dict(request.url.params)
        seen["headers"] = dict(request.headers)
        return httpx.Response(200, json={
            "symbol": "BTCUSDT", "side": "BUY", "origQty": "0.00015",
            "price": "100000.00", "orderId": 42, "status": "NEW",
        })

    broker = _mock_broker(handler)
    order = broker.submit_order("BTCUSDT", "buy", 0.00015, limit_price=100_000.0)
    assert seen["path"] == "/api/v3/order"
    assert seen["params"]["quantity"] == "0.00015"          # never '1.5e-05'
    assert seen["params"]["type"] == "LIMIT"
    assert "signature" in seen["params"]
    assert seen["headers"]["x-mbx-apikey"] == "k"
    assert order.accepted and order.order_id == "42"
    assert order.limit_price == pytest.approx(100_000.0)


def test_get_account_values_positions_via_prices():
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/api/v3/account":
            return httpx.Response(200, json={"balances": [
                {"asset": "USDT", "free": "500", "locked": "0"},
                {"asset": "BTC", "free": "0.01", "locked": "0"},
            ]})
        if request.url.path == "/api/v3/ticker/price":
            return httpx.Response(200, json=[{"symbol": "BTCUSDT", "price": "100000"}])
        raise AssertionError(request.url.path)

    broker = _mock_broker(handler)
    account = broker.get_account()
    assert account.cash == 500.0
    assert account.total_assets == pytest.approx(1500.0)
    positions = broker.get_positions()
    assert positions[0].symbol == "BTCUSDT" and positions[0].qty == 0.01


def test_cancel_all_groups_per_symbol():
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path, dict(request.url.params)))
        if request.method == "GET" and request.url.path == "/api/v3/openOrders":
            return httpx.Response(200, json=[
                {"symbol": "BTCUSDT", "side": "BUY", "origQty": "1", "price": "1", "status": "NEW"},
                {"symbol": "ETHUSDT", "side": "SELL", "origQty": "1", "price": "1", "status": "NEW"},
                {"symbol": "BTCUSDT", "side": "SELL", "origQty": "1", "price": "1", "status": "NEW"},
            ])
        if request.method == "DELETE":
            sym = request.url.params["symbol"]
            n = 2 if sym == "BTCUSDT" else 1
            return httpx.Response(200, json=[{} for _ in range(n)])
        raise AssertionError(request.url.path)

    broker = _mock_broker(handler)
    assert broker.cancel_all_orders() == 3
    deletes = [c for c in calls if c[0] == "DELETE"]
    assert sorted(c[2]["symbol"] for c in deletes) == ["BTCUSDT", "ETHUSDT"]


# --- PIT round trip: klines into the store, guarded as usual --------------------
def test_binance_bars_respect_asof_in_store():
    store = PointInTimeStore()
    rows = [_kline(utc(2024, 1, 1) + timedelta(days=i), 100 + i, 101 + i, 99 + i, 100 + i, 10)
            for i in range(5)]
    store.append_many(map_klines("BTCUSDT", rows, as_of=utc(2024, 2, 1)))
    clock = AsOfClock.at(utc(2024, 1, 3))            # bars close Jan 2,3,4,5,6
    visible = store.get_ohlcv("BTCUSDT", clock)
    assert [b.event_time for b in visible] == [utc(2024, 1, 2), utc(2024, 1, 3)]
