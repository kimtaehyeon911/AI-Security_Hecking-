"""Binance spot market-data adapter (public endpoints — no API key).

Pure mappers (JSON → PIT records) + a thin httpx fetch layer, mirroring the
Alpha Vantage adapter's structure so the look-ahead-critical logic — how
``knowledge_time`` is derived — is unit-testable offline.

Point-in-time semantics for klines:

- A kline row is ``[openTime, open, high, low, close, volume, closeTime, ...]``
  with millisecond UTC epochs. A bar becomes knowable exactly when it CLOSES, so
  ``event_time == knowledge_time == closeTime`` (rounded up from Binance's
  ``closeTime = openTime + interval - 1ms``).
- The /klines endpoint includes the current, STILL-FORMING candle as its last
  row. An unclosed bar is future knowledge; the mapper drops any bar whose close
  lies after the caller-supplied ``as_of`` ceiling rather than trusting the
  vendor's row boundary.
- Crypto has no corporate actions, no filings, and Binance serves no historical
  news — those fetchers return empty lists honestly instead of fabricating.
"""

from __future__ import annotations

import math
from datetime import datetime, timezone
from typing import Callable

import httpx

from vts.pit.schema import CorporateAction, FundamentalFact, NewsItem, OHLCVBar

_UTC = timezone.utc
_SOURCE = "binance"

#: Public REST base. Data endpoints need no key; this is NOT the trading host.
DATA_BASE_URL = "https://api.binance.com"

_INTERVAL_MS = {"1d": 86_400_000, "4h": 14_400_000, "1h": 3_600_000}


def _ms_to_dt(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000.0, tz=_UTC)


def _finite(value: object) -> float | None:
    """Parse a Binance numeric string; reject non-finite/garbage rows."""
    try:
        f = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def map_klines(
    symbol: str,
    rows: list[list],
    *,
    interval: str = "1d",
    as_of: datetime | None = None,
) -> list[OHLCVBar]:
    """Map /api/v3/klines rows to PIT bars, keeping only CLOSED bars.

    ``as_of`` is the knowledge ceiling (normally "now" at ingest): a bar whose
    close time is after it is still forming — or from the future — and is
    dropped. Rows with non-finite or non-positive prices are skipped rather than
    poisoning the store (a zero close would break share/return math downstream).
    """
    if interval not in _INTERVAL_MS:
        raise ValueError(f"unsupported interval {interval!r}")
    sym = symbol.strip().upper()
    bars: list[OHLCVBar] = []
    for row in rows:
        if len(row) < 7:
            continue
        o, h, l, c = (_finite(row[1]), _finite(row[2]), _finite(row[3]), _finite(row[4]))
        v = _finite(row[5])
        if None in (o, h, l, c, v) or min(o, h, l, c) <= 0 or v < 0:  # type: ignore[arg-type]
            continue
        # closeTime is openTime + interval - 1ms; stamp the bar at the exact
        # interval boundary so daily bars land on clean UTC midnights.
        close_dt = _ms_to_dt(int(row[6]) + 1)
        if as_of is not None and close_dt > as_of:
            continue  # still-forming (or future) candle — not yet knowledge
        bars.append(
            OHLCVBar(
                symbol=sym, event_time=close_dt, knowledge_time=close_dt,
                source=_SOURCE, interval=interval,  # type: ignore[arg-type]
                open=o, high=h, low=l, close=c, volume=v,
            )
        )
    bars.sort(key=lambda b: b.event_time)
    return bars


class BinanceSource:
    """httpx-backed public-data adapter implementing the DataSource protocol."""

    name = _SOURCE
    _MAX_LIMIT = 1000  # Binance hard cap per /klines request

    def __init__(
        self,
        *,
        interval: str = "1d",
        base_url: str = DATA_BASE_URL,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
        now_fn: Callable[[], datetime] | None = None,
    ) -> None:
        if interval not in _INTERVAL_MS:
            raise ValueError(f"unsupported interval {interval!r}")
        self._interval = interval
        self._base = base_url.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout)
        self._now_fn = now_fn or (lambda: datetime.now(_UTC))

    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime) -> list[OHLCVBar]:
        """Closed bars in ``[start, end]``, paginating past the 1000-row cap.

        The knowledge ceiling passed to the mapper is ``min(end, NOW)``: with a
        future ``end``, Binance's still-forming candle carries a scheduled
        closeTime inside the window and would otherwise be stored as a closed bar
        whose OHLC later changes — a restatement the PIT store cannot detect
        because the stamped knowledge_time looks legitimate. Wall clock at the
        ingest boundary is correct usage: ingestion IS a wall-clock activity
        (backtest reads stay governed by the as-of clock, not this).
        """
        sym = symbol.strip().upper()
        step_ms = _INTERVAL_MS[self._interval]
        # One interval early: Binance filters klines by OPEN time, but our
        # [start, end] contract is on CLOSE time — the bar closing exactly at
        # `start` opened one interval before it and would otherwise be missing.
        start_ms = int(start.timestamp() * 1000) - step_ms
        end_ms = int(end.timestamp() * 1000)
        as_of = min(end, self._now_fn())
        out: list[OHLCVBar] = []
        cursor = start_ms
        while cursor <= end_ms:
            resp = self._client.get(
                f"{self._base}/api/v3/klines",
                params={
                    "symbol": sym, "interval": self._interval,
                    "startTime": cursor, "endTime": end_ms, "limit": self._MAX_LIMIT,
                },
            )
            resp.raise_for_status()
            rows = resp.json()
            if not rows:
                break
            out.extend(map_klines(sym, rows, interval=self._interval, as_of=as_of))
            last_open = int(rows[-1][0])
            next_cursor = last_open + step_ms
            if next_cursor <= cursor:  # defensive: never loop on a stuck cursor
                break
            cursor = next_cursor
            if len(rows) < self._MAX_LIMIT:
                break
        # Pagination overlap safety: dedupe on event_time.
        seen: set[datetime] = set()
        deduped = []
        for b in out:
            if b.event_time not in seen:
                seen.add(b.event_time)
                deduped.append(b)
        return [b for b in deduped if start <= b.event_time <= end]

    def fetch_corporate_actions(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[CorporateAction]:
        return []  # crypto spot: no splits/dividends

    def fetch_news(self, symbol: str, start: datetime, end: datetime) -> list[NewsItem]:
        return []  # Binance serves no historical news; wire a news vendor separately

    def fetch_fundamentals(self, symbol: str) -> list[FundamentalFact]:
        return []  # no filings in crypto — honest empty, never fabricated

    def close(self) -> None:
        self._client.close()
