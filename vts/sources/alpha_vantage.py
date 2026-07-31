"""Alpha Vantage adapter.

The mapping functions (``map_*``) are **pure**: vendor JSON in, PIT records out,
no network. The :class:`AlphaVantageSource` class adds httpx fetching on top. This
split keeps the look-ahead-critical logic (how ``knowledge_time`` is derived)
unit-testable against fixtures, with zero API key or network in the test suite.

Why Alpha Vantage for a point-in-time US-equity backtest (Step 0/1 decision):
- ``NEWS_SENTIMENT`` returns ``time_published`` and accepts ``time_from``/
  ``time_to`` — genuine publish-time retrieval (yfinance cannot serve history).
- Prices are stored **raw** from ``TIME_SERIES_DAILY``; splits/dividends come from
  ``TIME_SERIES_DAILY_ADJUSTED`` as separately-timed :class:`CorporateAction`s, so
  no future adjustment ever mutates a stored bar.
- ``EARNINGS`` carries ``reportedDate`` — an exact knowledge date for EPS.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

from vts.pit.schema import CorporateAction, FundamentalFact, NewsItem, OHLCVBar

_UTC = timezone.utc


@dataclass(frozen=True, slots=True)
class AVTimeConventions:
    """Intraday-time assumptions used to place a knowledge_time on date-only data.

    Alpha Vantage gives calendar dates for bars, ex-dates and (some) reports but
    not the intraday instant. We choose *conservative* defaults so a same-day
    decision cannot peek at data a real observer wouldn't yet have:

    - ``market_close_hour_utc`` — a daily bar's close is knowable only at close.
    - ``earnings_release_hour_utc`` — assume after-market-close (AMC); worst case
      this delays visibility by one session, never advances it.
    - ``corporate_action_hour_utc`` — ex-date adjustment applies from the open.
    - ``*_report_lag_days`` — when only a fiscal period end is known (statements
      without a report date), estimate the filing date as period_end + lag and
      flag ``knowledge_time_estimated=True``.
    """

    market_close_hour_utc: int = 21
    earnings_release_hour_utc: int = 21
    corporate_action_hour_utc: int = 14
    quarterly_report_lag_days: int = 45
    annual_report_lag_days: int = 90


DEFAULT_CONVENTIONS = AVTimeConventions()

_SOURCE = "alpha_vantage"


def _date_at(date_str: str, hour_utc: int) -> datetime:
    d = datetime.strptime(date_str, "%Y-%m-%d")
    return d.replace(hour=hour_utc, minute=0, second=0, tzinfo=_UTC)


def _parse_published(ts: str) -> datetime | None:
    """Parse Alpha Vantage ``time_published`` (UTC).

    The documented format is ``YYYYMMDDTHHMMSS`` (15 chars); tolerate the
    minute-precision ``YYYYMMDDTHHMM`` (13 chars) variant too. Dispatch by length
    rather than trying formats in sequence: ``strptime`` backtracks, so a
    seconds-format parse of ``"0930"`` would wrongly yield 09:03:00. Returns
    ``None`` on any unparseable value so a single malformed row is skipped rather
    than aborting the whole news batch.
    """
    ts = ts.strip()
    fmt = {15: "%Y%m%dT%H%M%S", 13: "%Y%m%dT%H%M"}.get(len(ts))
    if fmt is None:
        return None
    try:
        return datetime.strptime(ts, fmt).replace(tzinfo=_UTC)
    except ValueError:
        return None


_SENTINEL_STRINGS = (None, "", "None", "none", "N/A", "na", "-")


def _f(value: object) -> float | None:
    try:
        if value in _SENTINEL_STRINGS:
            return None
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _s(value: object, default: str) -> str:
    """Return ``value`` as a string, mapping vendor sentinel strings to ``default``.

    Alpha Vantage emits the literal string ``"None"`` (not JSON null) for unset
    fields, so ``x or default`` is wrong — ``"None"`` is truthy.
    """
    return default if value in _SENTINEL_STRINGS else str(value)


# --------------------------------------------------------------------- mappers
def map_daily_series(
    symbol: str,
    payload: dict,
    conv: AVTimeConventions = DEFAULT_CONVENTIONS,
) -> list[OHLCVBar]:
    """Map ``TIME_SERIES_DAILY`` JSON to raw OHLCV bars.

    A bar's ``event_time`` and ``knowledge_time`` are both the session close
    (``market_close_hour_utc``): the close is known exactly at the close.
    """
    series = payload.get("Time Series (Daily)", {})
    bars: list[OHLCVBar] = []
    for date_str, row in series.items():
        close_dt = _date_at(date_str, conv.market_close_hour_utc)
        volume = _f(row.get("5. volume") or row.get("6. volume")) or 0.0
        bars.append(
            OHLCVBar(
                symbol=symbol,
                event_time=close_dt,
                knowledge_time=close_dt,
                source=_SOURCE,
                interval="1d",
                open=_f(row.get("1. open")) or 0.0,
                high=_f(row.get("2. high")) or 0.0,
                low=_f(row.get("3. low")) or 0.0,
                close=_f(row.get("4. close")) or 0.0,
                volume=volume,
            )
        )
    return sorted(bars, key=lambda b: b.event_time)


def map_daily_adjusted_actions(
    symbol: str,
    payload: dict,
    conv: AVTimeConventions = DEFAULT_CONVENTIONS,
) -> list[CorporateAction]:
    """Extract splits/dividends from ``TIME_SERIES_DAILY_ADJUSTED`` as PIT actions.

    Only non-trivial actions are emitted (dividend > 0, split coefficient != 1).
    ``knowledge_time == event_time == ex-date`` at the open: an action is certainly
    known by its ex-date, and its price effect applies from the open.
    """
    series = payload.get("Time Series (Daily)", {})
    actions: list[CorporateAction] = []
    for date_str, row in series.items():
        when = _date_at(date_str, conv.corporate_action_hour_utc)
        dividend = _f(row.get("7. dividend amount")) or 0.0
        split = _f(row.get("8. split coefficient")) or 1.0
        if dividend > 0.0:
            actions.append(
                CorporateAction(
                    symbol=symbol, event_time=when, knowledge_time=when, source=_SOURCE,
                    action_type="dividend", value=dividend,
                )
            )
        if split not in (0.0, 1.0):
            actions.append(
                CorporateAction(
                    symbol=symbol, event_time=when, knowledge_time=when, source=_SOURCE,
                    action_type="split", value=split,
                )
            )
    return sorted(actions, key=lambda a: a.event_time)


def map_news_feed(symbol: str, payload: dict) -> list[NewsItem]:
    """Map ``NEWS_SENTIMENT`` JSON to news items keyed by publish time.

    Per-ticker relevance/sentiment is pulled from ``ticker_sentiment`` when the
    requested symbol appears there; otherwise the article's overall sentiment.
    """
    feed = payload.get("feed", [])
    sym = symbol.strip().upper()
    items: list[NewsItem] = []
    for art in feed:
        ts = art.get("time_published")
        published = _parse_published(ts) if ts else None
        if published is None:
            # No (usable) publish time -> cannot place it in time -> must not be
            # usable in a historical window. Skip rather than guess (Step 0 policy).
            continue
        sentiment = _f(art.get("overall_sentiment_score"))
        relevance = None
        for ts_row in art.get("ticker_sentiment", []):
            if ts_row.get("ticker", "").strip().upper() == sym:
                # `is not None`, not `or`: a genuine per-ticker score of exactly
                # 0.0 (neutral) is falsy and must not fall back to overall.
                ticker_score = _f(ts_row.get("ticker_sentiment_score"))
                if ticker_score is not None:
                    sentiment = ticker_score
                relevance = _f(ts_row.get("relevance_score"))
                break
        # Clamp sentiment into schema bounds [-1, 1] defensively.
        if sentiment is not None:
            sentiment = max(-1.0, min(1.0, sentiment))
        items.append(
            NewsItem(
                symbol=sym,
                event_time=published,
                knowledge_time=published,
                source=_SOURCE,
                title=art.get("title") or "(untitled)",
                url=art.get("url") or "",
                publisher=art.get("source") or "",
                summary=art.get("summary") or "",
                sentiment_score=sentiment,
                relevance=relevance,
            )
        )
    return sorted(items, key=lambda n: n.knowledge_time)


def map_earnings(
    symbol: str,
    payload: dict,
    conv: AVTimeConventions = DEFAULT_CONVENTIONS,
) -> list[FundamentalFact]:
    """Map ``EARNINGS`` JSON to EPS facts keyed by report date (exact knowledge)."""
    facts: list[FundamentalFact] = []
    for row in payload.get("quarterlyEarnings", []):
        fiscal = row.get("fiscalDateEnding")
        reported = row.get("reportedDate")
        eps = _f(row.get("reportedEPS"))
        if not fiscal or not reported:
            continue
        facts.append(
            FundamentalFact(
                symbol=symbol,
                event_time=_date_at(fiscal, 0),
                knowledge_time=_date_at(reported, conv.earnings_release_hour_utc),
                source=_SOURCE,
                metric="reportedEPS",
                value=eps,
                period="quarterly",
            )
        )
    for row in payload.get("annualEarnings", []):
        fiscal = row.get("fiscalDateEnding")
        eps = _f(row.get("reportedEPS"))
        if not fiscal:
            continue
        # Annual earnings lack a reportedDate -> estimate.
        event = _date_at(fiscal, 0)
        facts.append(
            FundamentalFact(
                symbol=symbol,
                event_time=event,
                knowledge_time=event + timedelta(days=conv.annual_report_lag_days),
                source=_SOURCE,
                metric="reportedEPS",
                value=eps,
                period="annual",
                knowledge_time_estimated=True,
            )
        )
    return facts


def map_statements(
    symbol: str,
    payload: dict,
    conv: AVTimeConventions = DEFAULT_CONVENTIONS,
) -> list[FundamentalFact]:
    """Map INCOME_STATEMENT / BALANCE_SHEET / CASH_FLOW JSON to fundamental facts.

    These endpoints give no filing date, so knowledge_time is estimated as
    ``fiscal_period_end + reporting_lag`` and flagged estimated.
    """
    facts: list[FundamentalFact] = []
    for period_key, lag, period_label in (
        ("quarterlyReports", conv.quarterly_report_lag_days, "quarterly"),
        ("annualReports", conv.annual_report_lag_days, "annual"),
    ):
        for row in payload.get(period_key, []):
            fiscal = row.get("fiscalDateEnding")
            if not fiscal:
                continue
            event = _date_at(fiscal, 0)
            know = event + timedelta(days=lag)
            currency = _s(row.get("reportedCurrency"), "USD")
            for metric, raw in row.items():
                if metric in ("fiscalDateEnding", "reportedCurrency"):
                    continue
                value = _f(raw)
                if value is None:
                    continue
                facts.append(
                    FundamentalFact(
                        symbol=symbol,
                        event_time=event,
                        knowledge_time=know,
                        source=_SOURCE,
                        metric=metric,
                        value=value,
                        period=period_label,  # type: ignore[arg-type]
                        currency=currency,
                        knowledge_time_estimated=True,
                    )
                )
    return facts


# ------------------------------------------------------------------- HTTP class
class AlphaVantageSource:
    """httpx-backed Alpha Vantage adapter.

    The API key is read from ``ALPHAVANTAGE_API_KEY`` (never hard-coded, never
    committed — see ``.env.example``). Fetch methods delegate to the pure mappers.
    """

    name = _SOURCE
    _BASE = "https://www.alphavantage.co/query"

    def __init__(
        self,
        api_key: str | None = None,
        *,
        conventions: AVTimeConventions = DEFAULT_CONVENTIONS,
        client: httpx.Client | None = None,
        timeout: float = 30.0,
    ) -> None:
        self._api_key = api_key or os.environ.get("ALPHAVANTAGE_API_KEY", "")
        self._conv = conventions
        self._client = client or httpx.Client(timeout=timeout)

    def _get(self, params: dict[str, str]) -> dict:
        if not self._api_key:
            raise RuntimeError(
                "ALPHAVANTAGE_API_KEY is not set; add it to your .env "
                "(never commit the key)."
            )
        params = {**params, "apikey": self._api_key}
        resp = self._client.get(self._BASE, params=params)
        resp.raise_for_status()
        data = resp.json()
        # Alpha Vantage returns 200 with a 'Note'/'Information' body on throttling.
        if "Note" in data or "Information" in data:
            raise RuntimeError(f"Alpha Vantage throttled/errored: {data}")
        return data

    def fetch_ohlcv(self, symbol: str, start: datetime, end: datetime) -> list[OHLCVBar]:
        payload = self._get(
            {"function": "TIME_SERIES_DAILY", "symbol": symbol, "outputsize": "full"}
        )
        bars = map_daily_series(symbol, payload, self._conv)
        return [b for b in bars if start <= b.event_time <= end]

    def fetch_corporate_actions(
        self, symbol: str, start: datetime, end: datetime
    ) -> list[CorporateAction]:
        payload = self._get(
            {"function": "TIME_SERIES_DAILY_ADJUSTED", "symbol": symbol, "outputsize": "full"}
        )
        actions = map_daily_adjusted_actions(symbol, payload, self._conv)
        return [a for a in actions if start <= a.event_time <= end]

    def fetch_news(self, symbol: str, start: datetime, end: datetime) -> list[NewsItem]:
        params = {
            "function": "NEWS_SENTIMENT",
            "tickers": symbol,
            "time_from": start.astimezone(_UTC).strftime("%Y%m%dT%H%M"),
            "time_to": end.astimezone(_UTC).strftime("%Y%m%dT%H%M"),
            "sort": "EARLIEST",
            "limit": "1000",
        }
        payload = self._get(params)
        return map_news_feed(symbol, payload)

    def fetch_fundamentals(self, symbol: str) -> list[FundamentalFact]:
        facts: list[FundamentalFact] = []
        facts.extend(map_earnings(symbol, self._get({"function": "EARNINGS", "symbol": symbol}), self._conv))
        for fn in ("INCOME_STATEMENT", "BALANCE_SHEET", "CASH_FLOW"):
            facts.extend(map_statements(symbol, self._get({"function": fn, "symbol": symbol}), self._conv))
        return facts

    def close(self) -> None:
        self._client.close()
