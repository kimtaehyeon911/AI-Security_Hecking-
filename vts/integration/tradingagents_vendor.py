"""Register the point-in-time store as a TradingAgents data vendor.

Step 0 showed data tools resolve through ``dataflows.interface.VENDOR_METHODS``
(``method -> {vendor: impl}``) selected by ``config['data_vendors']``. We add a
``"pit"`` vendor whose implementations read exclusively from the
:class:`~vts.pit.store.PointInTimeStore`, capped at a per-decision as-of clock.
No yfinance is imported on this path — that is how "yfinance 의존 제거" is met
for the agent graph.

The as-of clock is injected per decision via a :class:`contextvars.ContextVar`
that the Step 2 backtest harness sets before each ``propagate(ticker, date)``.
This module is import-safe without TradingAgents installed: the registration
function imports it lazily and raises a clear error only if actually called.
"""

from __future__ import annotations

import contextvars
from datetime import datetime, timezone

from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore

# The simulated 'now' for the in-flight decision. The harness sets this; the
# vendor functions read it. Unset -> reads are refused (never fall back to live).
CURRENT_CLOCK: contextvars.ContextVar[AsOfClock | None] = contextvars.ContextVar(
    "vts_current_clock", default=None
)


def _require_clock() -> AsOfClock:
    clock = CURRENT_CLOCK.get()
    if clock is None:
        raise RuntimeError(
            "no as-of clock set: the vts PIT vendor refuses to read without a "
            "simulated clock (set vts.integration.CURRENT_CLOCK before propagate())."
        )
    return clock


def _parse_date(s: str) -> datetime:
    """Parse a 'YYYY-MM-DD' (TradingAgents passes date strings) as UTC midnight."""
    return datetime.strptime(s[:10], "%Y-%m-%d").replace(tzinfo=timezone.utc)


class PITDataProvider:
    """Formats point-in-time store reads into the strings TradingAgents tools return.

    Pure and clock-explicit, so it is unit-testable without the agent graph.
    """

    def __init__(self, store: PointInTimeStore) -> None:
        self._store = store

    # -- prices ---------------------------------------------------------------
    def get_stock_data(self, symbol: str, start_date: str, end_date: str, clock: AsOfClock) -> str:
        # end_date is a calendar day; a daily bar's event_time is the session close
        # (e.g. 21:00 UTC), so an inclusive end must extend to end-of-day or the
        # decision date's own bar is dropped. Push the upper bound to next midnight
        # (the clock still caps visibility, so this cannot leak).
        bars = self._store.get_ohlcv(
            symbol, clock, start=_parse_date(start_date), end=_parse_date(end_date) + _one_day()
        )
        if not bars:
            return f"No point-in-time price data for {symbol} up to {clock.as_of.date()}."
        lines = [f"## {symbol} daily OHLCV (raw, as-of {clock.as_of.date()})", "date,open,high,low,close,volume"]
        for b in bars:
            lines.append(
                f"{b.event_time.date()},{b.open:g},{b.high:g},{b.low:g},{b.close:g},{b.volume:g}"
            )
        return "\n".join(lines)

    # -- news -----------------------------------------------------------------
    def get_news(self, symbol: str, start_date: str, end_date: str, clock: AsOfClock) -> str:
        items = self._store.get_news(symbol, clock, since=_parse_date(start_date))
        items = [n for n in items if n.knowledge_time <= _parse_date(end_date) + _one_day()]
        if not items:
            return f"No news for {symbol} in [{start_date}, {end_date}] knowable as-of {clock.as_of.date()}."
        lines = [f"## {symbol} news (publish-time only, as-of {clock.as_of.date()}):"]
        for n in items:
            score = "" if n.sentiment_score is None else f" [sentiment {n.sentiment_score:+.2f}]"
            lines.append(f"### {n.title} ({n.publisher}, {n.knowledge_time.date()}){score}")
            if n.summary:
                lines.append(n.summary)
        return "\n".join(lines)

    # -- fundamentals ---------------------------------------------------------
    def get_fundamentals(self, symbol: str, curr_date: str, clock: AsOfClock) -> str:
        facts = self._store.get_fundamentals(symbol, clock)
        if not facts:
            return f"No fundamentals for {symbol} knowable as-of {clock.as_of.date()}."
        lines = [f"## {symbol} fundamentals (latest-known as-of {clock.as_of.date()}):"]
        for f in sorted(facts, key=lambda x: (x.metric, x.event_time)):
            est = " (est. filing date)" if f.knowledge_time_estimated else ""
            lines.append(
                f"- {f.metric} [{f.period} ending {f.event_time.date()}]: {f.value} "
                f"{f.currency}{est}"
            )
        return "\n".join(lines)

    # -- graceful stubs for categories not yet ingested in Step 1 --------------
    def get_global_news(self, curr_date: str, clock: AsOfClock) -> str:
        return (
            f"No point-in-time global/macro news in the vts store as-of "
            f"{clock.as_of.date()} (Step 1 ingests per-symbol news only)."
        )

    def get_insider_transactions(self, symbol: str, clock: AsOfClock) -> str:
        return (
            f"No insider-transaction data in the vts store for {symbol} "
            f"(not ingested in Step 1)."
        )


def _one_day():
    from datetime import timedelta

    return timedelta(days=1)


def build_pit_impls(provider: PITDataProvider) -> dict:
    """Return the ``method_name -> callable`` map the ``"pit"`` vendor provides.

    Every callable reads the current as-of clock (never live data) and delegates
    to ``provider``. Covers the full method set of the price / fundamentals / news
    categories so routing an analyst tool to ``"pit"`` can never hit a missing
    implementation. Indicators, macro and prediction-markets are intentionally NOT
    here — they keep their own vendors until Step 2 wires them to the PIT store.
    """
    def _stock(symbol: str, start_date: str, end_date: str, *_a, **_k) -> str:
        return provider.get_stock_data(symbol, start_date, end_date, _require_clock())

    def _news(ticker: str, start_date: str, end_date: str, *_a, **_k) -> str:
        return provider.get_news(ticker, start_date, end_date, _require_clock())

    def _fundamentals(ticker: str, curr_date: str = "", *_a, **_k) -> str:
        return provider.get_fundamentals(ticker, curr_date, _require_clock())

    def _global_news(curr_date: str, *_a, **_k) -> str:
        return provider.get_global_news(curr_date, _require_clock())

    def _insider(ticker: str, *_a, **_k) -> str:
        return provider.get_insider_transactions(ticker, _require_clock())

    # Balance sheet / cash flow / income statement share the unified Step 1
    # fundamentals view (statement-type separation is a Step 2 refinement).
    return {
        "get_stock_data": _stock,
        "get_news": _news,
        "get_fundamentals": _fundamentals,
        "get_balance_sheet": _fundamentals,
        "get_cashflow": _fundamentals,
        "get_income_statement": _fundamentals,
        "get_global_news": _global_news,
        "get_insider_transactions": _insider,
    }


def register_pit_vendor(store: PointInTimeStore, *, set_as_default: bool = True) -> list[str]:
    """Register the ``"pit"`` vendor into TradingAgents' ``VENDOR_METHODS``.

    Raises ``ImportError`` with a clear message if TradingAgents is not installed.
    Returns the list of method names registered. When ``set_as_default`` is True,
    routes **only the implemented methods** to ``"pit"`` via per-method
    ``tool_vendors`` (never a category-wide default), so a method we do not
    implement is never dispatched to ``"pit"`` — closing the over-scoping crash.
    """
    try:
        from tradingagents.dataflows import interface as ta_interface  # type: ignore
        from tradingagents.dataflows.config import get_config, set_config  # type: ignore
    except ImportError as exc:  # pragma: no cover - exercised only with TA installed
        raise ImportError(
            "TradingAgents is not installed; `register_pit_vendor` needs the agent "
            "graph. Install the fork (Step 2 wiring) before calling this."
        ) from exc

    impls = build_pit_impls(PITDataProvider(store))
    for method, fn in impls.items():
        ta_interface.VENDOR_METHODS.setdefault(method, {})["pit"] = fn

    if set_as_default:
        cfg = get_config()
        tool_vendors = cfg.setdefault("tool_vendors", {})
        for method in impls:
            tool_vendors[method] = "pit"  # per-method: only implemented methods route to pit
        set_config(cfg)

    return list(impls)
