"""Regression tests for the Step 1 adversarial-review findings."""

from __future__ import annotations

import pytest
from conftest import utc

from vts.config import Settings
from vts.integration.tradingagents_vendor import PITDataProvider
from vts.pit.clock import AsOfClock
from vts.pit.schema import FundamentalFact, NewsItem, OHLCVBar
from vts.pit.store import PointInTimeStore
from vts.sources.alpha_vantage import map_earnings, map_news_feed, map_statements


def _bar(day: int, close: float) -> OHLCVBar:
    t = utc(2024, 1, day, 21)
    return OHLCVBar(symbol="AAPL", event_time=t, knowledge_time=t, source="t",
                    open=close, high=close, low=close, close=close, volume=1000)


# --- HIGH: get_stock_data must include the decision date's own bar -----------
def test_get_stock_data_includes_final_day():
    store = PointInTimeStore()
    store.append_many([_bar(9, 100), _bar(10, 101)])
    provider = PITDataProvider(store)
    clock = AsOfClock.at(utc(2024, 1, 10, 21))
    # end_date == 2024-01-10; the bar stamped 21:00 that day must appear.
    out = provider.get_stock_data("AAPL", "2024-01-10", "2024-01-10", clock)
    assert "2024-01-10" in out
    assert "No point-in-time price data" not in out


# --- MEDIUM: per-ticker sentiment of exactly 0.0 preserved -------------------
def test_zero_ticker_sentiment_not_overwritten():
    feed = {"feed": [{
        "title": "t", "url": "u", "time_published": "20240103T133000",
        "overall_sentiment_score": "0.75",
        "ticker_sentiment": [{"ticker": "NVDA", "relevance_score": "0.9",
                              "ticker_sentiment_score": "0.0"}],
    }]}
    items = map_news_feed("NVDA", feed)
    assert items[0].sentiment_score == 0.0  # not 0.75


# --- LOW: seconds-less / garbage publish time handling -----------------------
def test_minute_precision_timestamp_parsed():
    feed = {"feed": [{"title": "t", "url": "u", "time_published": "20240110T0930"}]}
    items = map_news_feed("NVDA", feed)
    assert len(items) == 1
    assert items[0].knowledge_time == utc(2024, 1, 10, 9, 30)


def test_one_bad_timestamp_does_not_drop_whole_batch():
    feed = {"feed": [
        {"title": "good", "url": "u1", "time_published": "20240110T093000"},
        {"title": "bad", "url": "u2", "time_published": "not-a-timestamp"},
    ]}
    items = map_news_feed("NVDA", feed)
    assert {i.title for i in items} == {"good"}  # bad skipped, good survives


# --- LOW: reportedCurrency "None" sentinel falls back to USD ------------------
def test_currency_none_sentinel_becomes_usd():
    income = {"quarterlyReports": [{
        "fiscalDateEnding": "2023-12-31", "reportedCurrency": "None",
        "netIncome": "12285000000",
    }]}
    facts = map_statements("AAPL", income)
    assert facts and all(f.currency == "USD" for f in facts)


# --- LOW: exact microsecond conversion (no float truncation) -----------------
def test_sub_microsecond_boundary_hidden_without_crash():
    store = PointInTimeStore()
    kt = utc(2004, 8, 21, 13, 9, 3).replace(microsecond=960569)
    store.append(NewsItem(symbol="AAPL", event_time=kt, knowledge_time=kt, source="t",
                          title="future", url="z"))
    # as_of is 1 microsecond BEFORE knowledge_time -> genuinely future -> hidden.
    clock = AsOfClock.at(utc(2004, 8, 21, 13, 9, 3).replace(microsecond=960568))
    assert store.get_news("AAPL", clock) == []  # returns [], does not raise


def test_distinct_sub_microsecond_events_do_not_collide():
    store = PointInTimeStore()
    base = utc(2023, 12, 31, 0, 0, 0)
    for us, val in ((960568, 1.0), (960569, 2.0)):
        e = base.replace(microsecond=us)
        store.append(FundamentalFact(symbol="AAPL", event_time=e, knowledge_time=e,
                                     source="t", metric="netIncome", value=val))
    got = store.get_fundamentals("AAPL", AsOfClock.at(utc(2024, 6, 1)), metric="netIncome")
    assert len(got) == 2  # distinct series keys, neither dropped by collapse


# --- LOW: config float parsing raises a clear, named error -------------------
def test_bad_temperature_env_raises_named_error(monkeypatch):
    monkeypatch.setenv("VTS_TEMPERATURE", "high")
    with pytest.raises(ValueError, match="VTS_TEMPERATURE"):
        Settings.from_env()


def test_non_finite_budget_rejected(monkeypatch):
    monkeypatch.setenv("VTS_MONTHLY_BUDGET_USD", "nan")
    with pytest.raises(ValueError, match="finite"):
        Settings.from_env()
