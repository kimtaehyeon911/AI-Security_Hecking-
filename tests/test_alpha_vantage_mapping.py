"""Pure Alpha Vantage mappers: knowledge_time derivation is the whole point."""

from __future__ import annotations

from conftest import utc

from vts.sources.alpha_vantage import (
    map_daily_adjusted_actions,
    map_daily_series,
    map_earnings,
    map_news_feed,
    map_statements,
)

DAILY = {
    "Time Series (Daily)": {
        "2024-01-03": {"1. open": "184.2", "2. high": "185.0", "3. low": "183.0",
                        "4. close": "184.0", "5. volume": "50000000"},
        "2024-01-02": {"1. open": "187.0", "2. high": "188.0", "3. low": "183.9",
                        "4. close": "185.6", "5. volume": "82000000"},
    }
}

DAILY_ADJ = {
    "Time Series (Daily)": {
        "2024-02-09": {"1. open": "1", "2. high": "1", "3. low": "1", "4. close": "1",
                        "6. volume": "1", "7. dividend amount": "0.24", "8. split coefficient": "1.0"},
        "2024-06-10": {"1. open": "1", "2. high": "1", "3. low": "1", "4. close": "1",
                        "6. volume": "1", "7. dividend amount": "0.0", "8. split coefficient": "10.0"},
    }
}

NEWS = {
    "feed": [
        {
            "title": "Chipmaker beats estimates",
            "url": "https://example.com/a",
            "time_published": "20240103T133000",
            "summary": "Strong quarter.",
            "source": "Example Wire",
            "overall_sentiment_score": "0.42",
            "ticker_sentiment": [
                {"ticker": "NVDA", "relevance_score": "0.9", "ticker_sentiment_score": "0.55"}
            ],
        },
        {"title": "no timestamp", "url": "x"},  # must be skipped (unplaceable in time)
    ]
}

EARNINGS = {
    "quarterlyEarnings": [
        {"fiscalDateEnding": "2023-12-31", "reportedDate": "2024-02-21", "reportedEPS": "5.16"}
    ],
    "annualEarnings": [{"fiscalDateEnding": "2023-12-31", "reportedEPS": "12.96"}],
}

INCOME = {
    "quarterlyReports": [
        {"fiscalDateEnding": "2023-12-31", "reportedCurrency": "USD",
         "totalRevenue": "22103000000", "netIncome": "12285000000"}
    ]
}


def test_daily_series_is_raw_and_close_timed():
    bars = map_daily_series("NVDA", DAILY)
    assert [b.event_time.date().isoformat() for b in bars] == ["2024-01-02", "2024-01-03"]
    # knowledge_time == event_time == session close (21:00 UTC default).
    assert bars[0].knowledge_time == utc(2024, 1, 2, 21)
    assert bars[0].close == 185.6  # raw, unadjusted


def test_corporate_actions_separated_from_prices():
    actions = map_daily_adjusted_actions("NVDA", DAILY_ADJ)
    kinds = {(a.action_type, a.event_time.date().isoformat()): a.value for a in actions}
    assert kinds[("dividend", "2024-02-09")] == 0.24
    assert kinds[("split", "2024-06-10")] == 10.0
    # Trivial (split=1.0, div=0.0) rows produce no action.
    assert len(actions) == 2


def test_news_uses_publish_time_and_skips_undated():
    items = map_news_feed("NVDA", NEWS)
    assert len(items) == 1  # undated article dropped
    n = items[0]
    assert n.knowledge_time == utc(2024, 1, 3, 13, 30)
    assert n.sentiment_score == 0.55  # ticker-specific overrides overall
    assert n.relevance == 0.9


def test_earnings_knowledge_is_report_date_not_fiscal_end():
    facts = map_earnings("NVDA", EARNINGS)
    q = [f for f in facts if f.period == "quarterly"][0]
    assert q.event_time.date().isoformat() == "2023-12-31"       # fiscal period end
    assert q.knowledge_time.date().isoformat() == "2024-02-21"    # report date (exact)
    assert q.knowledge_time_estimated is False
    a = [f for f in facts if f.period == "annual"][0]
    assert a.knowledge_time_estimated is True  # annual lacks report date -> estimated


def test_statements_estimate_filing_lag():
    facts = map_statements("NVDA", INCOME)
    ni = [f for f in facts if f.metric == "netIncome"][0]
    assert ni.knowledge_time_estimated is True
    # knowledge_time = fiscal end + 45d default.
    assert ni.knowledge_time > ni.event_time
    assert (ni.knowledge_time - ni.event_time).days == 45
