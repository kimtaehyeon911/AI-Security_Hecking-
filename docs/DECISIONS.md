# Verifiable TradingAgents System — Decision Log

A running record of decisions, one section per Step. Base: fork of
[TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) (Apache-2.0).
Priority order: (1) backtest harness → (2) paper trading → (3) live (kill-switch mandatory, last).

## Step 0 — Code reading (done)
Full analysis: [`docs/step0_code_reading.md`](./step0_code_reading.md).

Key decisions:
1. **Keep the vendor-dispatch architecture** (`dataflows/interface.py::VENDOR_METHODS`). A new
   broker/exchange is a registered vendor function + a config key, not a call-site rewrite.
2. **Adopt the existing 5-tier `PortfolioRating`** (`Buy/Overweight/Hold/Underweight/Sell`) as the
   Trading-R1 decision output. **Add** a decision-level `confidence` + structured `evidence` list
   (Steps 2/4); do not invent a new enum.
3. **Replace the data layer with a point-in-time store** in Step 1 (`as_of` on every record +
   `assert_no_lookahead`). Existing guards are "download-now-then-filter" (auto-adjust restatement leak;
   yfinance news cannot serve history) — non-authoritative for backtests.
4. **Neutralize deferred-reflection look-ahead** (`propagate → _resolve_pending_entries →
   _fetch_returns` pulls realized forward returns via live yfinance). Backtest harness must gate
   resolution to `entry.date + holding_days <= trade_date` and route returns through the PIT store.
   **Write the falsification test first (Step 2).**
5. **Add a model knowledge-cutoff registry** (absent in `model_catalog.py`) and gate the backtest window
   on it; label pre-cutoff results "참고용(오염 가능)".

Determinism gaps to close in Step 2: no `seed`, no response cache — add fixed temperature + on-disk
cache keyed by `(ticker, date, model, prompt-hash)` + `N=3` majority-vote rating with dispersion metric.

**Blocked on user inputs** (environment placeholders unfilled): asset class, broker/exchange API,
LLM provider + monthly budget, trading cadence. See §12 of the Step 0 doc.

## Step 1 — Data layer (done)
New `vts` package (Python 3.12, uv, pydantic v2, pytest). Environment values fixed as reasonable
defaults, all env-overridable via `VTS_*` (see `.env.example`, `vts/config.py`):

| 항목 | 값 |
|---|---|
| 대상 자산 | 미국주식 (US equities) |
| 데이터 소스 (백테스트) | Alpha Vantage (`NEWS_SENTIMENT` publish-time; `EARNINGS` reportedDate; raw prices) |
| LLM | deepseek — deep=`deepseek-v4-pro`, quick=`deepseek-v4-flash`, temp=0 |
| 월 예산 | $30 |
| 매매 주기 | 일봉 1회 (`1d`) |

Key decisions:
1. **Two-timestamp model.** Every record carries `event_time` (subject moment) and `knowledge_time`
   (when first knowable). As-of reads return only `knowledge_time <= clock`. This defeats both Step 0
   leaks: (a) auto-adjust price restatement — we store **raw** OHLCV and treat adjustment as an as-of
   *view* using only known splits (`vts/pit/adjust.py`); (b) future news — news is keyed by publish time.
2. **Enforced guard at egress.** `assert_no_lookahead` / `filter_visible(strict=True)` re-check every
   record leaving the store (redundant with the SQL `knowledge_time <= T` filter) → a leaked future row
   raises `LookaheadError`, never silently inflates a metric.
3. **Append-only + restatement collapse.** A correction is a new row (higher `revision`, later
   `knowledge_time`); reads return the latest-known revision per series as-of `T`. Fundamentals keyed by
   `reportedDate` (exact) or `fiscal_end + lag` (estimated, flagged).
4. **yfinance-free path.** PIT store registers as a TradingAgents `"pit"` vendor
   (`vts/integration/`), clock injected per decision via a `contextvar`. Full graph wiring is Step 2.
5. **Determinism/seed/cache and the reflection look-ahead gate are Step 2** (not the data layer).

Tests: 29 passing incl. falsification (future ingestion cannot change a past as-of view; store never
returns a future record; guard catches hand-crafted leaks; restatement flips at restatement date; split
adjustment ignores not-yet-known splits). Secrets: `.env` gitignored, no keys committed.

Follow-up (tracked for Step 2): route TradingAgents' two hard yfinance couplings
(`get_verified_market_snapshot`, `resolve_instrument_identity`) and `_fetch_returns` through the PIT
store; neutralize `_resolve_pending_entries` reflection look-ahead; add model knowledge-cutoff registry;
SQLite `check_same_thread`/locking for threaded graph reads.

## Step 2 — Backtest engine (pending)
## Step 3 — Evaluation (pending)
## Step 4 — Risk layer (pending)
## Step 5 — Paper trading (pending)
## Step 6 — Live (pending, explicit approval only)
