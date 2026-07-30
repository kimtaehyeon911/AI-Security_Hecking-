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

## Step 1 — Data layer (pending approval)
## Step 2 — Backtest engine (pending)
## Step 3 — Evaluation (pending)
## Step 4 — Risk layer (pending)
## Step 5 — Paper trading (pending)
## Step 6 — Live (pending, explicit approval only)
