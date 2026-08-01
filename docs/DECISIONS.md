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

## Step 2 — Backtest engine (done, pending approval)
Walk-forward backtest harness in `vts/backtest/` + decision schema `vts/decision.py`. Same strategy as
Step 1: a real TradingAgents adapter (guarded) plus a deterministic offline model so the whole engine is
testable without an LLM/API/network.

Key decisions:
1. **Look-ahead-free timing.** At each rebalance date `D` the model decides using only data as-of `D`'s
   close (the as-of clock); the resulting weights earn the **realized** close-to-close return from `D` to
   `D_next` (pure P&L, never a decision input). Costs are charged on turnover at `D` using trailing ADV
   known at `D`. Prices/ADV come from the PIT store, so a decision can never read a future price.
2. **Reflection look-ahead neutralized (Step 0's #1 leak).** `vts/backtest/reflection_gate.py`
   `partition_pending` resolves a prior decision only once its holding window closed
   (`entry_date + holding_days <= trade_date`); `vts/backtest/returns.py::realized_return` scores it from
   the PIT store capped at the as-of clock (replacing TradingAgents' live-yfinance `_fetch_returns`).
   Falsification test written first (`tests/test_reflection_gate.py`).
3. **Anti-contamination gate.** `vts/backtest/cutoff.py` + `model_cutoffs.json`: a decision dated on/before
   the model's knowledge cutoff is flagged `contaminated` (참고용); segments are `clean` only if every date
   is after the cutoff. Cutoff dates are **not fabricated** — the registry ships `null`/unverified, so
   unknown models classify `unknown` (never guessed clean). User fills verified dates before trusting a
   clean segment.
4. **Determinism.** Fixed temperature (Step 1 config), an on-disk `DecisionCache` keyed by
   `(ticker, date, model, prompt-hash, sample)`, and `N=3` majority-vote rating with a **dispersion**
   metric (`vts/decision.py::aggregate_decisions`); ties break to Hold.
5. **Cost model.** commission + sell-side tax + participation slippage (`impact = coef·√(notional/ADV)`,
   capped). All coefficients documented, env/venue-overridable.

Decision schema adopts TradingAgents' 5-tier `Rating` and **adds** `confidence` + structured `evidence`
(the Trading-R1 gap). Metrics suite + benchmark-beat gate are Step 3; Step 2 emits equity curve, per-date
records (weights/ratings/cost/turnover/dispersion/contamination) and per-fold OOS `SegmentReport`s.

Tests: falsification (future ingestion cannot change engine ratings; model decision unaffected by future
bars; reflection gate never resolves an unfinished window), plus cost monotonicity/participation, cutoff
classification, cache determinism, walk-forward tiling, vote/dispersion.

## Step 3 — Evaluation (done, pending approval)
`vts/backtest/{metrics,benchmarks,llm_costs,evaluate}.py` — the mandated metric set, benchmark suite,
LLM cost accounting, and the hard gate.

Key decisions:
1. **Metrics** (`metrics.py`, pure functions, hand-computed tests): total return, CAGR, Sharpe, Sortino,
   MDD, 승률, 평균손익비, 연환산 회전율. Annualization uses actual elapsed calendar time, not bar counts.
   Conventions stated once: MDD positive fraction; flat curve → Sharpe/Sortino 0 (no evidence ≠ infinite
   skill); no-loss window → 손익비 None (undefined, not ∞).
2. **Benchmarks** (`benchmarks.py`): share-based buy&hold and 60/40, both entry-costed and drift-honest
   (no free rebalancing — Step 2 review lesson applied). 60/40's 40% is **cash at configurable
   `rf_annual` (default 0)**, not a fabricated bond return; documented in-module.
3. **Gate in code** (`evaluate.py::_gate`): after-cost total return must **strictly beat every**
   benchmark (buy&hold AND 60/40) or verdict = FAIL. Beat threshold includes a 1e-9 epsilon so
   floating-point noise between compounded and ratio returns can never flip the gate (found by test).
   A PASS on a non-cutoff-clean window is labelled 참고용(오염 가능) and `certifiable=False` — numbers
   can never certify success on a contaminated/unknown window.
4. **LLM cost metrics** (`llm_costs.py`): cache-miss = real call; tracker feeds LLM 호출당 비용 and
   결정 1건당 총 비용 + monthly-budget check ($30 default from Step 1 env).
5. `render_report` emits the side-by-side markdown table (strategy | buy&hold | 60/40) with after-cost
   alpha per benchmark, cost block, contamination label, benchmark-assumption disclosure, and gate verdict.

Adversarial review round 3 (metric-math / gate-fairness / robustness lenses): **14 confirmed findings,
all fixed** with regressions (`tests/test_review_fixes_step3.py`). Highlights:
- HIGH: the final rebalance date's cost was charged internally but never reached the equity curve the
  gate scores → terminal curve point / `final_equity` are now post-cost.
- HIGH: evaluate() divided the tracker's *lifetime* LLM cost by one run's decision count → per-run
  deltas are now snapshotted onto `BacktestResult` and evaluate() reads only those.
- Fairness: benchmark entry cost was absorbed into the curve's first point (cancelled out of
  total_return) → entry point is pre-cost, cost drags later points (matching the strategy convention);
  pre-entry flat padding removed (it diluted benchmark Sharpe/win-rate); frozen basket membership is
  disclosed in the report with a warning when it differs from the universe; strategy idle cash now
  accrues `BacktestConfig.rf_annual` so a nonzero rf no longer credits only the 60/40 cash sleeve.
- Math: CAGR computed in log space (no OverflowError on short windows); curves are truncated at the
  first non-positive equity point so ppy/n_periods/win-rate share one sample; per-period returns clamped
  at -100%; all-flat 손익비 → None; all-negative-curve MDD → 1.0; pstdev convention documented.
- Robustness: schema-valid zero-close bars excluded from share division (engine + benchmarks); budget
  check normalized to a monthly run-rate; tracker pickle/deepcopy-safe.

## Step 4 — Risk layer (done, pending approval)
`vts/risk/` — deterministic code between agent output and any execution; the same `RiskEngine` object
serves backtest / paper / live so the risk path is identical everywhere.

Key decisions:
1. **Limits are frozen constants** (`limits.py::RiskLimits`, `extra="forbid"`), loaded from code
   defaults or `VTS_RISK_*` env vars at startup only. No constructor path accepts agent output — the
   금지사항 ("리스크 한도를 LLM이 결정하게 만드는 설계") is enforced structurally, not by convention.
2. **Decision gate** (`gates.py`): low confidence / low vote agreement / high dispersion / vote tie →
   forced Hold with an audit-trail reason; unparseable agent output → `hold_fallback` (same Hold path).
3. **Halt latch** (`killswitch.py::HaltState`): a period loss ≤ -`daily_loss_limit` latches the halt —
   targets flat, and a good day does NOT unlatch; reset requires an explicit named operator. Env kill
   switch `VTS_KILL_SWITCH` (Step 6 groundwork) is checked fresh on every apply and also latches.
4. **Order pre-validation** (`orders.py`): 잔고 (cash for buys / position for sells, no naked shorts),
   호가단위 (Decimal tick check on price bands — KRX ladder shipped, US flat $0.01), 최소주문금액 + lot
   size. All violations collected, not just the first.
5. **Engine integration**: `Backtester(risk=RiskEngine(...))` replaces naive sizing with
   gate → per-symbol cap → gross cap → halt; `DateRecord` records `halted` / `forced_holds`.

Adversarial review round 4 (bypass / halt-semantics / order-validation): **13 confirmed, all fixed**
(`tests/test_risk_layer.py`, `test_risk_engine_integration.py`):
- HIGH: the "daily" loss limit was actually per-decision-period → at non-daily cadence a real
  single-day crash could be missed and multi-period bleed slip through. The engine now feeds the halt
  **genuine daily marks** (iterates the PIT store's daily bars between decision dates, forward-filled,
  as-of the clock so no look-ahead), plus a companion **cumulative drawdown-from-peak** stop
  (`max_drawdown_limit`, default 20%).
- Kill switch fail-open: any non-empty, non-explicitly-false value now engages (fail-safe); a backtest
  refuses to run under an engaged `VTS_KILL_SWITCH` (no silently-flat sim); a stale halt latch from a
  reused engine raises loudly; `BacktestResult.risk_enabled`/`any_halt` surface un-gated/halted runs.
- Audit trail records distinct concurrent stop reasons (was dropped when already halted).
- Single-sample (N=1) makes agreement/dispersion gates vacuous → opt-in `hold_on_single_sample`.
- Orders: `AccountState` position keys normalized (lowercase broker keys no longer strand a sell);
  tick ladder must be strictly increasing (a duplicate bound silently collapsed a price band);
  fee-blind default cash check documented (live callers must pass a `cash_buffer`).
## Step 5 — Paper trading (done, pending approval)
`vts/paper/` — a forward-running loop on a live clock that reuses the Step 2–4 components verbatim
(`sample_decisions` → `RiskEngine.apply` → `validate_order`); the only thing added over the backtest is
that target weights become concrete integer-share orders passing real pre-trade validation and filling
through a simulated broker.

Key decisions:
1. **Same code path.** The loop calls the exact backtest/risk functions; the weight→order translation +
   broker is the paper-specific delta, and is precisely where implementation shortfall arises.
2. **Dry-run default True** (`PaperState`, `PaperBroker`, `PaperTrader`). `PaperBroker(dry_run=False)`
   raises `LiveTradingNotEnabled` — real routing is Step 6. Resuming a run with a different dry_run flag
   than persisted raises rather than silently switching.
3. **Persistent, resumable state** (`state.py`): cash, share positions, equity curve, decision log,
   shortfall log, processed dates, AND the halt latch are serialized after every step, so an 8-week run
   survives per-day process restarts and a halted run stays halted across a restart. `step()` is
   idempotent on already-processed dates.
4. **Daily implementation shortfall** (`shortfall.py`): `shortfall = backtest_equity − paper_equity`
   (positive = paper underperformed the model) logged every day vs a deterministic reference backtest
   over the same window; per-day and cumulative, in bps of capital.
5. **Live-path risk semantics**: one step = one trading day (일봉 cadence), so each step's return feeds
   the daily-loss halt directly; the kill switch FLATTENS the paper book (the intended live behavior),
   unlike the backtest which raises. Under an engaged kill switch the reference backtest is skipped.

Note: an 8-week real-time run cannot execute in this session; the loop is validated by an offline
fast-forward over 44 daily bars (> 8 trading weeks) asserting daily shortfall logging, resume, and halt
persistence. Tests: 155 passing.
## Step 6 — Live (pending, explicit approval only)
