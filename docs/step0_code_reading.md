# Step 0 — Code Reading: TradingAgents structure, data adapters & LLM call sites

> Purpose: map `tradingagents/{agents,dataflows,graph}` (+ `llm_clients`) so we know the
> **data-source adapter interface**, every **LLM call site**, and **exactly which files to change**
> to attach a new asset class (KR stocks / US stocks / crypto) for a *verifiable* trading system.
>
> Base: fork of [TauricResearch/TradingAgents](https://github.com/TauricResearch/TradingAgents) (Apache-2.0).
> Papers: arXiv 2412.20138 (TradingAgents), arXiv 2509.11420 (Trading-R1).
> Analysis performed against the upstream `main` at read time (commit cloned into scratchpad).

---

## 0. TL;DR for the impatient

- The fork is **already well past the paper**: yfinance is one vendor among several behind a
  **config-driven vendor-dispatch layer** (`VENDOR_METHODS` in `dataflows/interface.py`). Adding a
  broker/exchange = **register a vendor function + set a config key**, *not* rewriting call sites.
- The decision output is **already Trading-R1's 5-tier rating** (`Buy/Overweight/Hold/Underweight/Sell`,
  `schemas.PortfolioRating`) with an evidence-anchored `investment_thesis`. Adoption work is **small**.
- **Two hard yfinance couplings bypass the dispatch layer** and must be cut for a non-US / point-in-time
  build: `get_verified_market_snapshot` → `load_ohlcv`, and `resolve_instrument_identity`. Plus a
  module-level `import yfinance` in `graph/trading_graph.py` used by `_fetch_returns`.
- **The single biggest backtest hazard is not the data layer — it is deferred reflection**
  (`propagate → _resolve_pending_entries → _fetch_returns`), which pulls **realized forward returns via
  live yfinance** and folds them back into later decisions as `past_context`. A naive chronological
  replay leaks the future. **Write the falsification test for this first (Step 2).**
- Existing point-in-time guards are **"download-now-then-filter"**, not true point-in-time:
  `load_ohlcv` uses `auto_adjust=True` (split/dividend back-adjustment = restatement leak) and
  yfinance news `get_news` can only return *currently surfaced* headlines, so it **cannot serve
  historical news** for a backtest date in the past. This is why Step 1 replaces the data layer.
- **No knowledge-cutoff dates** are encoded anywhere (`model_catalog.py`), and there is **no seed /
  no response cache** for `(ticker, date)` calls. Both are gaps we must fill for Steps 2–3.

---

## 1. Package map (what each subsystem does)

```
tradingagents/
├── agents/                 # LLM agent nodes + their LangChain tools + decision schemas
│   ├── analysts/           #   market, news, sentiment, social_media, fundamentals
│   ├── researchers/        #   bull, bear  (debate)
│   ├── managers/           #   research_manager, portfolio_manager  ← the 2 DEEP-model nodes
│   ├── risk_mgmt/          #   aggressive / conservative / neutral debators
│   ├── trader/trader.py    #   turns research plan into a 3-tier Buy/Hold/Sell proposal
│   ├── schemas.py          #   ★ Pydantic decision schemas (5-tier rating, thesis)
│   └── utils/              #   tool wrappers, rating parser, structured-output plumbing, memory
├── dataflows/              # ★ Data-source adapter layer (vendor dispatch)
│   ├── interface.py        #   ★ VENDOR_METHODS registry + route_to_vendor
│   ├── config.py           #   get_config/set_config (thread-local active config)
│   ├── default? (see default_config.py at package root)
│   ├── y_finance.py, yfinance_news.py, stockstats_utils.py   # yfinance vendor impls
│   ├── alpha_vantage*.py   #   alpha_vantage vendor impls
│   ├── fred.py             #   macro (FRED)
│   ├── polymarket.py       #   prediction markets
│   ├── reddit.py, stocktwits.py  # social sources
│   ├── symbol_utils.py     #   ★ symbol normalization (Yahoo-centric)
│   └── market_data_validator.py  # deterministic OHLCV verification snapshot
├── graph/                  # LangGraph orchestration
│   ├── trading_graph.py    #   ★ TradingAgentsGraph.propagate(ticker, date)  ← backtest entry atom
│   ├── setup.py            #   node/edge assembly; assigns quick vs deep model per node
│   ├── propagation.py      #   create_initial_state(company, trade_date, asset_type)
│   ├── signal_processing.py#   parse 5-tier rating from PM markdown (NO LLM call anymore)
│   ├── conditional_logic.py#   ReAct loop + debate/risk round caps
│   └── reflection.py       #   ★ LLM lesson from realized returns (lookahead source)
├── llm_clients/            # provider clients + factory + model catalog
│   ├── factory.py          #   create_llm_client(provider, model, ...)
│   ├── openai_client.py     #   OpenAI-compatible family (temperature threaded here)
│   ├── anthropic_client.py, google_client.py, azure_client.py, bedrock_client.py
│   ├── model_catalog.py    #   ★ known model IDs — NO knowledge-cutoff dates
│   └── capabilities.py, validators.py, api_key_env.py
├── default_config.py       # ★ DEFAULT_CONFIG: llm/model, data_vendors, benchmark_map
└── reporting.py            # writes report tree to disk
```

---

## 2. Data-source adapter interface

### 2.1 How dispatch works (the seam we plug a broker into)

```
agent tool  (@tool get_stock_data)          # agents/utils/*_tools.py  — thin, vendor-agnostic
   └─ dataflows.interface.route_to_vendor("get_stock_data", ...)
        └─ VENDOR_METHODS["get_stock_data"] = {           # dataflows/interface.py
               "alpha_vantage": get_alpha_vantage_stock,
               "yfinance":      get_YFin_data_online,
           }
        └─ vendor chosen by config: data_vendors[category]  or  tool_vendors[method]  (default_config.py)
```

- The configured vendor list **is** the fallback chain — there is **no silent routing** to vendors you
  did not choose. `"default"` uses all registered vendors in insertion order.
- `macro_data` and `prediction_markets` are `OPTIONAL_CATEGORIES` → **fail soft** (sentinel string on
  failure). Core categories (prices / fundamentals / news) **raise loudly**.
- **Adding a vendor = one row in `VENDOR_METHODS` + one config key.** The 6 tool wrapper files do NOT
  change on a vendor swap.

### 2.2 Adapter / tool inventory

| Tool (`@tool`) | File | Backing vendor(s) | yfinance-dependent | Routed? | Returns |
|---|---|---|---|---|---|
| `get_stock_data(symbol, start, end)` | `agents/utils/core_stock_tools.py` | `yfinance`→`get_YFin_data_online`; `alpha_vantage` | ✅ (default) | ✅ | str (OHLCV) |
| `get_indicators(symbol, indicator, curr_date, look_back_days=30)` | `agents/utils/technical_indicators_tools.py` | `yfinance`→stockstats; `alpha_vantage` | ✅ | ✅ | str |
| `get_fundamentals(ticker, curr_date)` | `agents/utils/fundamental_data_tools.py` | `yfinance`; `alpha_vantage` | ✅ | ✅ | str |
| `get_balance_sheet / get_cashflow / get_income_statement(ticker, freq, curr_date)` | `agents/utils/fundamental_data_tools.py` | `yfinance`; `alpha_vantage` | ✅ | ✅ | str |
| `get_news(ticker, start, end)` | `agents/utils/news_data_tools.py` | `yfinance`→`get_news_yfinance`; `alpha_vantage` | ✅ | ✅ | str |
| `get_global_news(curr_date, look_back_days, limit)` | `agents/utils/news_data_tools.py` | `yfinance`; `alpha_vantage` | ✅ | ✅ | str |
| `get_insider_transactions(ticker)` | `agents/utils/news_data_tools.py` | `yfinance`; `alpha_vantage` | ✅ | ✅ | str |
| `get_macro_indicators(indicator, curr_date, look_back_days)` | `agents/utils/macro_data_tools.py` | **`fred` only** | ❌ | ✅ | str |
| `get_prediction_markets(topic, limit)` | `agents/utils/prediction_markets_tools.py` | **`polymarket` only** | ❌ | ✅ | str |
| `get_verified_market_snapshot(symbol, curr_date, look_back_days=30)` | `agents/utils/market_data_validation_tools.py` | **hard-locked yfinance** via `market_data_validator → stockstats_utils.load_ohlcv` | ✅ | ❌ **bypasses dispatch** | str |
| `resolve_instrument_identity(ticker)` *(not a `@tool`)* | `agents/utils/agent_utils.py` | **hard-locked** `yf.Ticker(...).info` | ✅ | ❌ **bypasses dispatch** | dict |

**Two couplings that ignore `data_vendors` and must be cut for a non-US / point-in-time build:**
1. `get_verified_market_snapshot` → `dataflows/market_data_validator.py` → `dataflows/stockstats_utils.py::load_ohlcv` (`yf.download`, `auto_adjust=True`).
2. `resolve_instrument_identity` (`agent_utils.py`) → `yf.Ticker(normalize_symbol(ticker)).info` (fails open to `{}`).
3. (bonus) `graph/trading_graph.py` has a **module-level `import yfinance as yf`** used by `_fetch_returns`.

### 2.3 Symbol handling (`dataflows/symbol_utils.py`)

`normalize_symbol` is **Yahoo-centric**: crypto `BTCUSD→BTC-USD`, forex `EURUSD→EURUSD=X`, metals/index CFD
alias table (`XAUUSD→GC=F`, `SPX500→^GSPC`). **KR equities (`.KS`/`.KQ`) are not first-classed** — they
ride the plain-equity passthrough and rely on yfinance recognizing the suffix.

---

## 3. LLM call sites

Two model roles are configured (`default_config.py`): `deep_think_llm` (default `gpt-5.5`) and
`quick_think_llm` (default `gpt-5.4-mini`). **Only 2 nodes use the deep model.** Node→role assignment is
**hardcoded in `graph/setup.py`**, not config-driven.

| # | Node / call site | File | Model role | Tool-calling? | Purpose |
|---|---|---|---|---|---|
| 1 | Market Analyst | `agents/analysts/market_analyst.py` (wired `setup.py:77`) | quick | ✅ ReAct | technical/price report |
| 2 | Sentiment Analyst | `agents/analysts/sentiment_analyst.py` | quick | ✅ | `SentimentReport` (6-tier band + 0–10 + confidence) |
| 3 | News Analyst | `agents/analysts/news_analyst.py` | quick | ✅ | news report |
| 4 | Social Media Analyst | `agents/analysts/social_media_analyst.py` | quick | ✅ | social report |
| 5 | Fundamentals Analyst | `agents/analysts/fundamentals_analyst.py` | quick | ✅ | fundamentals report |
| 6 | Bull Researcher | `agents/researchers/bull_researcher.py` (`setup.py:83`) | quick | ❌ | debate |
| 7 | Bear Researcher | `agents/researchers/bear_researcher.py` (`setup.py:84`) | quick | ❌ | debate |
| 8 | **Research Manager** | `agents/managers/research_manager.py` (`setup.py:85`) | **deep** | ❌ | `ResearchPlan` (5-tier recommendation) |
| 9 | Trader | `agents/trader/trader.py` (`setup.py:86`) | quick | ❌ | `TraderProposal` (3-tier Buy/Hold/Sell) |
| 10–12 | Aggressive / Conservative / Neutral risk debators | `agents/risk_mgmt/*.py` (`setup.py:89-91`) | quick | ❌ | risk debate |
| 13 | **Portfolio Manager** | `agents/managers/portfolio_manager.py` (`setup.py:92`) | **deep** | ❌ | **`PortfolioDecision` = final trade decision** |
| 14 | Reflector | `graph/reflection.py:57` `self.quick_thinking_llm.invoke(...)` | quick | ❌ | lesson from realized returns → memory log |
| — | SignalProcessor | `graph/signal_processing.py:31` | *(none)* | — | **deterministic** `parse_rating` — **no LLM call** |

Structured output for the 4 structured agents (Research Mgr, Portfolio Mgr, Trader, Sentiment) is bound in
`agents/utils/structured.py` (`bind_structured` = `with_structured_output`; `invoke_structured_or_freetext`
does one free-text retry). Enforcement is **best-effort**: on failure it falls back to prose and downstream
still relies on `parse_rating` over the rendered markdown.

**Per-decision LLM budget (single run):** ~13 agent invocations (11 quick + 2 deep) + 1 reflection, plus
analyst tool-calling loops (each analyst may make several tool round-trips). Cost per `(ticker, date)` in
Step 3 must count all of these, not just the final decision.

---

## 4. Decision schema — Trading-R1 adoption status

`agents/schemas.py` **already implements the target 5-tier scale**:

- `PortfolioRating(str, Enum)` = `Buy / Overweight / Hold / Underweight / Sell` — used by
  `ResearchPlan.recommendation` and `PortfolioDecision.rating`.
- `PortfolioDecision` fields: `rating`, `executive_summary`, `investment_thesis`
  ("Detailed reasoning anchored in specific evidence…"), optional `price_target`, `time_horizon`.
- `TraderAction` = 3-tier `Buy/Hold/Sell` (sizing/Overweight/Underweight deliberately deferred to PM).
- Canonical scale + heuristic parser centralized in `agents/utils/rating.py` (`RATINGS_5_TIER`,
  `parse_rating(default="Hold")`), reused by signal processor and memory log.
- Currency-agnostic: price fields say "in the instrument's quote currency" — **KRW / crypto validate as-is.**

**Gaps vs Trading-R1's "evidence-based thesis":**
- No **structured evidence list** — evidence is free-text embedded inside `investment_thesis`.
- **No confidence field on the decision schemas** (only `SentimentReport.confidence` exists). Step 4 needs a
  decision-level confidence to implement "low confidence → force Hold", so we will **add** one.

---

## 5. Point-in-time / look-ahead audit (Step 1 & Step 2 relevant)

**Existing guards (good, but not sufficient):**
- `stockstats_utils.load_ohlcv`: filters `data[Date <= curr_date]`; rejects stale frames
  (`_assert_ohlcv_not_stale`, >10 days old); TTL-refreshes same-day partial candles.
- `filter_financials_by_date`: drops fiscal columns after `curr_date`.
- `yfinance_news._in_news_window`: half-open `[start, end+1day)`, UTC-normalized; **undated articles
  excluded** in a historical window (won't leak future news).

**Why these are still not point-in-time (the Step 1 problem):**
- `load_ohlcv` downloads **current** 5y data with `auto_adjust=True`, then filters by date. Auto-adjust
  **back-adjusts historical prices for later splits/dividends** → a restatement leak (the price you "see"
  on date D depends on events after D). There is **no `as_of` field** stored.
- `get_news` calls `yf.Ticker(...).get_news()` which returns **only currently surfaced** headlines. For a
  backtest date months/years in the past it returns **"No news found"** — the layer structurally cannot
  reconstruct the historical news set. yfinance news is unusable for historical backtests.
- Nothing carries an `as_of` timestamp; nothing raises on future access.
  → **Step 1 must build a point-in-time store with `as_of` on every record and an
  `assert_no_lookahead` guard in the data-access layer.**

**★ The dominant hazard — deferred reflection leaks realized forward returns:**
```
TradingAgentsGraph.propagate(company, trade_date)      # graph/trading_graph.py:362
  → _resolve_pending_entries(company)                  # :375  (runs BEFORE the pipeline)
      → _fetch_returns(ticker, entry.date,             # :251  LIVE yfinance, window
                       holding_days=5)                  #        [entry.date, entry.date+holding+7]
      → reflector.reflect_on_final_decision(...)        # LLM lesson from raw+alpha return
      → memory_log.batch_update_with_outcomes(...)
  → past_context = memory_log.get_past_context(company) # :423  injected into THIS decision
```
In a chronological replay, running date **D2** resolves the decision made on **D1** using returns realized
over `[D1, D1+holding+7]` — which can extend **past D2** — and folds that hindsight into D2's decision.
`_fetch_returns` has **no cap at the simulated `trade_date`**.

**Backtest harness (Step 2) must therefore:**
1. Gate resolution: only resolve entries with `entry.date + holding_days <= trade_date`.
2. Route `_fetch_returns` through the point-in-time store capped at the simulated `as_of`, not live yfinance.
3. Ensure `get_past_context` surfaces only reflections resolved strictly before `trade_date`.
4. **Write a falsification test first**: shuffle/blank future data and assert the decision is unchanged;
   assert no data access with `as_of > trade_date`.

---

## 6. Determinism audit (Step 2 relevant)

- **Temperature is threaded** to OpenAI-compatible clients via `_PASSTHROUGH_KWARGS`
  (`openai_client.py:166`); `config["temperature"]` / `TRADINGAGENTS_TEMPERATURE` set it.
- **No `seed`** parameter is forwarded anywhere.
- **No response cache** keyed on `(ticker, date, prompt)` — repeated runs re-hit the API.
- Reasoning models largely ignore temperature; even `temperature=0` is not bit-identical run-to-run.
  → **Step 2 must add**: fixed temperature, an on-disk response cache keyed by `(ticker, date, model,
  prompt-hash)`, and `N=3` runs with **majority-vote rating + rating-dispersion metric**.

---

## 7. Knowledge-cutoff gate (Step 2 — anti-contamination)

`llm_clients/model_catalog.py` lists provider model IDs (e.g. `gpt-5.5`, `claude-fable-5`,
`deepseek-v4-pro`) but **encodes no knowledge-cutoff dates**. To honor "backtest window must be **after** the
model's knowledge cutoff", we must **add a cutoff registry** (`{model_id: cutoff_date}`) and have the
backtest engine (a) refuse / flag any test window that starts before the selected model's cutoff, and
(b) label pre-cutoff results as **"참고용(오염 가능) / reference-only, possibly contaminated"**.

---

## 8. Integration points — "which files do I change to attach my asset class?"

`asset_type` currently branches only `"stock"` vs `"crypto"` (`agent_utils.build_instrument_context`,
`propagation.create_initial_state`, checkpoint signature). Below is the concrete change list **per asset
class**. Files are the *minimum* set; ✅ = new code, ✎ = edit.

### 8.1 Common to any new venue (US / KR / crypto)
| Concern | File | Change |
|---|---|---|
| Register the vendor | `dataflows/interface.py` (`VENDOR_METHODS`) | ✎ add `{method: {"<broker>": impl_fn}}` for each method you serve |
| New vendor implementation module | `dataflows/<broker>.py` (e.g. `kis.py`, `binance.py`) | ✅ implement `get_stock_data / get_indicators / get_fundamentals / get_news …` returning the same shapes |
| Select the vendor | `default_config.py` (`data_vendors` / `tool_vendors`) | ✎ point categories at `<broker>` |
| Point-in-time store + guard (Step 1) | `dataflows/pit_store.py`, `dataflows/guards.py` | ✅ `as_of` on every record; `assert_no_lookahead` |
| Cut the 2 hard yfinance couplings | `agents/utils/market_data_validation_tools.py` → `dataflows/market_data_validator.py` / `stockstats_utils.py`; `agents/utils/agent_utils.py::resolve_instrument_identity` | ✎ route through the store / broker instead of `yf.*` |
| Realized-return fetch | `graph/trading_graph.py::_fetch_returns` (+ module `import yfinance`) | ✎ route through the point-in-time store, capped at `as_of` |
| Benchmark for alpha | `default_config.py` (`benchmark_map`) | ✎ add the venue's index (see below) |

### 8.2 US stocks (Binance N/A) — lightest lift
Mostly works today (yfinance/alpha_vantage). Real work is only **Step 1's point-in-time replacement** and
cutting the hard couplings; symbols and `benchmark_map` (`'' → SPY`) already fit.

### 8.3 KR stocks (키움 REST / KIS Open API)
| Concern | File | Change |
|---|---|---|
| Symbol suffix | `dataflows/symbol_utils.py` | ✎ first-class `.KS` (KOSPI) / `.KQ` (KOSDAQ); add to instrument-context suffix examples in `agent_utils.build_instrument_context` |
| Benchmark | `default_config.py` (`benchmark_map`) | ✎ `.KS → ^KS11` (KOSPI), `.KQ → ^KQ11` (KOSDAQ) — currently **missing → wrongly defaults to SPY** |
| Macro | `agents/utils/macro_data_tools.py` + new `dataflows/bok.py` | ✎ US/FRED aliases only today; add BOK/ECOS vendor + KR aliases (base rate, CPI, KRW) |
| Broker data vendor | `dataflows/kis.py` (or `kiwoom.py`) | ✅ OHLCV/fundamentals/news via broker REST; KRW quote currency (schema already agnostic) |
| Trading calendar / hours | new `dataflows/calendars.py` | ✅ KRX sessions, tick size, lot rules (needed for Step 4 order pre-validation) |

### 8.4 Crypto spot (Binance)
| Concern | File | Change |
|---|---|---|
| `asset_type="crypto"` path | already exists (`build_instrument_context`, `propagate`) | ✎ verify it suppresses equity-only tools (fundamentals/insider) |
| Symbol | `dataflows/symbol_utils.py` | crypto rule exists (`BTCUSD→BTC-USD`); ✎ add USDT-pair mapping to Binance symbols |
| Benchmark | `default_config.py` | ✎ set explicit `benchmark_ticker` (e.g. `BTC-USD`) — crypto has no dotted suffix |
| Exchange data vendor | `dataflows/binance.py` | ✅ klines (OHLCV), 24h stats; **no fundamentals** → fundamentals analyst should be dropped or repurposed |
| 24/7 sessions | `dataflows/calendars.py` | ✅ no market-hours gate; funding/maker-taker fee model for Step 2 cost model |

---

## 9. Backtest entry point (Step 2 hook)

`TradingAgentsGraph.propagate(company_name, trade_date, asset_type="stock")` is the **atom of work**:
returns `(final_state, parsed_signal)` where `parsed_signal ∈ {Buy, Overweight, Hold, Underweight, Sell}`.
`main.py` shows the pattern (`ta.propagate("NVDA", "2024-05-10")`). The Step 2 harness constructs the graph
once and calls `propagate` per `(ticker, date)` in **chronological** order, wrapping it with: the reflection
gate (§5), the response cache + N=3 vote (§6), and the cost model (Step 2).

---

## 10. Hardcoded US assumptions to revisit (non-exhaustive)

- `benchmark_map` default `SPY`; no Korea entries.
- Default models OpenAI `gpt-5.5` / `gpt-5.4-mini`.
- `global_news_queries` are Fed/S&P-centric (`default_config.py`).
- `macro_data_tools` aliases entirely FRED/US; `prediction_markets` Polymarket + US-centric example topics.
- Alpha computed in USD.

---

## 11. Decisions taken in Step 0

1. **Keep the vendor-dispatch architecture** — it is the correct seam; we extend it, not replace it.
2. **Adopt the existing `PortfolioRating` 5-tier schema** as the Trading-R1 decision output; **add** a
   decision-level `confidence` and a structured `evidence` list (Step 2/4), rather than inventing a new enum.
3. **Replace the data layer with a point-in-time store** (`as_of` + `assert_no_lookahead`) in Step 1; treat
   `load_ohlcv`/yfinance-news as non-authoritative for history.
4. **Neutralize deferred-reflection lookahead in the backtest harness** and cover it with a falsification
   test **before** trusting any performance number (Step 2).
5. **Add a model knowledge-cutoff registry** and gate the backtest window on it (Step 2).

## 12. Open inputs blocking Step 1 (need user decision)

Environment placeholders are unfilled. Step 1's concrete file set depends on:
- **Asset class**: KR stocks / US stocks / crypto spot.
- **Broker/exchange API**: 키움 REST / KIS Open API / Binance.
- **LLM provider + monthly API budget** (drives model choice, cutoff gate, and the Step 3 cost gate).
- **Cadence**: daily bar (1/day) / 4-hour bar.
