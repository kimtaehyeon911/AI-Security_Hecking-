"""Walk-forward backtest engine.

Timing contract (the look-ahead-free core): at each rebalance date ``D_i`` the
model decides using only data knowable at ``D_i``'s close (the as-of clock). The
resulting weights then earn the **realized** close-to-close return from ``D_i`` to
``D_{i+1}`` — returns that lie in the future relative to the decision but are pure
P&L accounting, never a decision input. Costs are charged on turnover at ``D_i``
using trailing ADV known at ``D_i``. Prices and ADV come from the point-in-time
store, so a decision can never read a price stamped after its clock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime

from vts.backtest.cache import DecisionCache, decision_key
from vts.backtest.costs import CostModel
from vts.backtest.cutoff import Contamination, CutoffRegistry
from vts.backtest.llm_costs import LLMCostTracker
from vts.backtest.model import DecisionModel
from vts.backtest.splitter import Fold
from vts.decision import AggregatedDecision, aggregate_decisions, rating_to_signed_weight
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore


@dataclass(frozen=True, slots=True)
class BacktestConfig:
    n_samples: int = 3
    initial_capital: float = 100_000.0
    max_gross: float = 1.0        # sum of |weights| ceiling (full risk limits are Step 4)
    long_only: bool = True
    adv_lookback: int = 20
    # Annual risk-free rate accrued on the strategy's UNINVESTED fraction between
    # rebalances. Set it to the same value passed to evaluate(rf_annual=...) so the
    # strategy's idle cash and the 60/40 benchmark's cash sleeve earn the same
    # rate — otherwise a low-exposure strategy is structurally penalized.
    rf_annual: float = 0.0


@dataclass(frozen=True, slots=True)
class DateRecord:
    date: datetime
    equity: float                 # mark-to-market equity carried INTO this date (pre-rebalance)
    cost: float                   # turnover cost charged at this date
    turnover: float               # sum |Δweight|
    weights: dict[str, float]
    ratings: dict[str, str]
    mean_dispersion: float
    mean_agreement: float
    contamination: str


@dataclass
class BacktestResult:
    config: BacktestConfig
    model_id: str
    records: list[DateRecord] = field(default_factory=list)
    benchmark_curve: list[tuple[datetime, float]] = field(default_factory=list)
    # Per-run LLM accounting (delta of the tracker over THIS run only, so a
    # reused Backtester/tracker never double-counts into evaluate()).
    llm_calls: int = 0
    llm_cache_hits: int = 0
    llm_cost_usd: float = 0.0

    @property
    def equity_curve(self) -> list[tuple[datetime, float]]:
        """Equity per date. Each date's turnover cost surfaces in the NEXT record's
        mark-to-market; the final date has no successor, so the terminal point is
        made post-cost here — otherwise the last rebalance's cost would vanish
        from total_return and the gate would compare a partially pre-cost number.
        """
        pts = [(r.date, r.equity) for r in self.records]
        if pts:
            last = self.records[-1]
            pts[-1] = (last.date, last.equity - last.cost)
        return pts

    @property
    def final_equity(self) -> float:
        if not self.records:
            return self.config.initial_capital
        last = self.records[-1]
        return last.equity - last.cost  # post-cost, matching equity_curve's terminal point

    @property
    def total_return(self) -> float:
        return self.final_equity / self.config.initial_capital - 1.0

    @property
    def n_decisions(self) -> int:
        """Aggregated (ticker, date) decisions made — the denominator of 결정 1건당 비용."""
        return sum(len(r.ratings) for r in self.records)


@dataclass(frozen=True, slots=True)
class SegmentReport:
    fold_index: int
    test_start: datetime
    test_end: datetime
    segment_return: float
    contamination: str
    mean_dispersion: float
    mean_agreement: float
    n_dates: int


# --------------------------------------------------------------------- sampling
def sample_decisions(
    model: DecisionModel,
    cache: DecisionCache,
    ticker: str,
    clock: AsOfClock,
    n: int,
    tracker: LLMCostTracker | None = None,
) -> AggregatedDecision:
    """Draw ``n`` cached samples and return the majority-vote aggregate.

    A cache miss is a real model invocation and is recorded on ``tracker`` (a hit
    costs nothing) — this is where the cost-per-call metric gets its counts.
    """
    prompt = model.prompt_for(ticker, clock)
    samples = []
    for i in range(n):
        key = decision_key(ticker, clock.as_of.isoformat(), model.model_id, prompt, i)
        cached = cache.get(key)
        if cached is None:
            cached = model.decide(ticker, clock)
            cache.put(key, cached)
            if tracker is not None:
                tracker.record_call()
        elif tracker is not None:
            tracker.record_cache_hit()
        samples.append(cached)
    return aggregate_decisions(samples)


# ------------------------------------------------------------------ price helpers
def _last_close(store: PointInTimeStore, ticker: str, clock: AsOfClock) -> float | None:
    bars = store.get_ohlcv(ticker, clock)
    return bars[-1].close if bars else None


def _dollar_adv(store: PointInTimeStore, ticker: str, clock: AsOfClock, lookback: int) -> float:
    bars = store.get_ohlcv(ticker, clock)
    if not bars:
        return 0.0
    window = bars[-lookback:]
    return sum(b.close * b.volume for b in window) / len(window)


def _target_weights(
    ratings: dict[str, object], long_only: bool, max_gross: float
) -> dict[str, float]:
    raw = {t: rating_to_signed_weight(agg.rating) for t, agg in ratings.items()}  # type: ignore[attr-defined]
    if long_only:
        raw = {t: max(0.0, w) for t, w in raw.items()}
    gross = sum(abs(w) for w in raw.values())
    if gross > max_gross and gross > 0:
        raw = {t: w * (max_gross / gross) for t, w in raw.items()}
    return raw


class Backtester:
    """Runs a walk-forward backtest with look-ahead-free execution and costs."""

    def __init__(
        self,
        store: PointInTimeStore,
        model: DecisionModel,
        *,
        cost_model: CostModel | None = None,
        cutoff: CutoffRegistry | None = None,
        cache: DecisionCache | None = None,
        config: BacktestConfig | None = None,
        llm_tracker: LLMCostTracker | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.cost_model = cost_model or CostModel()
        self.cutoff = cutoff or CutoffRegistry.load()
        self.cache = cache or DecisionCache()
        self.config = config or BacktestConfig()
        self.llm_tracker = llm_tracker or LLMCostTracker()

    def run(self, universe: list[str], decision_dates: list[datetime]) -> BacktestResult:
        cfg = self.config
        result = BacktestResult(config=cfg, model_id=self.model.model_id)

        # Snapshot the tracker so this run's LLM accounting is a clean delta even
        # when the Backtester (and its tracker) is reused across multiple runs.
        calls0 = self.llm_tracker.calls
        hits0 = self.llm_tracker.cache_hits
        cost0 = self.llm_tracker.total_cost_usd

        equity = cfg.initial_capital
        # `weights` holds the DRIFTED actual weights carried into each date (not the
        # last booked target), so turnover costs reflect the real trades required.
        weights: dict[str, float] = {t: 0.0 for t in universe}
        prev_prices: dict[str, float] | None = None
        prev_date: datetime | None = None

        # True equal-weight BUY&HOLD benchmark: shares bought once and held (weights
        # drift with prices), with a one-time entry cost so the comparison is fair.
        bench_shares: dict[str, float] | None = None
        bench_entry_cost = 0.0

        for d in sorted(decision_dates):
            clock = AsOfClock.at(d)
            prices = {t: _last_close(self.store, t, clock) for t in universe}
            # p > 0, not just present: a schema-valid zero close must not enter
            # share division or return math (consistent with the P&L guard below).
            priced = {t: p for t, p in prices.items() if p is not None and p > 0}

            # 1) Mark-to-market the realized return since the previous rebalance,
            #    then reconcile intra-period drift into the carried weights so the
            #    next turnover calc charges for the real drift-correction trades.
            if prev_prices is not None:
                rets = {
                    t: priced[t] / prev_prices[t] - 1.0
                    for t in weights
                    if t in priced and t in prev_prices and prev_prices[t] > 0
                }
                r = sum(weights.get(t, 0.0) * rt for t, rt in rets.items())
                # Idle cash earns the configured risk-free rate (same rate the
                # 60/40 benchmark's cash sleeve gets — see BacktestConfig.rf_annual).
                if cfg.rf_annual != 0.0 and prev_date is not None:
                    cash_frac = max(0.0, 1.0 - sum(abs(w) for w in weights.values()))
                    dt_years = (d - prev_date).total_seconds() / (365.25 * 24 * 3600)
                    r += cash_frac * ((1.0 + cfg.rf_annual) ** dt_years - 1.0)
                equity *= 1.0 + r
                denom = 1.0 + r
                if denom > 0:
                    weights = {
                        t: weights.get(t, 0.0) * (1.0 + rets.get(t, 0.0)) / denom
                        for t in weights
                    }

            # 2) Decide (N-sample majority vote) using only data as-of the clock.
            aggs = {
                t: sample_decisions(
                    self.model, self.cache, t, clock, cfg.n_samples, self.llm_tracker
                )
                for t in priced
            }
            target = _target_weights(aggs, cfg.long_only, cfg.max_gross)
            target = {t: target.get(t, 0.0) for t in universe}

            # 3) Charge turnover costs (drifted -> target) using ADV known at the clock.
            carried_in = equity
            cost_total = 0.0
            turnover = 0.0
            for t in universe:
                dw = target[t] - weights.get(t, 0.0)
                if dw == 0.0 or t not in priced:
                    continue
                turnover += abs(dw)
                notional = abs(dw) * equity
                adv = _dollar_adv(self.store, t, clock, cfg.adv_lookback)
                side = "buy" if dw > 0 else "sell"
                cost_total += self.cost_model.cost(notional, adv, side).total
            equity -= cost_total

            # Benchmark buys equal-weight shares on the first priced date and holds.
            bench_entered_now = False
            if bench_shares is None and priced:
                alloc = cfg.initial_capital / len(priced)
                bench_shares = {t: alloc / priced[t] for t in priced}
                bench_entry_cost = sum(
                    self.cost_model.cost(
                        alloc, _dollar_adv(self.store, t, clock, cfg.adv_lookback), "buy"
                    ).total
                    for t in priced
                )
                bench_entered_now = True
            if bench_shares:
                if bench_entered_now:
                    # Entry point is PRE-cost capital (same convention as the
                    # strategy's first curve point); subtracting the entry cost
                    # here would cancel it out of last/first total_return.
                    bench_equity = cfg.initial_capital
                else:
                    bench_equity = sum(
                        bench_shares[t] * priced[t] for t in bench_shares if t in priced
                    ) - bench_entry_cost
            else:
                bench_equity = cfg.initial_capital

            contamination = self.cutoff.segment_status(self.model.model_id, [d]).value
            disp = [a.dispersion for a in aggs.values()]
            agr = [a.agreement for a in aggs.values()]
            result.records.append(
                DateRecord(
                    date=d,
                    equity=carried_in,  # mark-to-market equity carried in (before costs)
                    cost=cost_total,
                    turnover=turnover,
                    weights=dict(target),
                    ratings={t: a.rating.value for t, a in aggs.items()},
                    mean_dispersion=sum(disp) / len(disp) if disp else 0.0,
                    mean_agreement=sum(agr) / len(agr) if agr else 0.0,
                    contamination=contamination,
                )
            )
            result.benchmark_curve.append((d, bench_equity))

            weights = target
            prev_prices = priced or prev_prices
            prev_date = d

        result.llm_calls = self.llm_tracker.calls - calls0
        result.llm_cache_hits = self.llm_tracker.cache_hits - hits0
        result.llm_cost_usd = self.llm_tracker.total_cost_usd - cost0
        return result

    def segment_reports(self, result: BacktestResult, folds: list[Fold]) -> list[SegmentReport]:
        """Per-fold OOS reports, each tagged with its contamination status."""
        by_date = {r.date: r for r in result.records}
        reports: list[SegmentReport] = []
        for fold in folds:
            test_recs = [by_date[d] for d in fold.test if d in by_date]
            if len(test_recs) < 2:
                continue
            seg_ret = test_recs[-1].equity / test_recs[0].equity - 1.0
            status = self.cutoff.segment_status(result.model_id, [r.date for r in test_recs]).value
            disp = [r.mean_dispersion for r in test_recs]
            agr = [r.mean_agreement for r in test_recs]
            reports.append(
                SegmentReport(
                    fold_index=fold.index,
                    test_start=fold.test_start,
                    test_end=fold.test_end,
                    segment_return=seg_ret,
                    contamination=status,
                    mean_dispersion=sum(disp) / len(disp),
                    mean_agreement=sum(agr) / len(agr),
                    n_dates=len(test_recs),
                )
            )
        return reports
