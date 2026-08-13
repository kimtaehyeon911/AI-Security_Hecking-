"""Step 3 evaluation: mandated metrics vs benchmarks, LLM costs, and the gate.

The gate implements the mandate verbatim: *"비용 차감 후 벤치마크를 못 이기면
실패"* — the strategy's AFTER-COST total return must strictly beat EVERY listed
benchmark over the same window, or the evaluation verdict is FAIL. There is no
soft pass. Additionally, a verdict is only *certifiable* when every decision date
is knowledge-cutoff-clean; contaminated/unknown windows are labelled
``참고용(오염 가능)`` and can never certify a success, no matter the numbers.
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict

from vts.backtest.benchmarks import buy_and_hold_curve, entry_basket, sixty_forty_curve
from vts.backtest.costs import CostModel
from vts.backtest.cutoff import Contamination
from vts.backtest.engine import Backtester, BacktestResult
from vts.backtest.metrics import PerformanceMetrics, compute_metrics
from vts.pit.store import PointInTimeStore


# A strategy must beat a benchmark by more than floating-point noise: compounding
# per-period returns vs a single end/start ratio differs by ~1 ulp, and a strict
# `>` would let that noise decide the gate. 1e-9 is far below any real edge.
_BEAT_EPS = 1e-9


class BenchmarkCheck(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    strategy_return: float
    benchmark_return: float
    after_cost_alpha: float          # strategy CAGR - benchmark CAGR
    beat: bool


class GateResult(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    passed: bool                      # beat EVERY benchmark after costs
    certifiable: bool                 # every decision date cutoff-clean
    contamination: str                # clean | contaminated | unknown
    checks: tuple[BenchmarkCheck, ...]
    verdict: str                      # human-readable one-liner


class CostReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    llm_calls: int
    cache_hits: int
    total_cost_usd: float
    cost_per_call_usd: float | None       # LLM 호출당 비용
    cost_per_decision_usd: float | None   # 결정 1건당 총 비용
    n_decisions: int
    monthly_budget_usd: float | None
    # Budget semantics are per-month, so the check normalizes total window cost to
    # a monthly run-rate (window floored at one day to avoid a zero divisor).
    monthly_run_rate_usd: float | None
    within_budget: bool | None


class EvaluationReport(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    model_id: str
    period_start: datetime
    period_end: datetime
    strategy: PerformanceMetrics
    benchmarks: dict[str, PerformanceMetrics]
    gate: GateResult
    costs: CostReport
    # Benchmark assumptions, disclosed so the rendered report is self-describing.
    rf_annual: float
    benchmark_basket: tuple[str, ...]
    basket_entry_date: datetime | None
    universe: tuple[str, ...]


def _gate(
    result: BacktestResult,
    strategy: PerformanceMetrics,
    benchmarks: dict[str, PerformanceMetrics],
) -> GateResult:
    statuses = {r.contamination for r in result.records}
    if Contamination.CONTAMINATED.value in statuses:
        contamination = Contamination.CONTAMINATED.value
    elif Contamination.UNKNOWN.value in statuses:
        contamination = Contamination.UNKNOWN.value
    else:
        contamination = Contamination.CLEAN.value
    certifiable = contamination == Contamination.CLEAN.value

    checks = tuple(
        BenchmarkCheck(
            name=name,
            strategy_return=strategy.total_return,
            benchmark_return=m.total_return,
            after_cost_alpha=strategy.cagr - m.cagr,
            beat=strategy.total_return > m.total_return + _BEAT_EPS,
        )
        for name, m in benchmarks.items()
    )
    passed = bool(checks) and all(c.beat for c in checks)

    if not passed:
        losing = ", ".join(c.name for c in checks if not c.beat) or "no benchmarks"
        verdict = f"FAIL — 비용 차감 후 벤치마크를 못 이김 ({losing})"
    elif not certifiable:
        verdict = (
            "PASS (참고용/오염 가능) — beats all benchmarks after costs, but the "
            f"window is '{contamination}' vs the model's knowledge cutoff; "
            "not certifiable as clean"
        )
    else:
        verdict = "PASS — beats all benchmarks after costs on a cutoff-clean window"

    return GateResult(
        passed=passed,
        certifiable=certifiable,
        contamination=contamination,
        checks=checks,
        verdict=verdict,
    )


def evaluate(
    backtester: Backtester,
    result: BacktestResult,
    universe: list[str],
    *,
    rf_annual: float = 0.0,
    monthly_budget_usd: float | None = None,
) -> EvaluationReport:
    """Build the full Step 3 report for one backtest run."""
    if not result.records:
        raise ValueError("cannot evaluate an empty backtest result")

    store: PointInTimeStore = backtester.store
    cost_model: CostModel = backtester.cost_model
    dates = [r.date for r in result.records]
    capital = result.config.initial_capital

    strategy_curve = result.equity_curve
    turnovers = [r.turnover for r in result.records]
    strategy = compute_metrics(strategy_curve, turnover_per_period=turnovers, rf_annual=rf_annual)

    bench_curves = {
        "buy&hold": buy_and_hold_curve(
            store, universe, dates, capital=capital, cost_model=cost_model,
            adv_lookback=result.config.adv_lookback,
        ),
        "60/40": sixty_forty_curve(
            store, universe, dates, capital=capital, rf_annual=rf_annual,
            cost_model=cost_model, adv_lookback=result.config.adv_lookback,
        ),
    }
    benchmarks = {name: compute_metrics(c, rf_annual=rf_annual) for name, c in bench_curves.items()}

    # Per-run LLM accounting from the result itself — NOT the tracker's lifetime
    # totals, which accumulate across runs when a Backtester is reused.
    n_dec = result.n_decisions
    total_cost = result.llm_cost_usd
    window_days = max((dates[-1] - dates[0]).total_seconds() / 86400.0, 1.0)
    months = window_days / 30.4375
    run_rate = total_cost / months
    costs = CostReport(
        llm_calls=result.llm_calls,
        cache_hits=result.llm_cache_hits,
        total_cost_usd=total_cost,
        cost_per_call_usd=(total_cost / result.llm_calls) if result.llm_calls else None,
        cost_per_decision_usd=(total_cost / n_dec) if n_dec else None,
        n_decisions=n_dec,
        monthly_budget_usd=monthly_budget_usd,
        monthly_run_rate_usd=None if monthly_budget_usd is None else run_rate,
        within_budget=(None if monthly_budget_usd is None else run_rate <= monthly_budget_usd),
    )

    basket, basket_entry = entry_basket(store, universe, dates)

    return EvaluationReport(
        model_id=result.model_id,
        period_start=dates[0],
        period_end=dates[-1],
        strategy=strategy,
        benchmarks=benchmarks,
        gate=_gate(result, strategy, benchmarks),
        costs=costs,
        rf_annual=rf_annual,
        benchmark_basket=tuple(basket),
        basket_entry_date=basket_entry,
        universe=tuple(sorted(t.upper() for t in universe)),
    )


def _fmt(x: float | None, pct: bool = False) -> str:
    if x is None:
        return "n/a"
    return f"{x:+.2%}" if pct else f"{x:.3f}"


def render_report(report: EvaluationReport) -> str:
    """Render the mandated side-by-side table as markdown."""
    cols = ["strategy", *report.benchmarks.keys()]
    all_metrics: dict[str, PerformanceMetrics] = {"strategy": report.strategy, **report.benchmarks}

    rows = [
        ("Total return", lambda m: _fmt(m.total_return, pct=True)),
        ("CAGR", lambda m: _fmt(m.cagr, pct=True)),
        ("Sharpe", lambda m: _fmt(m.sharpe)),
        ("Sortino", lambda m: _fmt(m.sortino)),
        ("Max drawdown", lambda m: _fmt(m.max_drawdown, pct=True)),
        ("Win rate", lambda m: _fmt(m.win_rate, pct=True)),
        ("Avg win/loss", lambda m: _fmt(m.avg_win_loss_ratio)),
        ("Annual turnover", lambda m: _fmt(m.annual_turnover)),
    ]

    basket = ", ".join(report.benchmark_basket) or "(empty — universe never priced)"
    missing = sorted(set(report.universe) - set(report.benchmark_basket))
    lines = [
        f"# Evaluation — {report.model_id}",
        f"Period: {report.period_start.date()} → {report.period_end.date()}  ",
        f"Contamination: **{report.gate.contamination}**",
        "",
        "Benchmark assumptions: 60/40 = 60% equal-weight equity basket + 40% **cash** at "
        f"rf_annual={report.rf_annual:.2%} (cash proxy, not bonds). "
        f"Basket (frozen at entry{f', {report.basket_entry_date.date()}' if report.basket_entry_date else ''}): {basket}."
        + (
            f" ⚠ universe tickers never in the basket (strategy-only opportunity set): {', '.join(missing)}."
            if missing
            else ""
        ),
        "",
        "| metric | " + " | ".join(cols) + " |",
        "|---|" + "---|" * len(cols),
    ]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(all_metrics[c]) for c in cols) + " |")

    lines += ["", "## After-cost alpha (CAGR vs benchmark)"]
    for c in report.gate.checks:
        lines.append(f"- vs {c.name}: {_fmt(c.after_cost_alpha, pct=True)} ({'beat' if c.beat else 'LOST'})")

    cost = report.costs
    lines += [
        "",
        "## LLM cost",
        f"- calls: {cost.llm_calls} (cache hits: {cost.cache_hits})",
        f"- total: ${cost.total_cost_usd:.4f}",
        f"- LLM 호출당 비용: " + (f"${cost.cost_per_call_usd:.4f}" if cost.cost_per_call_usd is not None else "n/a"),
        f"- 결정 1건당 총 비용: " + (f"${cost.cost_per_decision_usd:.4f}" if cost.cost_per_decision_usd is not None else "n/a")
        + f" ({cost.n_decisions} decisions)",
    ]
    if cost.monthly_budget_usd is not None:
        state = "within" if cost.within_budget else "OVER"
        lines.append(
            f"- budget: {state} ${cost.monthly_budget_usd:.2f}/month "
            f"(run-rate ${cost.monthly_run_rate_usd:.2f}/month over the evaluated window)"
        )

    lines += ["", f"## Gate: {'PASS' if report.gate.passed else 'FAIL'}", report.gate.verdict]
    return "\n".join(lines)
