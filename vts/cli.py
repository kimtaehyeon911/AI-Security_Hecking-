"""Operational CLI — the runnable entry points for every stage of the system.

    python -m vts ingest    --start 2025-01-01 --end 2025-06-30
    python -m vts backtest  --start 2025-02-01 --end 2025-06-30 [--report out.md]
    python -m vts paper     --start 2025-07-01 --end 2025-08-31
    python -m vts live      --date 2025-09-01 [--capital 100] [--go-live]
    python -m vts status
    python -m vts reset-halt --scope paper --operator you@ops

Safety posture mirrors the library: the risk layer is ALWAYS wired in (there is
no CLI flag to run un-gated), `live` is dry-run unless ``--go-live`` — and even
then the arming token, hardcoded 1% capital cap and kill switch still gate it.
The default decision model is the deterministic offline momentum model; pass
``--model tradingagents`` once the fork + LLM keys are installed.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

from vts.backtest.cache import DecisionCache
from vts.backtest.engine import Backtester, BacktestConfig
from vts.backtest.evaluate import evaluate, render_report
from vts.backtest.model import FakeMomentumModel
from vts.config import Settings, load_settings
from vts.ingest import ingest_symbol
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore
from vts.risk.limits import BINANCE_SPOT_DEFAULT, RiskLimits, VenueRules
from vts.risk.risk_engine import RiskEngine

_UTC = timezone.utc


def _parse_date(s: str) -> datetime:
    return datetime.strptime(s, "%Y-%m-%d").replace(tzinfo=_UTC)


def _make_source(settings: Settings):
    """Build the configured data source (separate function so tests can stub it)."""
    if settings.data_source == "binance":
        from vts.sources.binance_data import BinanceSource

        return BinanceSource(interval=settings.cadence)
    if settings.data_source == "alpha_vantage":
        from vts.sources.alpha_vantage import AlphaVantageSource

        return AlphaVantageSource()
    raise SystemExit(f"unknown data source {settings.data_source!r}")


def _make_model(name: str, store: PointInTimeStore, settings: Settings):
    if name == "momentum":
        return FakeMomentumModel(store)
    if name == "tradingagents":
        try:
            from vts.integration.ta_decision import TradingAgentsDecisionModel
        except ImportError as exc:  # pragma: no cover - needs the fork installed
            raise SystemExit(
                "the TradingAgents fork is not installed; install it and set the "
                "LLM keys in .env before using --model tradingagents"
            ) from exc
        def _graph_factory():
            # Build the graph with the vts-selected LLM. The fork constructs its
            # deep/quick clients from config["llm_provider"|"deep_think_llm"|
            # "quick_think_llm"] (verified against TradingAgentsGraph.__init__), so we
            # base on its live config (preserving non-LLM defaults) and override only
            # those three. Provider/models come from VTS_LLM_PROVIDER /
            # VTS_DEEP_THINK_LLM / VTS_QUICK_THINK_LLM — e.g. google + gemini-2.5-pro.
            # The API key is read by the fork's client from the environment
            # (GOOGLE_API_KEY for Gemini). Temperature is left to the fork's default:
            # do NOT pin it to 0, or the N-sample vote collapses to zero dispersion.
            from tradingagents.dataflows.config import get_config
            from tradingagents.graph.trading_graph import TradingAgentsGraph
            cfg = dict(get_config())
            cfg["llm_provider"] = settings.llm_provider
            cfg["deep_think_llm"] = settings.deep_think_llm
            cfg["quick_think_llm"] = settings.quick_think_llm
            return TradingAgentsGraph(config=cfg)

        # model_id keys the contamination gate on the ACTUAL reasoning model, not a
        # generic "tradingagents" label: the deep-think LLM (VTS_DEEP_THINK_LLM) is
        # what could have memorized outcomes, so its id must match a cutoff-registry
        # entry (e.g. gemini-2.5-pro) or the gate degrades to UNKNOWN. For Gemini the
        # API model name and the registry key coincide, so deep_think_llm serves both.
        return TradingAgentsDecisionModel(
            store, model_id=settings.deep_think_llm, graph_factory=_graph_factory
        )
    raise SystemExit(f"unknown model {name!r}")


def _venue(settings: Settings) -> VenueRules:
    return BINANCE_SPOT_DEFAULT  # per-symbol exchangeInfo rules are a live concern


def _samples(args: argparse.Namespace) -> int:
    """Default N: 1 for the deterministic momentum model, 3 (mandated) for LLMs."""
    if args.samples is not None:
        return args.samples
    return 3 if args.model == "tradingagents" else 1


def _cache(args: argparse.Namespace, settings: Settings) -> DecisionCache:
    if getattr(args, "no_cache", False):
        return DecisionCache()  # throwaway in-memory
    return DecisionCache(settings.data_dir / "decision_cache.sqlite")


def _decision_dates(
    store: PointInTimeStore, symbols: list[str], start: datetime, end: datetime
) -> list[datetime]:
    """Daily decision dates = the UNION of all universe symbols' bar closes.

    Anchoring on a single symbol silently shrinks the whole run when that symbol
    has sparse coverage (a confirmed review finding); the union keeps every
    trading day any symbol traded, and per-symbol gaps are reported loudly.
    """
    clock = AsOfClock.at(end + timedelta(days=2))
    per_symbol: dict[str, set[datetime]] = {}
    for sym in symbols:
        bars = store.get_ohlcv(sym, clock, start=start, end=end + timedelta(days=1))
        per_symbol[sym] = {b.event_time for b in bars if start <= b.event_time <= end}
    union: set[datetime] = set().union(*per_symbol.values()) if per_symbol else set()
    for sym, times in per_symbol.items():
        missing = len(union) - len(times)
        if union and missing > 0:
            print(f"warning: {sym} covers {len(times)}/{len(union)} decision dates "
                  f"in this window", file=sys.stderr)
    return sorted(union)


# ------------------------------------------------------------------- commands
def cmd_ingest(args: argparse.Namespace, settings: Settings) -> int:
    store = PointInTimeStore(settings.store_path)
    source = _make_source(settings)
    start, end = _parse_date(args.start), _parse_date(args.end)
    for symbol in settings.universe:
        report = ingest_symbol(store, source, symbol, start, end)
        print(f"{symbol}: ohlcv={report.ohlcv} news={report.news} "
              f"fundamentals={report.fundamentals} actions={report.corporate_actions}")
    print(f"store: {settings.store_path}")
    return 0


def cmd_backtest(args: argparse.Namespace, settings: Settings) -> int:
    store = PointInTimeStore(settings.store_path)
    start, end = _parse_date(args.start), _parse_date(args.end)
    dates = _decision_dates(store, settings.universe, start, end)
    if len(dates) < 2:
        print("not enough bars in the store for this window — run `ingest` first",
              file=sys.stderr)
        return 2
    model = _make_model(args.model, store, settings)
    bt = Backtester(
        store, model,
        cache=_cache(args, settings),
        config=BacktestConfig(n_samples=_samples(args)),
        risk=RiskEngine(RiskLimits.from_env()),
    )
    result = bt.run(settings.universe, dates)
    report = evaluate(bt, result, settings.universe,
                      monthly_budget_usd=settings.monthly_budget_usd)
    text = render_report(report)
    if args.report:
        Path(args.report).write_text(text, encoding="utf-8")
        print(f"report written to {args.report}")
    else:
        print(text)
    return 0 if report.gate.passed else 1


def cmd_paper(args: argparse.Namespace, settings: Settings) -> int:
    from vts.paper.loop import PaperTrader

    store = PointInTimeStore(settings.store_path)
    start, end = _parse_date(args.start), _parse_date(args.end)
    dates = _decision_dates(store, settings.universe, start, end)
    if not dates:
        print("no bars in the store for this window — run `ingest` first", file=sys.stderr)
        return 2
    trader = PaperTrader(
        store, _make_model(args.model, store, settings), RiskEngine(RiskLimits.from_env()),
        venue=_venue(settings), state_path=settings.data_dir / "paper_state.json",
        cache=_cache(args, settings),
        config=BacktestConfig(n_samples=_samples(args)),
    )
    state = trader.run(settings.universe, dates)
    last = state.shortfall_log[-1] if state.shortfall_log else {}
    print(f"processed {len(state.processed_dates)} days; "
          f"equity={state.equity_curve[-1][1]:,.2f}; halted={state.halted}")
    if last.get("gap") is not None:
        print(f"implementation shortfall (total gap): {last['gap']:,.2f} "
              f"({last['gap_bps_of_capital']:.1f} bps of capital)")
    print(f"state: {settings.data_dir / 'paper_state.json'}")
    return 0


def cmd_live(args: argparse.Namespace, settings: Settings) -> int:
    from vts.live.binance_broker import BinanceBroker
    from vts.live.config import LiveConfig
    from vts.live.trader import LiveTrader

    store = PointInTimeStore(settings.store_path)
    date = _parse_date(args.date) if args.date else datetime.now(_UTC)
    trader = LiveTrader(
        store, _make_model(args.model, store, settings), RiskEngine(RiskLimits.from_env()),
        BinanceBroker(),  # TESTNET unless the adapter is explicitly pointed at prod
        venue=_venue(settings),
        live_config=LiveConfig(allocated_capital=args.capital,
                               dry_run=not args.go_live),
        cache=_cache(args, settings),
        config=BacktestConfig(n_samples=_samples(args)),
        audit_path=settings.data_dir / "live_audit.log",
        state_path=settings.data_dir / "live_state.json",
    )
    result = trader.run_cycle(settings.universe, date)
    mode = "LIVE" if args.go_live else "dry-run"
    print(f"[{mode}] halted={result.halted} submitted={len(result.submitted)} "
          f"would_submit={len(result.would_submit)} rejections={len(result.rejections)}")
    for note in result.notes:
        print(f"  note: {note}")
    return 0


def cmd_status(args: argparse.Namespace, settings: Settings) -> int:
    print(f"profile: {settings.asset_class} / {settings.data_source} / "
          f"{','.join(settings.universe)}")
    store_path = settings.store_path
    if store_path.exists():
        store = PointInTimeStore(store_path)
        clock = AsOfClock.at(datetime.now(_UTC) + timedelta(days=1))
        for sym in settings.universe:
            bars = store.get_ohlcv(sym, clock)
            span = (f"{bars[0].event_time.date()} → {bars[-1].event_time.date()}"
                    if bars else "-")
            print(f"  {sym}: {len(bars)} bars ({span})")
    else:
        print(f"  store not found at {store_path} — run `ingest`")
    for scope, fname in (("paper", "paper_state.json"), ("live", "live_state.json")):
        path = settings.data_dir / fname
        if not path.exists():
            continue
        if scope == "paper":
            from vts.paper.state import PaperState

            st = PaperState.load(path)
            print(f"  paper: {len(st.processed_dates)} days, halted={st.halted}, "
                  f"dust={st.dust_symbols}")
        else:
            from vts.live.state import LiveState

            st = LiveState.load_or_new(path)
            print(f"  live: sleeve={st.sleeve_positions}, halted={st.halted}, "
                  f"dust={st.dust_symbols}")
        for reason in st.halt_reasons[-3:]:
            print(f"    halt: {reason}")
    return 0


def cmd_reset_halt(args: argparse.Namespace, settings: Settings) -> int:
    """Named-operator halt reset — the ONLY way a latched stop is cleared."""
    if args.scope == "paper":
        from vts.paper.state import PaperState

        path = settings.data_dir / "paper_state.json"
        st = PaperState.load(path)
        st.halt_reasons.append(f"reset by {args.operator}")
        st.halted = False
        st.peak_equity = 0.0
        st.save(path)
    else:
        from vts.live.state import LiveState

        path = settings.data_dir / "live_state.json"
        st = LiveState.load_or_new(path)
        st.halt_reasons.append(f"reset by {args.operator}")
        st.halted = False
        st.liquidation_verified_flat = False
        st.save(path)
    print(f"{args.scope} halt reset by {args.operator} — recorded in the audit trail")
    return 0


def cmd_cache_clear(args: argparse.Namespace, settings: Settings) -> int:
    path = settings.data_dir / "decision_cache.sqlite"
    if path.exists():
        path.unlink()
        print(f"deleted {path}")
    else:
        print("no decision cache to delete")
    return 0


# ---------------------------------------------------------------------- main
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="vts", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)

    def _common(sp):
        sp.add_argument("--model", default="momentum",
                        choices=["momentum", "tradingagents"])
        sp.add_argument("--samples", type=int, default=None,
                        help="N samples per decision (majority vote). Default: 1 for "
                             "the deterministic momentum model, 3 for LLM models "
                             "(the mandated N=3 vote).")
        sp.add_argument("--no-cache", action="store_true",
                        help="use a throwaway in-memory decision cache for this run")

    sp = sub.add_parser("ingest", help="pull point-in-time data into the store")
    sp.add_argument("--start", required=True)
    sp.add_argument("--end", required=True)
    sp.set_defaults(func=cmd_ingest)

    sp = sub.add_parser("backtest", help="walk-forward backtest + gate report")
    sp.add_argument("--start", required=True)
    sp.add_argument("--end", required=True)
    sp.add_argument("--report", help="write the markdown report here")
    _common(sp)
    sp.set_defaults(func=cmd_backtest)

    sp = sub.add_parser("paper", help="run/resume the paper loop over a window")
    sp.add_argument("--start", required=True)
    sp.add_argument("--end", required=True)
    _common(sp)
    sp.set_defaults(func=cmd_paper)

    sp = sub.add_parser("live", help="one live cycle (dry-run unless --go-live)")
    sp.add_argument("--date", help="YYYY-MM-DD (default: now)")
    sp.add_argument("--capital", type=float, default=100.0,
                    help="allocated capital (checked against the hardcoded 1%% cap)")
    sp.add_argument("--go-live", action="store_true",
                    help="disable dry-run; STILL requires the arming token env var")
    _common(sp)
    sp.set_defaults(func=cmd_live)

    sp = sub.add_parser("status", help="store/state/halt overview")
    sp.set_defaults(func=cmd_status)

    sp = sub.add_parser("reset-halt", help="clear a latched halt (named operator only)")
    sp.add_argument("--scope", required=True, choices=["paper", "live"])
    sp.add_argument("--operator", required=True,
                    help="who is clearing the halt (recorded in the audit trail)")
    sp.set_defaults(func=cmd_reset_halt)

    sp = sub.add_parser("cache-clear",
                        help="delete the shared decision cache (e.g. after a re-ingest)")
    sp.set_defaults(func=cmd_cache_clear)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    settings = load_settings()
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    return args.func(args, settings)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
