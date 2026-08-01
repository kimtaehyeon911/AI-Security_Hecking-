"""The paper-trading loop: one step per trading day, same pipeline as the backtest.

Each ``step`` runs the identical decision→risk path the backtest uses
(``sample_decisions`` → ``RiskEngine.apply``), then does the paper-specific work
the backtest abstracts away: translate target weights into integer-share orders,
pass every order through :func:`vts.risk.validate_order`, fill accepted orders via
:class:`~vts.paper.broker.PaperBroker`, mark to market, and log implementation
shortfall vs a deterministic reference backtest. State is persisted atomically
after every step so an 8-week run survives restarts.

Parity notes (kept byte-for-byte with the backtest where it matters):
- The daily-loss halt is fed GENUINE single-day marks via the same
  ``_interval_daily_returns`` helper the backtest uses, so it is truly daily
  regardless of how far apart processed dates fall.
- Every held symbol is marked with its forward-filled last close (even after it
  leaves the active universe), so a data/universe gap never craters equity.
- Symbols are canonicalized to upper-case at the boundary so the loop's reads and
  the broker's writes share one key space.
The kill switch here FLATTENS the book (the intended live behavior) rather than
raising as the backtest does.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from vts.backtest.cache import DecisionCache
from vts.backtest.costs import CostModel
from vts.backtest.engine import (
    Backtester,
    BacktestConfig,
    _dollar_adv,
    _interval_daily_returns,
    _last_close,
    sample_decisions,
)
from vts.paper.broker import PaperBroker
from vts.paper.shortfall import make_record
from vts.paper.state import PaperState
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore
from vts.risk.killswitch import kill_switch_active
from vts.risk.limits import VenueRules
from vts.risk.orders import AccountState, Order, validate_order
from vts.risk.risk_engine import RiskEngine


class PaperTrader:
    """Runs the paper loop, reusing the Step 2–4 components verbatim."""

    def __init__(
        self,
        store: PointInTimeStore,
        model,
        risk: RiskEngine,
        *,
        venue: VenueRules,
        state_path: str | Path,
        cost_model: CostModel | None = None,
        cache: DecisionCache | None = None,
        config: BacktestConfig | None = None,
        dry_run: bool = True,
    ) -> None:
        self.store = store
        self.model = model
        self.risk = risk
        self.venue = venue
        self.state_path = Path(state_path)
        self.cost_model = cost_model or CostModel()
        self.cache = cache or DecisionCache()
        self.config = config or BacktestConfig()
        self.dry_run = dry_run
        self.broker = PaperBroker(self.cost_model, dry_run=dry_run)
        self.state = PaperState.load_or_new(
            self.state_path, self.config.initial_capital, dry_run=dry_run
        )
        # Rehydrate the halt latch. Only copy state INTO the engine when the engine
        # is in its default un-halted state; an already-halted engine handed in with
        # a not-halted state is a config error (mirrors Backtester's guard) — never
        # silently un-latch a halt without an operator reset.
        if self.risk.halt.halted and not self.state.halted:
            raise RuntimeError(
                "passed a halted RiskEngine but the persisted state is not halted; "
                "use a fresh RiskEngine or reset() it with an operator first."
            )
        self.risk.halt.halted = self.state.halted
        self.risk.halt.halt_reasons = list(self.state.halt_reasons)
        self.risk.halt._peak_equity = self.state.peak_equity

    # ------------------------------------------------------------------ helpers
    def _reference_curve(
        self, universe: list[str], dates: list[datetime]
    ) -> dict[str, tuple[float, bool]]:
        """Deterministic backtest over the same window -> {iso: (post_cost_equity, halted)}.

        Post-cost equity (``r.equity - r.cost``) so the reference and the paper's
        post-cost mark are measured at the same point (identical trades -> ~0 gap).
        Skipped under an engaged kill switch (the backtest refuses to run then).
        """
        if kill_switch_active():
            return {}
        ref_risk = RiskEngine(self.risk.limits, long_only=self.risk.long_only)
        bt = Backtester(
            self.store, self.model, cost_model=self.cost_model, cache=self.cache,
            config=self.config, risk=ref_risk,
        )
        result = bt.run(universe, dates)
        return {r.date.isoformat(): (r.equity - r.cost, r.halted) for r in result.records}

    def _marks(self, symbols: set[str], clock: AsOfClock) -> dict[str, float]:
        """Forward-filled last close for each symbol (held names always resolve)."""
        out: dict[str, float] = {}
        for s in symbols:
            p = _last_close(self.store, s, clock)
            if p is not None and p > 0:
                out[s] = p
        return out

    def _weights_to_orders(
        self,
        target: dict[str, float],
        marks: dict[str, float],
        equity: float,
        universe: set[str],
    ) -> list[Order]:
        """Integer-share delta orders vs current holdings, sells sequenced first.

        Uses ``int(...)`` (truncate toward zero) so a position never exceeds the
        risk-engine-intended weight for either longs or shorts. Held symbols that
        have left the active universe are liquidated (a full sell), so a curated
        universe on resume can never strand a position."""
        orders: list[Order] = []
        lot = self.venue.lot_size
        for sym, w in target.items():
            if sym not in marks:
                continue
            price = marks[sym]
            desired = int(w * equity / price / lot) * lot     # toward zero
            delta = desired - self.state.positions.get(sym, 0.0)
            if abs(delta) < lot:
                continue
            orders.append(Order(symbol=sym, side="buy" if delta > 0 else "sell",
                                qty=abs(delta), limit_price=price))
        for sym, held in self.state.positions.items():
            if sym in universe or held == 0.0 or sym not in marks:
                continue
            orders.append(Order(symbol=sym, side="sell", qty=abs(held), limit_price=marks[sym]))
        # Sells first: cash freed by liquidations funds the rotation's buys.
        orders.sort(key=lambda o: 0 if o.side == "sell" else 1)
        return orders

    def _prior_gap(self) -> float:
        for rec in reversed(self.state.shortfall_log):
            if rec.get("gap") is not None:
                return rec["gap"]
        return 0.0

    # --------------------------------------------------------------------- step
    def step(
        self, universe: list[str], date: datetime, reference: tuple[float, bool] | None
    ) -> None:
        """Process one trading day: mark, decide, gate, trade, log, persist."""
        iso = date.isoformat()
        if iso in self.state.processed_dates:
            return  # idempotent resume
        uni = [t.strip().upper() for t in universe]
        uni_set = set(uni)
        clock = AsOfClock.at(date)

        # 1) Mark to market (forward-filled prices for every held symbol) and feed
        #    the halt GENUINE daily marks over the interval since the last step.
        marks = self._marks(uni_set | set(self.state.positions), clock)
        equity_in = self.state.mark_to_market(marks)
        if self.state.processed_dates:
            prev_dt = datetime.fromisoformat(self.state.processed_dates[-1])
            prev_eq = self.state.equity_curve[-1][1]
            prev_clock = AsOfClock.at(prev_dt)
            prev_weights: dict[str, float] = {}
            for s, sh in self.state.positions.items():
                pp = _last_close(self.store, s, prev_clock)
                if pp and prev_eq > 0:
                    prev_weights[s] = sh * pp / prev_eq
            for dr in _interval_daily_returns(self.store, prev_weights, prev_dt, date):
                self.risk.observe_daily_return(dr)
        self.risk.observe_equity(equity_in)

        # 2) Decide + gate + size — the SAME pipeline the backtest uses.
        priced_uni = {t: marks[t] for t in uni if t in marks}
        aggs = {
            t: sample_decisions(self.model, self.cache, t, clock, self.config.n_samples)
            for t in priced_uni
        }
        verdict = self.risk.apply(aggs)
        target = {t: verdict.weights.get(t, 0.0) for t in uni}

        # 3) Orders -> validate (fee-covering buffer) -> fill accepted.
        orders = self._weights_to_orders(target, marks, equity_in, uni_set)
        fills, rejections = [], []
        for order in orders:
            adv = _dollar_adv(self.store, order.symbol, clock, self.config.adv_lookback)
            est_cost = self.cost_model.cost(order.notional, adv, order.side).total
            account = AccountState(cash=self.state.cash, positions=dict(self.state.positions))
            rejection = validate_order(order, account, self.venue, cash_buffer=est_cost)
            if rejection is not None:
                rejections.append(rejection)
                continue
            fills.append(self.broker.execute(self.state, order, order.limit_price, adv))

        # 4) Record equity, decision, shortfall; persist atomically.
        marks_out = self._marks(uni_set | set(self.state.positions), clock)
        equity_out = self.state.mark_to_market(marks_out)
        ref_equity, ref_halted = (reference if reference is not None else (None, False))
        self.state.equity_curve.append((iso, equity_out))
        self.state.processed_dates.append(iso)
        self.state.decision_log.append({
            "date": iso,
            "ratings": {t: a.rating.value for t, a in aggs.items()},
            "target_weights": {t: round(w, 4) for t, w in target.items() if w != 0.0},
            "halted": verdict.halted,
            "forced_holds": sum(1 for g in verdict.gated.values() if g.forced_hold),
            "fills": len(fills),
            "rejections": [r.violations for r in rejections],
        })
        rec = make_record(
            iso, equity_out, ref_equity, self.state.initial_capital, self._prior_gap(),
            paper_halted=verdict.halted, reference_halted=ref_halted,
        )
        self.state.shortfall_log.append(rec.model_dump())

        # Persist the halt latch so a resumed run stays stopped.
        self.state.halted = self.risk.halt.halted
        self.state.halt_reasons = list(self.risk.halt.halt_reasons)
        self.state.peak_equity = self.risk.halt._peak_equity
        self.state.save(self.state_path)

    def run(self, universe: list[str], dates: list[datetime]) -> PaperState:
        """Run (or resume) the loop over ``dates`` (one step per trading day).

        ``dates`` must be the FULL window each call so the reference backtest stays
        anchored to the same initial capital across a resume boundary.
        """
        ordered = sorted(dates)
        reference = self._reference_curve(universe, ordered)
        for d in ordered:
            self.step(universe, d, reference.get(d.isoformat()))
        return self.state
