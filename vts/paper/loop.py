"""The paper-trading loop: one step per trading day, same pipeline as the backtest.

Each ``step`` runs the identical decision→risk path the backtest uses
(``sample_decisions`` → ``RiskEngine.apply``), then does the paper-specific work
the backtest abstracts away: translate target weights into integer-share orders,
pass every order through :func:`vts.risk.validate_order`, fill accepted orders via
:class:`~vts.paper.broker.PaperBroker`, mark to market, and log implementation
shortfall vs a deterministic reference backtest. State is persisted after every
step so an 8-week run survives restarts.

Cadence: one step per trading day (the 일봉 1회 cadence), so each step's return IS
a daily return and feeds the daily-loss halt directly. The kill switch here
FLATTENS (the intended live behavior) rather than raising as the backtest does.
"""

from __future__ import annotations

import math
from datetime import datetime
from pathlib import Path

from vts.backtest.cache import DecisionCache
from vts.backtest.costs import CostModel
from vts.backtest.engine import (
    Backtester,
    BacktestConfig,
    _dollar_adv,
    _last_close,
    sample_decisions,
)
from vts.paper.broker import PaperBroker
from vts.paper.shortfall import ShortfallRecord, make_record
from vts.paper.state import PaperState
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore
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
        # Rehydrate the halt latch so a resumed run stays halted.
        self.risk.halt.halted = self.state.halted
        self.risk.halt.halt_reasons = list(self.state.halt_reasons)
        self.risk.halt._peak_equity = self.state.peak_equity

    # ------------------------------------------------------------------ helpers
    def _reference_curve(self, universe: list[str], dates: list[datetime]) -> dict[str, float]:
        """Deterministic backtest over the same window (fresh RiskEngine, shared cache).

        Under an engaged kill switch the backtest deliberately refuses to run, so
        there is no meaningful reference — the shortfall log records None for those
        days and the paper loop proceeds (flat) regardless.
        """
        from vts.risk.killswitch import kill_switch_active

        if kill_switch_active():
            return {}
        ref_risk = RiskEngine(self.risk.limits, long_only=self.risk.long_only)
        bt = Backtester(
            self.store, self.model, cost_model=self.cost_model, cache=self.cache,
            config=self.config, risk=ref_risk,
        )
        result = bt.run(universe, dates)
        return {r.date.isoformat(): r.equity for r in result.records}

    def _prices(self, universe: list[str], clock: AsOfClock) -> dict[str, float]:
        out = {}
        for t in universe:
            p = _last_close(self.store, t, clock)
            if p is not None and p > 0:
                out[t] = p
        return out

    def _weights_to_orders(
        self, target: dict[str, float], prices: dict[str, float], equity: float
    ) -> list[Order]:
        """Translate target weights into integer-share delta orders vs current holdings."""
        orders: list[Order] = []
        for sym, w in target.items():
            if sym not in prices:
                continue
            price = prices[sym]
            desired_shares = math.floor(w * equity / price / self.venue.lot_size) * self.venue.lot_size
            held = self.state.positions.get(sym, 0.0)
            delta = desired_shares - held
            if abs(delta) < self.venue.lot_size:
                continue
            side = "buy" if delta > 0 else "sell"
            orders.append(Order(symbol=sym, side=side, qty=abs(delta), limit_price=price))
        return orders

    # --------------------------------------------------------------------- step
    def step(self, universe: list[str], date: datetime, reference_equity: float | None) -> None:
        """Process one trading day: mark, decide, gate, trade, log, persist."""
        iso = date.isoformat()
        if iso in self.state.processed_dates:
            return  # idempotent resume
        clock = AsOfClock.at(date)
        prices = self._prices(universe, clock)

        # 1) Mark to market and feed the daily-loss / drawdown halt (daily cadence).
        equity_in = self.state.mark_to_market(prices)
        if self.state.equity_curve:
            prev_eq = self.state.equity_curve[-1][1]
            if prev_eq > 0:
                self.risk.observe_daily_return(equity_in / prev_eq - 1.0)
        self.risk.observe_equity(equity_in)

        # 2) Decide + gate + size — the SAME pipeline the backtest uses.
        aggs = {
            t: sample_decisions(self.model, self.cache, t, clock, self.config.n_samples)
            for t in prices
        }
        verdict = self.risk.apply(aggs)
        target = {t: verdict.weights.get(t, 0.0) for t in universe}

        # 3) Translate to integer-share orders, validate each, fill the accepted.
        orders = self._weights_to_orders(target, prices, equity_in)
        fills, rejections = [], []
        for order in orders:
            adv = _dollar_adv(self.store, order.symbol, clock, self.config.adv_lookback)
            est_cost = self.cost_model.cost(order.notional, adv, order.side).total
            account = AccountState(cash=self.state.cash, positions=dict(self.state.positions))
            rejection = validate_order(order, account, self.venue, cash_buffer=est_cost)
            if rejection is not None:
                rejections.append(rejection)
                continue
            fill = self.broker.execute(self.state, order, order.limit_price, adv)
            fills.append(fill)

        # 4) Record equity, decision, shortfall; persist.
        equity_out = self.state.mark_to_market(prices)
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
        prior_cum = (
            self.state.shortfall_log[-1]["cumulative_shortfall"]
            if self.state.shortfall_log else 0.0
        )
        rec: ShortfallRecord = make_record(
            iso, equity_out, reference_equity, self.state.initial_capital, prior_cum
        )
        self.state.shortfall_log.append(rec.model_dump())

        # Persist the halt latch so a resumed run stays stopped.
        self.state.halted = self.risk.halt.halted
        self.state.halt_reasons = list(self.risk.halt.halt_reasons)
        self.state.peak_equity = self.risk.halt._peak_equity
        self.state.save(self.state_path)

    def run(self, universe: list[str], dates: list[datetime]) -> PaperState:
        """Run (or resume) the loop over ``dates`` (one step per trading day)."""
        ordered = sorted(dates)
        reference = self._reference_curve(universe, ordered)
        for d in ordered:
            self.step(universe, d, reference.get(d.isoformat()))
        return self.state
