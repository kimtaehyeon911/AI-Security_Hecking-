"""LiveTrader — the guarded live path. Same decision pipeline, real consequences.

Every cycle runs the identical Step 2–4 chain the backtest and paper loop use
(``sample_decisions`` → ``RiskEngine.apply`` → ``validate_order``); what this
class adds is the safety envelope around routing:

- **Kill switch first.** Before anything else, ``VTS_KILL_SWITCH`` triggers
  :func:`~vts.live.liquidate.liquidate_all` (cancel + close everything) and
  latches the halt. Checked fresh from the environment on every cycle.
- **Capital cap on every cycle**, not just at startup — an account that shrinks
  must re-tighten the allocation, and the cap is re-derived from the broker's
  reported total assets each time.
- **Arming re-checked per order batch.** Disarming mid-session stops the next
  batch; there is no cached "we were armed earlier".
- **Dry run by default.** In dry run the full pipeline runs and the orders it
  *would* send are returned — nothing reaches the broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from vts.backtest.cache import DecisionCache
from vts.backtest.costs import CostModel
from vts.backtest.engine import BacktestConfig, _dollar_adv, _last_close, sample_decisions
from vts.live.broker import BrokerClient
from vts.live.config import LiveConfig, assert_capital_within_cap
from vts.live.liquidate import LiquidationReport, liquidate_all
from vts.pit.clock import AsOfClock
from vts.pit.store import PointInTimeStore
from vts.risk.killswitch import kill_switch_active
from vts.risk.limits import VenueRules
from vts.risk.orders import AccountState, Order, OrderRejection, validate_order
from vts.risk.risk_engine import RiskEngine


@dataclass
class CycleResult:
    """What one trading cycle did (or would have done in dry run)."""

    date: datetime
    dry_run: bool
    halted: bool = False
    liquidation: LiquidationReport | None = None
    submitted: list[Order] = field(default_factory=list)
    would_submit: list[Order] = field(default_factory=list)
    rejections: list[OrderRejection] = field(default_factory=list)
    forced_holds: int = 0
    notes: list[str] = field(default_factory=list)


class LiveTrader:
    """Runs live cycles behind the full safety envelope."""

    def __init__(
        self,
        store: PointInTimeStore,
        model,
        risk: RiskEngine,
        broker: BrokerClient,
        *,
        venue: VenueRules,
        live_config: LiveConfig,
        cost_model: CostModel | None = None,
        cache: DecisionCache | None = None,
        config: BacktestConfig | None = None,
        audit_path: str | Path | None = None,
    ) -> None:
        self.store = store
        self.model = model
        self.risk = risk
        self.broker = broker
        self.venue = venue
        self.live = live_config
        self.cost_model = cost_model or CostModel()
        self.cache = cache or DecisionCache()
        self.config = config or BacktestConfig()
        self.audit_path = Path(audit_path) if audit_path else None

    # ------------------------------------------------------------------ helpers
    def _audit(self, line: str) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")

    def _marks(self, symbols: set[str], clock: AsOfClock) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in symbols:
            p = _last_close(self.store, s, clock)
            if p is not None and p > 0:
                out[s] = p
        return out

    def _held(self) -> dict[str, float]:
        return {p.symbol.strip().upper(): p.qty for p in self.broker.get_positions()}

    def _orders_for(
        self, target: dict[str, float], marks: dict[str, float], capital: float,
        held: dict[str, float], universe: set[str],
    ) -> list[Order]:
        """Integer-share deltas vs broker-reported holdings; sells sequenced first."""
        orders: list[Order] = []
        lot = self.venue.lot_size
        for sym, w in target.items():
            if sym not in marks:
                continue
            price = marks[sym]
            desired = int(w * capital / price / lot) * lot   # toward zero
            delta = desired - held.get(sym, 0.0)
            if abs(delta) < lot:
                continue
            orders.append(Order(symbol=sym, side="buy" if delta > 0 else "sell",
                                qty=abs(delta), limit_price=price))
        for sym, qty in held.items():
            if sym in universe or qty == 0.0 or sym not in marks:
                continue
            orders.append(Order(symbol=sym, side="sell" if qty > 0 else "buy",
                                qty=abs(qty), limit_price=marks[sym]))
        orders.sort(key=lambda o: 0 if o.side == "sell" else 1)
        return orders

    # -------------------------------------------------------------------- cycle
    def run_cycle(self, universe: list[str], date: datetime) -> CycleResult:
        """One live cycle. Kill switch → cap → decide → validate → route."""
        result = CycleResult(date=date, dry_run=self.live.dry_run)
        uni = [t.strip().upper() for t in universe]
        uni_set = set(uni)
        clock = AsOfClock.at(date)

        # 1) Kill switch BEFORE anything else: liquidate everything, latch, stop.
        if kill_switch_active():
            self.risk.halt.check_kill_switch()
            report = liquidate_all(
                self.broker, reason="VTS_KILL_SWITCH engaged", dry_run=self.live.dry_run
            )
            result.halted = True
            result.liquidation = report
            result.notes.append("kill switch engaged — full liquidation")
            self._audit(f"{date.isoformat()} KILL_SWITCH liquidation={report.model_dump_json()}")
            return result

        # 2) An already-latched halt (daily loss / drawdown) also liquidates once
        #    and never trades again without a named operator reset.
        if self.risk.halt.halted:
            report = liquidate_all(
                self.broker, reason="; ".join(self.risk.halt.halt_reasons) or "halted",
                dry_run=self.live.dry_run,
            )
            result.halted = True
            result.liquidation = report
            result.notes.append("risk halt latched — full liquidation")
            self._audit(f"{date.isoformat()} HALT liquidation={report.model_dump_json()}")
            return result

        # 3) Capital cap re-derived from the broker EVERY cycle.
        account = self.broker.get_account()
        assert_capital_within_cap(self.live.allocated_capital, account.total_assets)
        if not self.live.dry_run:
            self.live.assert_armed_for_live(account.total_assets)

        # 4) Same decision pipeline as backtest/paper.
        held = self._held()
        marks = self._marks(uni_set | set(held), clock)
        priced = {t: marks[t] for t in uni if t in marks}
        aggs = {
            t: sample_decisions(self.model, self.cache, t, clock, self.config.n_samples)
            for t in priced
        }
        verdict = self.risk.apply(aggs)
        result.forced_holds = sum(1 for g in verdict.gated.values() if g.forced_hold)
        if verdict.halted:  # the gate latched inside apply (e.g. kill switch race)
            result.halted = True
            result.notes.append("risk engine returned halted verdict")
            return result
        target = {t: verdict.weights.get(t, 0.0) for t in uni}

        # 5) Orders sized against ALLOCATED capital (never the whole account).
        orders = self._orders_for(target, marks, self.live.allocated_capital, held, uni_set)
        if len(orders) > self.live.max_orders_per_session:
            result.notes.append(
                f"order count {len(orders)} exceeds max_orders_per_session "
                f"{self.live.max_orders_per_session} — refusing to route"
            )
            self._audit(f"{date.isoformat()} REFUSED too_many_orders={len(orders)}")
            return result

        # 6) Validate every order, then route (or report in dry run).
        cash = account.cash
        positions = dict(held)
        for order in orders:
            adv = _dollar_adv(self.store, order.symbol, clock, self.config.adv_lookback)
            est_cost = self.cost_model.cost(order.notional, adv, order.side).total
            rejection = validate_order(
                order, AccountState(cash=max(cash, 0.0), positions=positions),
                self.venue, cash_buffer=est_cost,
            )
            if rejection is not None:
                result.rejections.append(rejection)
                self._audit(f"{date.isoformat()} REJECT {order.symbol} {rejection.violations}")
                continue
            if self.live.dry_run:
                result.would_submit.append(order)
            else:
                # Re-check arming per order: disarming mid-batch must stop routing.
                self.live.assert_armed_for_live(account.total_assets)
                self.broker.submit_order(order.symbol, order.side, order.qty)
                result.submitted.append(order)
                self._audit(
                    f"{date.isoformat()} SENT {order.side} {order.qty:g} {order.symbol}"
                )
            # Keep the local view in step so later orders validate against it.
            signed = order.qty if order.side == "buy" else -order.qty
            positions[order.symbol] = positions.get(order.symbol, 0.0) + signed
            cash -= signed * order.limit_price + est_cost

        return result
