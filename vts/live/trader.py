"""LiveTrader — the guarded live path. Same decision pipeline, real consequences.

Every cycle runs the identical Step 2–4 chain the backtest and paper loop use
(``sample_decisions`` → ``RiskEngine.apply`` → ``validate_order``); what this
class adds is the safety envelope around routing:

- **Kill switch first.** ``VTS_KILL_SWITCH`` liquidates the sleeve and latches,
  durably. Checked fresh from the environment every cycle.
- **Sleeve-scoped.** Orders reconcile against this system's own ledger, never the
  raw broker account, so it can neither trade nor liquidate an operator's
  unrelated holdings.
- **Capital cap every cycle** (dry run included), re-derived from the broker's
  reported total assets, plus an aggregate-notional check so the batch as a whole
  cannot exceed the allocation even if a risk-limit env var is loosened.
- **Arming re-checked per order.** Disarming mid-batch stops the next order.
- **Dry run by default** — the full pipeline runs and reports what it *would*
  send; nothing reaches the broker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from vts.backtest.cache import DecisionCache
from vts.backtest.costs import CostModel
from vts.backtest.engine import BacktestConfig, _dollar_adv, _last_close, sample_decisions
from vts.live.broker import BrokerClient
from vts.live.config import LiveConfig, assert_capital_within_cap, live_trading_armed
from vts.live.liquidate import LiquidationReport, liquidate_all
from vts.live.state import LiveState
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
        state_path: str | Path | None = None,
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
        self.state_path = Path(state_path) if state_path else None
        self.state = LiveState.load_or_new(self.state_path)
        # Rehydrate the durable halt latch so a restart cannot resume trading.
        if self.state.halted:
            self.risk.halt.halted = True
            self.risk.halt.halt_reasons = list(self.state.halt_reasons)

    # ------------------------------------------------------------------ helpers
    def _audit(self, line: str) -> None:
        if self.audit_path is None:
            return
        self.audit_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.audit_path, "a", encoding="utf-8") as fh:
            fh.write(line.rstrip("\n") + "\n")

    def _persist(self) -> None:
        self.state.halted = self.risk.halt.halted
        self.state.halt_reasons = list(self.risk.halt.halt_reasons)
        if self.state_path is not None:
            self.state.save(self.state_path)

    def _marks(self, symbols: set[str], clock: AsOfClock) -> dict[str, float]:
        out: dict[str, float] = {}
        for s in symbols:
            p = _last_close(self.store, s, clock)
            if p is not None and p > 0:
                out[s] = p
        return out

    def _sleeve_held(self, result: CycleResult) -> dict[str, float]:
        """Sleeve holdings reconciled with the broker, net of still-working orders.

        Only symbols in this system's ledger are considered — the operator's other
        positions are invisible to the trader by construction. Working orders are
        netted in so an unfilled order is not duplicated next cycle.
        """
        sleeve = self.state.sleeve_symbols()
        broker_pos = {p.symbol.strip().upper(): p.qty for p in self.broker.get_positions()}
        held = {s: broker_pos.get(s, 0.0) for s in sleeve}
        for sym, qty in self.state.sleeve_positions.items():
            if sym in broker_pos and abs(broker_pos[sym] - qty) > 1e-9:
                result.notes.append(
                    f"ledger drift {sym}: sleeve={qty:g} broker={broker_pos[sym]:g}"
                )
        try:
            for o in self.broker.get_open_orders():
                sym = o.symbol.strip().upper()
                if sym in held:
                    held[sym] += o.qty if o.side == "buy" else -o.qty
        except Exception as exc:  # an adapter without open-order support must be loud
            result.notes.append(f"open-order netting unavailable: {exc}")
        return held

    def _orders_for(
        self, target: dict[str, float], marks: dict[str, float], capital: float,
        held: dict[str, float], universe: set[str], result: CycleResult,
    ) -> list[Order]:
        """Integer-share deltas vs sleeve holdings; sells sequenced first."""
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
            # Belt-and-braces: no single order may exceed the whole allocation.
            if abs(delta) * price > capital:
                result.notes.append(
                    f"clipped {sym}: delta notional {abs(delta) * price:,.2f} > "
                    f"allocation {capital:,.2f}"
                )
                delta = (int(capital / price / lot) * lot) * (1 if delta > 0 else -1)
                if abs(delta) < lot:
                    continue
            orders.append(Order(symbol=sym, side="buy" if delta > 0 else "sell",
                                qty=abs(delta), limit_price=price))
        # Exit sleeve positions that left the universe.
        for sym, qty in held.items():
            if sym in universe or qty == 0.0:
                continue
            if sym not in marks:
                result.notes.append(f"unpriced sleeve holding {sym} qty={qty:g} — not exited")
                continue
            orders.append(Order(symbol=sym, side="sell" if qty > 0 else "buy",
                                qty=abs(qty), limit_price=marks[sym]))
        orders.sort(key=lambda o: 0 if o.side == "sell" else 1)
        return orders

    def _liquidate(self, reason: str, result: CycleResult) -> None:
        """Emergency flatten of the SLEEVE (never the operator's other holdings)."""
        # An unarmed process only reports the plan unless explicitly allowed to act.
        may_act = not self.live.dry_run and (
            live_trading_armed() or self.live.allow_unarmed_liquidation
        )
        if not may_act and not self.live.dry_run:
            result.notes.append(
                "liquidation NOT executed: process is unarmed and "
                "allow_unarmed_liquidation is False — reporting plan only"
            )
        self._audit(f"{result.date.isoformat()} LIQUIDATE_INTENT reason={reason!r}")
        # Always pass an explicit symbol set — NEVER fall back to None ("everything").
        # An empty sleeve must close nothing, not the operator's whole account.
        report = liquidate_all(
            self.broker, reason=reason, dry_run=not may_act,
            symbols=self.state.sleeve_symbols(),
        )
        result.liquidation = report
        result.halted = True
        if report.complete:
            self.state.sleeve_positions.clear()
            self.state.liquidation_verified_flat = True
        elif report.residual:
            result.notes.append(f"LIQUIDATION INCOMPLETE — residual {report.residual}")
        self._audit(
            f"{result.date.isoformat()} LIQUIDATE_RESULT {report.model_dump_json()}"
        )
        self._persist()

    # -------------------------------------------------------------------- cycle
    def run_cycle(self, universe: list[str], date: datetime) -> CycleResult:
        """One live cycle. Kill switch → cap → decide → validate → route."""
        result = CycleResult(date=date, dry_run=self.live.dry_run)
        uni = [t.strip().upper() for t in universe]
        uni_set = set(uni)
        clock = AsOfClock.at(date)

        # 1) Kill switch BEFORE anything else.
        if kill_switch_active():
            self.risk.halt.check_kill_switch()
            if self.state.liquidation_verified_flat:
                result.halted = True
                result.notes.append("kill switch engaged — already verified flat")
                return result
            self._liquidate("VTS_KILL_SWITCH engaged", result)
            return result

        # 2) A latched halt (daily loss / drawdown, possibly restored from disk).
        if self.risk.halt.halted:
            if self.state.liquidation_verified_flat:
                result.halted = True
                result.notes.append("risk halt latched — already verified flat")
                return result
            self._liquidate("; ".join(self.risk.halt.halt_reasons) or "halted", result)
            return result

        # 3) Capital cap re-derived from the broker EVERY cycle.
        account = self.broker.get_account()
        assert_capital_within_cap(self.live.allocated_capital, account.total_assets)
        if not self.live.dry_run:
            self.live.assert_armed_for_live(account.total_assets)

        # 4) Same decision pipeline as backtest/paper.
        held = self._sleeve_held(result)
        marks = self._marks(uni_set | set(held), clock)
        priced = {t: marks[t] for t in uni if t in marks}
        aggs = {
            t: sample_decisions(self.model, self.cache, t, clock, self.config.n_samples)
            for t in priced
        }
        verdict = self.risk.apply(aggs)
        result.forced_holds = sum(1 for g in verdict.gated.values() if g.forced_hold)
        if verdict.halted:
            result.halted = True
            result.notes.append("risk engine returned halted verdict")
            self._persist()
            return result
        target = {t: verdict.weights.get(t, 0.0) for t in uni}

        # 5) Orders sized against ALLOCATED capital, then checked in aggregate.
        orders = self._orders_for(
            target, marks, self.live.allocated_capital, held, uni_set, result
        )
        if len(orders) > self.live.max_orders_per_session:
            result.notes.append(
                f"order count {len(orders)} exceeds max_orders_per_session "
                f"{self.live.max_orders_per_session} — refusing to route"
            )
            self._audit(f"{date.isoformat()} REFUSED too_many_orders={len(orders)}")
            return result
        gross_buy = sum(o.notional for o in orders if o.side == "buy")
        if gross_buy > self.live.allocated_capital:
            result.notes.append(
                f"aggregate buy notional {gross_buy:,.2f} exceeds allocation "
                f"{self.live.allocated_capital:,.2f} — refusing to route"
            )
            self._audit(f"{date.isoformat()} REFUSED gross_notional={gross_buy:.2f}")
            return result

        # 6) Validate every order, then route (or report in dry run).
        #    Buys are validated against genuinely-available broker cash, never the
        #    proceeds of a sell submitted moments ago and not yet confirmed filled.
        broker_cash = account.cash
        running_cash = broker_cash
        positions = dict(held)
        for order in orders:
            adv = _dollar_adv(self.store, order.symbol, clock, self.config.adv_lookback)
            est_cost = self.cost_model.cost(order.notional, adv, order.side).total
            usable_cash = min(broker_cash, running_cash)
            if usable_cash < 0:
                result.notes.append("running cash went negative — halting this batch")
                break
            rejection = validate_order(
                order, AccountState(cash=usable_cash, positions=positions),
                self.venue, cash_buffer=est_cost,
            )
            if rejection is not None:
                result.rejections.append(rejection)
                self._audit(f"{date.isoformat()} REJECT {order.symbol} {rejection.violations}")
                continue
            if self.live.dry_run:
                result.would_submit.append(order)
            else:
                try:
                    # Re-check arming per order: disarming mid-batch stops routing.
                    self.live.assert_armed_for_live(account.total_assets)
                    placed = self.broker.submit_order(
                        order.symbol, order.side, order.qty, limit_price=order.limit_price
                    )
                    if not placed.accepted:
                        result.notes.append(
                            f"{order.symbol} rejected by broker: status={placed.status!r}"
                        )
                        self._audit(f"{date.isoformat()} BROKER_REJECT {order.symbol}")
                        continue
                except Exception as exc:
                    # Never lose the record of what already went live.
                    result.notes.append(f"submit failed for {order.symbol}: {exc}")
                    self._audit(f"{date.isoformat()} SUBMIT_ERROR {order.symbol} {exc}")
                    self._persist()
                    return result
                result.submitted.append(order)
                self.state.apply_fill(order.symbol, order.side, order.qty)
                self._audit(
                    f"{date.isoformat()} SENT {order.side} {order.qty:g} {order.symbol} "
                    f"@{order.limit_price:g}"
                )
            signed = order.qty if order.side == "buy" else -order.qty
            positions[order.symbol] = positions.get(order.symbol, 0.0) + signed
            running_cash -= signed * order.limit_price + est_cost

        self._persist()
        return result
