"""Halt state: daily-loss auto-stop, cumulative drawdown stop, and the kill switch.

Three independent, deterministic stops:

- **Daily loss halt** (일일 손실 한도): the halt latches when a **single-day**
  mark-to-market return reaches ``-daily_loss_limit``. The caller must feed
  genuine daily marks (the backtest engine iterates the point-in-time store's
  daily bars between decision dates, so the "daily" bound is daily regardless of
  rebalance cadence — a single-day crash is caught even under weekly rebalancing).
- **Cumulative drawdown halt** (from-peak): when ``max_drawdown_limit`` is set, a
  peak-to-current equity drop of that fraction latches — catching slow bleeds
  that never breach the per-day limit.
- **Kill switch** (env var): ``VTS_KILL_SWITCH`` set to a truthy value forces
  flat. Read fresh from the environment each check; flipping it needs shell
  access, not model output.

Every latch requires an explicit named human :meth:`HaltState.reset` to clear — a
good day never un-latches an automatic stop. Distinct stop conditions each record
a reason (deduplicated), so the audit trail shows every reason a halt held.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

# Explicitly-false spellings; ANY other non-empty value engages the kill switch.
# Fail-safe: an unrecognized emergency value (2, STOP, kill, ...) still stops
# rather than being silently ignored.
_EXPLICIT_FALSE = {"", "0", "false", "no", "off", "none", "disable", "disabled"}

KILL_SWITCH_ENV = "VTS_KILL_SWITCH"


def kill_switch_active(env: dict[str, str] | None = None) -> bool:
    """Whether the kill switch is engaged (any non-empty, non-false value)."""
    source = env if env is not None else os.environ
    return source.get(KILL_SWITCH_ENV, "").strip().lower() not in _EXPLICIT_FALSE


@dataclass
class HaltState:
    """Mutable halt latch owned by the risk engine (never by an agent)."""

    daily_loss_limit: float
    max_drawdown_limit: float | None = None
    halted: bool = False
    halt_reasons: list[str] = field(default_factory=list)
    _peak_equity: float = 0.0

    def _latch(self, reason: str) -> None:
        """Record a distinct breach reason (deduplicated) and latch the halt.

        Recording is decoupled from the latch transition so a second, different
        stop condition arising while already halted is still logged — the audit
        trail must show every reason a halt held, not just the first.
        """
        if reason not in self.halt_reasons:
            self.halt_reasons.append(reason)
        self.halted = True

    def observe_daily_return(self, daily_return: float) -> bool:
        """Feed one SINGLE-DAY mark-to-market return; latch on a breach.

        ``<=`` so a loss of exactly the limit halts — the limit is a ceiling.
        """
        if daily_return <= -self.daily_loss_limit:
            self._latch(
                f"daily_loss_limit: daily return {daily_return:.2%} <= "
                f"-{self.daily_loss_limit:.2%}"
            )
        return self.halted

    def observe_equity(self, equity: float) -> bool:
        """Feed the running equity to update peak and check drawdown-from-peak."""
        self._peak_equity = max(self._peak_equity, equity)
        if self.max_drawdown_limit is not None and self._peak_equity > 0:
            drawdown = 1.0 - equity / self._peak_equity
            if drawdown >= self.max_drawdown_limit:
                self._latch(
                    f"max_drawdown: {drawdown:.2%} >= {self.max_drawdown_limit:.2%} from peak"
                )
        return self.halted

    def check_kill_switch(self, env: dict[str, str] | None = None) -> bool:
        """Latch the halt if the environment kill switch is engaged."""
        if kill_switch_active(env):
            self._latch(f"kill_switch: {KILL_SWITCH_ENV} engaged")
        return self.halted

    def reset(self, *, operator: str) -> None:
        """Explicit human reset. ``operator`` is recorded so the audit trail shows
        who un-latched a halt; there is no argument-less auto-reset on purpose."""
        self.halt_reasons.append(f"reset by {operator}")
        self.halted = False
        self._peak_equity = 0.0
