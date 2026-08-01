"""Halt state: daily-loss auto-stop and the environment kill switch.

Two independent stop mechanisms, both deterministic:

- **Daily loss halt** (일일 손실 한도): when the mark-to-market loss since the
  previous decision point reaches ``daily_loss_limit``, the halt LATCHES — targets
  go flat and stay flat until a human calls :meth:`HaltState.reset`. A latch, not
  a per-day reset, because an automated resume after a limit breach is exactly the
  kind of decision that must not be automatic.
- **Kill switch** (env var, groundwork for Step 6): ``VTS_KILL_SWITCH`` set to a
  truthy value forces flat targets on every evaluation. It is read from the
  environment each time — flipping it requires shell access, not model output.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

_TRUTHY = {"1", "true", "yes", "on"}

KILL_SWITCH_ENV = "VTS_KILL_SWITCH"


def kill_switch_active(env: dict[str, str] | None = None) -> bool:
    """Whether the environment kill switch is engaged (checked fresh every call)."""
    source = env if env is not None else os.environ
    return source.get(KILL_SWITCH_ENV, "").strip().lower() in _TRUTHY


@dataclass
class HaltState:
    """Mutable halt latch owned by the risk engine (never by an agent)."""

    daily_loss_limit: float
    halted: bool = False
    halt_reasons: list[str] = field(default_factory=list)

    def observe_period_return(self, period_return: float) -> bool:
        """Feed one mark-to-market period return; latch the halt on a breach.

        Returns the (possibly new) halted state. The comparison is ``<=`` so a
        loss of exactly the limit halts — the limit is a ceiling, not a target.
        """
        if period_return <= -self.daily_loss_limit and not self.halted:
            self.halted = True
            self.halt_reasons.append(
                f"daily_loss_limit: period return {period_return:.2%} <= "
                f"-{self.daily_loss_limit:.2%}"
            )
        return self.halted

    def check_kill_switch(self, env: dict[str, str] | None = None) -> bool:
        """Latch the halt if the environment kill switch is engaged."""
        if kill_switch_active(env) and not self.halted:
            self.halted = True
            self.halt_reasons.append(f"kill_switch: {KILL_SWITCH_ENV} engaged")
        return self.halted

    def reset(self, *, operator: str) -> None:
        """Explicit human reset. ``operator`` is recorded so the audit trail shows
        who un-latched a halt; there is no argument-less auto-reset on purpose."""
        self.halt_reasons.append(f"reset by {operator}")
        self.halted = False
