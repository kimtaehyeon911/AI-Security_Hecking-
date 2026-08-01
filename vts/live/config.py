"""Live-trading configuration: hardcoded capital cap and the arming gate.

The mandate: *"드라이런 플래그 기본값 True … 최초 자본은 총자산의 1% 이하로
하드코딩."* Both are enforced here as module constants, not settings:

- :data:`MAX_INITIAL_CAPITAL_FRACTION` is a ``Final`` module constant with **no
  environment override and no config field**. Env vars can only make the
  allocation smaller (by allocating less); nothing can raise the ceiling without
  a code change that shows up in review.
- ``dry_run`` defaults to True on every model and constructor in this package.

Arming uses an exact string token rather than a boolean because a stray
``VTS_LIVE=1`` in a shell profile must never be enough to move real money.
"""

from __future__ import annotations

import os
from typing import Final

from pydantic import BaseModel, ConfigDict, Field

#: Hardcoded ceiling: initial live capital may not exceed this fraction of total
#: assets. Deliberately a module constant — no env var, no config field, no
#: setter. Changing it requires editing this line in a reviewed commit.
MAX_INITIAL_CAPITAL_FRACTION: Final[float] = 0.01  # 1%

LIVE_ARM_ENV: Final[str] = "VTS_LIVE_TRADING_ARMED"
#: The exact value ``LIVE_ARM_ENV`` must hold. Not a boolean, by design.
LIVE_ARM_TOKEN: Final[str] = "I_UNDERSTAND_THE_RISK"


class LiveNotArmed(RuntimeError):
    """Raised when a live action is attempted without the full arming sequence."""


class CapitalCapExceeded(RuntimeError):
    """Raised when requested capital exceeds the hardcoded fraction of total assets."""


def live_trading_armed(env: dict[str, str] | None = None) -> bool:
    """Whether the arming env var holds the exact token (read fresh each call)."""
    source = env if env is not None else os.environ
    return source.get(LIVE_ARM_ENV, "").strip() == LIVE_ARM_TOKEN


def assert_capital_within_cap(allocated_capital: float, total_assets: float) -> None:
    """Raise :class:`CapitalCapExceeded` unless allocation is within the hard cap.

    Non-positive total assets can never satisfy the cap — refuse rather than
    divide-by-zero into a permissive answer.
    """
    if allocated_capital <= 0:
        raise CapitalCapExceeded(f"allocated capital must be positive, got {allocated_capital}")
    if total_assets <= 0:
        raise CapitalCapExceeded(
            f"total assets must be positive to size an allocation, got {total_assets}"
        )
    cap = total_assets * MAX_INITIAL_CAPITAL_FRACTION
    if allocated_capital > cap:
        raise CapitalCapExceeded(
            f"allocated capital {allocated_capital:,.2f} exceeds the hardcoded cap "
            f"{cap:,.2f} ({MAX_INITIAL_CAPITAL_FRACTION:.1%} of total assets "
            f"{total_assets:,.2f})"
        )


class LiveConfig(BaseModel):
    """Frozen live-session configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    allocated_capital: float = Field(
        gt=0.0,
        description=(
            "Capital this session may deploy. Checked against the hardcoded "
            "≤1%-of-total-assets cap at arm time and before every session."
        ),
    )
    dry_run: bool = Field(
        default=True,
        description="True simulates routing; False requires the arming token too.",
    )
    max_orders_per_session: int = Field(
        default=20, ge=1,
        description="Circuit breaker on runaway order generation in one session.",
    )

    def assert_armed_for_live(self, total_assets: float, env: dict[str, str] | None = None) -> None:
        """Verify every condition required to send a real order. Raises otherwise."""
        if self.dry_run:
            raise LiveNotArmed("dry_run is True; live routing is disabled")
        if not live_trading_armed(env):
            raise LiveNotArmed(
                f"{LIVE_ARM_ENV} is not set to the arming token; refusing to trade live"
            )
        assert_capital_within_cap(self.allocated_capital, total_assets)
