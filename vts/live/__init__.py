"""Live trading — the last step, gated behind an explicit safety envelope.

Nothing here routes real orders by default. Three independent conditions must ALL
hold before a single live order can be sent:

1. ``dry_run=False`` passed explicitly (every default in this package is True);
2. ``VTS_LIVE_TRADING_ARMED`` set to the exact arming token (a boolean would be
   too easy to set by accident);
3. the allocated capital within the **hardcoded** ≤1%-of-total-assets cap.

And one condition must NOT hold: the kill switch. ``VTS_KILL_SWITCH`` engages a
full liquidation (전량 청산) — open orders cancelled, every position closed — and
latches the halt so nothing trades again without a named operator reset.

Real broker adapters are deliberately NOT shipped here. :class:`BrokerClient` is
the interface an adapter must implement; it needs its own review and a live smoke
test on a paper endpoint before any real key is involved.
"""

from __future__ import annotations

from vts.live.broker import (
    Account,
    BrokerClient,
    BrokerOrder,
    Position,
    SimulatedBroker,
)
from vts.live.config import (
    LIVE_ARM_ENV,
    LIVE_ARM_TOKEN,
    MAX_INITIAL_CAPITAL_FRACTION,
    CapitalCapExceeded,
    LiveConfig,
    LiveNotArmed,
    assert_capital_within_cap,
    live_trading_armed,
)
from vts.live.liquidate import LiquidationReport, liquidate_all
from vts.live.state import LiveState
from vts.live.trader import LiveTrader

__all__ = [
    "Account",
    "BrokerClient",
    "BrokerOrder",
    "CapitalCapExceeded",
    "LIVE_ARM_ENV",
    "LIVE_ARM_TOKEN",
    "LiquidationReport",
    "LiveConfig",
    "LiveNotArmed",
    "LiveState",
    "LiveTrader",
    "MAX_INITIAL_CAPITAL_FRACTION",
    "Position",
    "SimulatedBroker",
    "assert_capital_within_cap",
    "live_trading_armed",
    "liquidate_all",
]
