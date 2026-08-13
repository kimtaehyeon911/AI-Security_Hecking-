"""Paper-trading loop — the SAME decision/risk code path as the backtest, run
forward on a live clock against a simulated broker.

Priority-order note: this is step (2) of the plan, strictly between the backtest
harness and any real-money path. The loop reuses the exact Step 2–4 components
(``sample_decisions`` → ``RiskEngine`` → ``validate_order``); the only thing that
differs from the backtest is that weights become concrete integer-share orders
that pass real pre-trade validation and fill through a broker — which is where
implementation shortfall (백테스트 성과와의 괴리) comes from, logged daily.

``dry_run`` defaults to True everywhere; routing to a real broker is deferred to
Step 6 and refused here.
"""

from __future__ import annotations

from vts.paper.broker import FillResult, LiveTradingNotEnabled, PaperBroker
from vts.paper.loop import PaperTrader
from vts.paper.state import PaperState
from vts.paper.shortfall import ShortfallRecord

__all__ = [
    "FillResult",
    "LiveTradingNotEnabled",
    "PaperBroker",
    "PaperState",
    "PaperTrader",
    "ShortfallRecord",
]
