"""LLM call/cost accounting for the mandated cost metrics.

Tracks every model invocation the backtest makes: real API calls (cache misses)
vs cache hits, and dollar cost. Adapters that know their true per-call cost
report it via ``record_call(cost_usd=...)``; otherwise the configured estimate is
used. Feeds the Step 3 report's ``LLM 호출당 비용`` (cost per call) and
``결정 1건당 총 비용`` (total cost per decision) and the monthly-budget check.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass


@dataclass
class LLMCostTracker:
    """Mutable, thread-safe call/cost counter for one backtest run."""

    est_cost_per_call_usd: float = 0.0
    calls: int = 0
    cache_hits: int = 0
    total_cost_usd: float = 0.0

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def record_call(self, cost_usd: float | None = None) -> None:
        """One real model invocation (a cache miss). ``None`` cost uses the estimate."""
        with self._lock:
            self.calls += 1
            self.total_cost_usd += self.est_cost_per_call_usd if cost_usd is None else cost_usd

    def record_cache_hit(self) -> None:
        with self._lock:
            self.cache_hits += 1

    @property
    def cost_per_call(self) -> float | None:
        """Average cost of a real call; None when no real calls were made."""
        return self.total_cost_usd / self.calls if self.calls else None

    def cost_per_decision(self, n_decisions: int) -> float | None:
        """Total run cost divided by aggregated decisions; None when no decisions."""
        return self.total_cost_usd / n_decisions if n_decisions else None
