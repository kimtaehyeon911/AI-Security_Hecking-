"""Durable live-session state.

Two things must survive a process restart or the safety envelope is theatre:

- the **halt latch**. A process-local latch means a crash-and-restart with the
  kill switch since cleared resumes trading as if the halt never happened.
- the **sleeve ledger** — what THIS system believes it owns. The broker reports
  the whole account, which may hold an operator's unrelated positions; reconciling
  against raw broker state would let the trader "flatten" the entire book. Orders
  are computed against the sleeve, and liquidation is scoped to it.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class LiveState(BaseModel):
    """Persisted live state (halt latch + sleeve ledger)."""

    model_config = ConfigDict(extra="forbid")

    sleeve_positions: dict[str, float] = Field(
        default_factory=dict, description="symbol -> shares this system believes it owns"
    )
    halted: bool = False
    halt_reasons: list[str] = Field(default_factory=list)
    liquidation_verified_flat: bool = Field(
        default=False,
        description="True once a liquidation was verified complete; stops re-liquidating.",
    )
    dust_symbols: list[str] = Field(
        default_factory=list,
        description="Sleeve positions too small to exit (sub-lot / sub-min-notional), noted once.",
    )

    def sleeve_symbols(self) -> set[str]:
        return {s for s, q in self.sleeve_positions.items() if q != 0.0}

    def apply_fill(self, symbol: str, side: str, qty: float) -> None:
        sym = symbol.strip().upper()
        signed = qty if side == "buy" else -qty
        self.sleeve_positions[sym] = self.sleeve_positions.get(sym, 0.0) + signed
        if abs(self.sleeve_positions[sym]) < 1e-9:
            self.sleeve_positions.pop(sym, None)

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Atomic write (tmp + fsync + replace) — a torn state file is unrecoverable."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)

    @classmethod
    def load_or_new(cls, path: str | Path | None) -> LiveState:
        if path is None:
            return cls()
        p = Path(path)
        return cls.model_validate_json(p.read_text(encoding="utf-8")) if p.exists() else cls()
