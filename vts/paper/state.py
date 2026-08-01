"""Persistent paper-trading state.

An 8-week paper run spans many process lifetimes (one invocation per trading
day), so the portfolio — cash, share positions, equity history, the halt latch,
and the shortfall log — is serialized to disk after every step and rehydrated on
resume. The halt latch is persisted too: a run that auto-stopped must stay
stopped across a restart, not silently resume trading.
"""

from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class PaperState(BaseModel):
    """Mutable portfolio + audit state for a paper-trading run."""

    model_config = ConfigDict(extra="forbid")

    initial_capital: float
    cash: float
    positions: dict[str, float] = Field(default_factory=dict)  # symbol -> shares
    dry_run: bool = True

    equity_curve: list[tuple[str, float]] = Field(default_factory=list)   # (iso, equity)
    decision_log: list[dict] = Field(default_factory=list)
    shortfall_log: list[dict] = Field(default_factory=list)
    processed_dates: list[str] = Field(default_factory=list)

    # Halt latch snapshot (mirrors vts.risk.HaltState so a resumed run stays halted).
    halted: bool = False
    halt_reasons: list[str] = Field(default_factory=list)
    peak_equity: float = 0.0

    @classmethod
    def new(cls, initial_capital: float, *, dry_run: bool = True) -> PaperState:
        return cls(
            initial_capital=initial_capital, cash=initial_capital, dry_run=dry_run,
            peak_equity=initial_capital,
        )

    def mark_to_market(self, prices: dict[str, float]) -> float:
        """Equity = cash + Σ shares·price.

        The caller MUST supply a price for every held symbol (the paper loop
        forward-fills the last known close via ``_last_close`` for held names even
        when they have left the active universe). A held symbol missing from
        ``prices`` is a caller bug, not a zero mark — raise rather than silently
        crater equity (which would false-trigger the loss halt)."""
        holdings = 0.0
        for sym, sh in self.positions.items():
            if sym not in prices:
                raise KeyError(
                    f"no mark for held symbol {sym!r}; the caller must forward-fill "
                    f"a price for every held position before marking to market"
                )
            holdings += sh * prices[sym]
        return self.cash + holdings

    def last_processed(self) -> str | None:
        return self.processed_dates[-1] if self.processed_dates else None

    # --------------------------------------------------------------- persistence
    def save(self, path: str | Path) -> None:
        """Atomically persist: write a temp file, fsync, then os.replace.

        A trading-state file is rewritten after every step; an in-place truncate
        would leave a corrupt, unrecoverable file if the process dies mid-write.
        The tmp+rename makes the on-disk file always either the old or the new
        complete state, never a partial one."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as fh:
            fh.write(self.model_dump_json(indent=2))
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)

    @classmethod
    def load(cls, path: str | Path) -> PaperState:
        return cls.model_validate_json(Path(path).read_text(encoding="utf-8"))

    @classmethod
    def load_or_new(
        cls, path: str | Path, initial_capital: float, *, dry_run: bool = True
    ) -> PaperState:
        p = Path(path)
        if p.exists():
            state = cls.load(p)
            # Resuming a run whose dry_run flag differs from the request is a
            # configuration error, not something to silently override.
            if state.dry_run != dry_run:
                raise ValueError(
                    f"persisted state dry_run={state.dry_run} != requested {dry_run}"
                )
            return state
        return cls.new(initial_capital, dry_run=dry_run)
