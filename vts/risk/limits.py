"""Risk limits and venue rules — frozen configuration, never model output.

Both models are ``frozen`` pydantic models with ``extra="forbid"``: once
constructed at startup they cannot be mutated, and unknown fields (e.g. an agent
"suggesting" a new limit key) are rejected at validation. The only legitimate
sources of these values are code defaults and ``VTS_RISK_*`` environment
variables — there is deliberately NO constructor path that accepts agent/LLM
output, and the risk engine takes limits at __init__ time only.
"""

from __future__ import annotations

import os
from decimal import Decimal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class RiskLimits(BaseModel):
    """Hard limits applied to every decision batch. All fractions of equity."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_weight_per_symbol: float = Field(
        default=0.20, gt=0.0, le=1.0,
        description="종목당 최대 비중 — per-symbol weight ceiling.",
    )
    max_gross_exposure: float = Field(
        default=1.0, gt=0.0, le=2.0,
        description="총 노출 한도 — ceiling on sum(|weights|). 1.0 = unlevered.",
    )
    daily_loss_limit: float = Field(
        default=0.03, gt=0.0, le=1.0,
        description=(
            "일일 손실 한도 — a SINGLE-DAY mark-to-market loss of this fraction halts "
            "trading (targets go flat, latched). Fed genuine daily marks by the "
            "engine, so it is a true daily bound at any rebalance cadence."
        ),
    )
    max_drawdown_limit: float | None = Field(
        default=0.20, gt=0.0, le=1.0,
        description=(
            "Cumulative peak-to-current drawdown that latches the halt — catches "
            "slow bleeds that never breach the per-day limit. None disables it."
        ),
    )
    min_confidence: float = Field(
        default=0.30, ge=0.0, le=1.0,
        description="Below this mean confidence the decision is forced to Hold.",
    )
    min_agreement: float = Field(
        default=0.50, ge=0.0, le=1.0,
        description="Below this vote agreement the decision is forced to Hold.",
    )
    max_dispersion: float = Field(
        default=1.5, ge=0.0,
        description="Above this ordinal rating dispersion the decision is forced to Hold.",
    )
    hold_on_single_sample: bool = Field(
        default=False,
        description=(
            "When True, a single-sample decision (N=1) is forced to Hold because "
            "agreement/dispersion cannot be assessed. Off by default so "
            "deterministic N=1 research runs are not neutered."
        ),
    )

    @classmethod
    def from_env(cls) -> RiskLimits:
        """Load limits from ``VTS_RISK_*`` env vars over the code defaults."""
        env = os.environ
        raw: dict[str, float] = {}
        mapping = {
            "VTS_RISK_MAX_WEIGHT": "max_weight_per_symbol",
            "VTS_RISK_MAX_GROSS": "max_gross_exposure",
            "VTS_RISK_DAILY_LOSS_LIMIT": "daily_loss_limit",
            "VTS_RISK_MAX_DRAWDOWN": "max_drawdown_limit",
            "VTS_RISK_MIN_CONFIDENCE": "min_confidence",
            "VTS_RISK_MIN_AGREEMENT": "min_agreement",
            "VTS_RISK_MAX_DISPERSION": "max_dispersion",
        }
        for var, field in mapping.items():
            if env.get(var) is not None and env[var].strip() != "":
                try:
                    raw[field] = float(env[var])
                except ValueError as exc:
                    raise ValueError(f"invalid numeric value for {var}: {env[var]!r}") from exc
        return cls(**raw)


class VenueRules(BaseModel):
    """Order-validation rules for one venue (tick size, lot size, min notional).

    ``tick_ladder`` is a list of ``(upper_bound_exclusive, tick)`` rows sorted by
    bound; the tick for a price is the first row whose bound exceeds it (KRX-style
    price-banded ticks; a flat-tick venue like US equities uses one row).
    Values are strings converted to Decimal so tick arithmetic is exact.

    ``lot_size`` is stored as an exact Decimal STRING (crypto steps like
    ``"0.00001"`` cannot be represented as a binary float without artifacts; an
    int like ``10`` is accepted and normalized). Use :meth:`quantize_qty` to snap
    a raw quantity onto the step grid — it truncates toward zero, so a quantized
    position can never exceed the intended magnitude for longs or shorts.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = "us_equity"
    tick_ladder: tuple[tuple[float, str], ...] = ((float("inf"), "0.01"),)
    lot_size: str = "1"
    min_notional: float = Field(default=1.0, ge=0.0)

    @field_validator("lot_size", mode="before")
    @classmethod
    def _lot_to_exact_string(cls, v) -> str:
        try:
            step = Decimal(str(v))
        except Exception as exc:  # noqa: BLE001 - normalize to a clear error
            raise ValueError(f"lot_size must be numeric, got {v!r}") from exc
        if not step.is_finite() or step <= 0:
            raise ValueError(f"lot_size must be a positive finite number, got {v!r}")
        return format(step, "f")

    @property
    def lot_step(self) -> Decimal:
        return Decimal(self.lot_size)

    def quantize_qty(self, raw: float) -> float:
        """Snap ``raw`` onto the lot grid, truncating toward zero (never rounds up)."""
        from decimal import ROUND_DOWN

        step = self.lot_step
        units = (Decimal(str(raw)) / step).to_integral_value(rounding=ROUND_DOWN)
        return float(units * step)

    @field_validator("tick_ladder")
    @classmethod
    def _ladder_strictly_increasing_positive(cls, v):
        if not v:
            raise ValueError("tick_ladder must be non-empty")
        bounds = [b for b, _ in v]
        # Strictly increasing: equal adjacent bounds would make the later row's
        # tick unreachable (first-match wins), silently collapsing a price band.
        if any(b2 <= b1 for b1, b2 in zip(bounds, bounds[1:])):
            raise ValueError("tick_ladder bounds must be strictly increasing")
        if any(Decimal(t) <= 0 for _, t in v):
            raise ValueError("ticks must be positive")
        if bounds[-1] != float("inf"):
            raise ValueError("last ladder row must have an infinite upper bound")
        return v

    def tick_for(self, price: float) -> Decimal:
        for bound, tick in self.tick_ladder:
            if price < bound:
                return Decimal(tick)
        return Decimal(self.tick_ladder[-1][1])  # pragma: no cover - inf bound guards


# KRX price-band tick ladder (KOSPI, 2023 revision) as a ready-made example for
# the kr_equity asset class. Bounds in KRW.
KRX_KOSPI = VenueRules(
    name="krx_kospi",
    tick_ladder=(
        (2_000.0, "1"),
        (5_000.0, "5"),
        (20_000.0, "10"),
        (50_000.0, "50"),
        (200_000.0, "100"),
        (500_000.0, "500"),
        (float("inf"), "1000"),
    ),
    lot_size=1,
    min_notional=0.0,
)

US_EQUITY = VenueRules(name="us_equity")

# Conservative Binance-spot fallback (BTCUSDT-magnitude filters). Real trading
# should build per-symbol rules from /api/v3/exchangeInfo via
# ``vts.live.binance_broker.venue_from_exchange_filters`` — Binance filters vary
# per symbol and drift over time; this default only keeps offline runs sane.
BINANCE_SPOT_DEFAULT = VenueRules(
    name="binance_spot",
    tick_ladder=((float("inf"), "0.01"),),
    lot_size="0.00001",
    min_notional=5.0,
)
