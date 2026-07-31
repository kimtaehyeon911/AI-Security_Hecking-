"""Transaction cost model: commission + tax + participation-based slippage.

Step 2 mandate: *"비용 모델 필수: 수수료 + 세금 + 슬리피지(거래대금 대비 참여율 기반)."*

Slippage uses the standard square-root market-impact form
``impact_bps = coef * sqrt(participation)`` where ``participation = order_notional /
dollar_ADV`` — bigger orders relative to average daily volume pay more. Every
coefficient is an explicit, documented assumption (defaults are deliberately
conservative), overridable per venue. Tax is charged sell-side only (a US-equity
default of 0; set e.g. KR ~23 bps for KRX).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

Side = Literal["buy", "sell"]


@dataclass(frozen=True, slots=True)
class CostParams:
    """Cost coefficients. bps = basis points (1 bp = 0.01%)."""

    commission_bps: float = 1.0        # per-side broker commission
    sell_tax_bps: float = 0.0          # transaction tax, sell-side only (US=0; KRX~23)
    half_spread_bps: float = 2.0       # half the quoted bid-ask spread, paid every fill
    impact_coef_bps: float = 10.0      # square-root impact coefficient
    max_slippage_bps: float = 200.0    # cap so a thin-ADV day can't produce absurd cost
    min_dollar_adv: float = 1.0        # floor to avoid divide-by-zero on missing ADV


@dataclass(frozen=True, slots=True)
class TradeCost:
    """Breakdown of the cost of one fill, in currency units."""

    commission: float
    tax: float
    slippage: float
    participation: float

    @property
    def total(self) -> float:
        return self.commission + self.tax + self.slippage


class CostModel:
    """Computes the currency cost of a fill given its notional and the day's ADV."""

    def __init__(self, params: CostParams | None = None) -> None:
        self.p = params or CostParams()

    def participation(self, notional: float, dollar_adv: float) -> float:
        adv = max(dollar_adv, self.p.min_dollar_adv)
        return abs(notional) / adv

    def slippage_bps(self, participation: float) -> float:
        raw = self.p.half_spread_bps + self.p.impact_coef_bps * math.sqrt(max(participation, 0.0))
        return min(raw, self.p.max_slippage_bps)

    def cost(self, notional: float, dollar_adv: float, side: Side) -> TradeCost:
        """Return the cost breakdown for trading ``|notional|`` currency at ``side``."""
        notional = abs(notional)
        part = self.participation(notional, dollar_adv)
        commission = notional * self.p.commission_bps / 1e4
        tax = notional * self.p.sell_tax_bps / 1e4 if side == "sell" else 0.0
        slippage = notional * self.slippage_bps(part) / 1e4
        return TradeCost(commission=commission, tax=tax, slippage=slippage, participation=part)
