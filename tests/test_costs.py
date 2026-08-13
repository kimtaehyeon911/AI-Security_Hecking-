"""Transaction cost model: commission + tax + participation slippage."""

from __future__ import annotations

import pytest

from vts.backtest.costs import CostModel, CostParams


def test_zero_notional_zero_cost():
    c = CostModel().cost(0.0, dollar_adv=1e9, side="buy")
    assert c.total == 0.0


def test_cost_monotonic_in_notional():
    m = CostModel()
    small = m.cost(10_000, dollar_adv=1e8, side="buy").total
    big = m.cost(100_000, dollar_adv=1e8, side="buy").total
    assert big > small


def test_slippage_rises_with_participation():
    m = CostModel()
    thin = m.cost(1_000_000, dollar_adv=1e6, side="buy")   # 100% participation
    liquid = m.cost(1_000_000, dollar_adv=1e9, side="buy")  # 0.1% participation
    assert thin.slippage > liquid.slippage
    assert thin.participation > liquid.participation


def test_slippage_capped():
    p = CostParams(max_slippage_bps=50.0, impact_coef_bps=1000.0)
    m = CostModel(p)
    c = m.cost(1e9, dollar_adv=1.0, side="buy")  # absurd participation
    assert c.slippage <= 1e9 * 50.0 / 1e4 + 1e-6


def test_tax_only_on_sell():
    m = CostModel(CostParams(sell_tax_bps=23.0))
    buy = m.cost(100_000, dollar_adv=1e8, side="buy")
    sell = m.cost(100_000, dollar_adv=1e8, side="sell")
    assert buy.tax == 0.0
    assert sell.tax == pytest.approx(100_000 * 23.0 / 1e4)


def test_min_adv_floor_prevents_div_by_zero():
    m = CostModel()
    c = m.cost(1000, dollar_adv=0.0, side="buy")  # no ADV data
    assert c.slippage == c.slippage  # not NaN/inf
    assert c.total < 1000  # sane
