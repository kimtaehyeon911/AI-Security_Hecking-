"""Majority-vote aggregation and dispersion."""

from __future__ import annotations

import pytest

from vts.decision import (
    Decision,
    Rating,
    aggregate_decisions,
    rating_to_signed_weight,
)


def _d(rating: Rating, conf: float = 0.5) -> Decision:
    return Decision(ticker="AAPL", as_of="2024-06-03T21:00:00+00:00", rating=rating, confidence=conf)


def test_majority_wins_and_dispersion_positive():
    agg = aggregate_decisions([_d(Rating.BUY), _d(Rating.BUY), _d(Rating.SELL)])
    assert agg.rating == Rating.BUY
    assert agg.agreement == pytest.approx(2 / 3)
    assert agg.dispersion > 0
    assert agg.n_samples == 3


def test_unanimous_has_zero_dispersion():
    agg = aggregate_decisions([_d(Rating.OVERWEIGHT)] * 3)
    assert agg.rating == Rating.OVERWEIGHT
    assert agg.dispersion == 0.0
    assert agg.agreement == 1.0


def test_tie_breaks_to_hold():
    agg = aggregate_decisions([_d(Rating.BUY), _d(Rating.SELL)])
    assert agg.rating == Rating.HOLD
    assert agg.tie_broken_to_hold is True


def test_mean_confidence_and_votes():
    agg = aggregate_decisions([_d(Rating.BUY, 0.9), _d(Rating.BUY, 0.7), _d(Rating.HOLD, 0.2)])
    assert agg.mean_confidence == pytest.approx((0.9 + 0.7 + 0.2) / 3)
    assert agg.votes == {"Buy": 2, "Hold": 1}


def test_empty_rejected():
    with pytest.raises(ValueError):
        aggregate_decisions([])


def test_signed_weights_scale():
    assert rating_to_signed_weight(Rating.BUY) == 1.0
    assert rating_to_signed_weight(Rating.OVERWEIGHT) == 0.5
    assert rating_to_signed_weight(Rating.HOLD) == 0.0
    assert rating_to_signed_weight(Rating.UNDERWEIGHT) == -0.5
    assert rating_to_signed_weight(Rating.SELL) == -1.0
