"""Unit tests for the forward-causal synthetic generation helpers.

Pure-numpy: no DB, no SDV fitting, no MLflow. Fast — these run in milliseconds.
"""
from __future__ import annotations

import numpy as np
import pytest

from src.synthetic import benchmarks as B
from src.synthetic.user_augmentation import (
    decide_purchase,
    derive_ad_views_d7,
    derive_sessions_d7,
    sample_engagement_potential,
)


rng = np.random.default_rng(42)


def test_engagement_potential_is_positive():
    """LogNormal draws must be strictly positive."""
    mult = np.ones(1000)
    engagement = sample_engagement_potential(1000, mult)
    assert (engagement > 0).all()


def test_engagement_potential_channel_multiplier_shifts_mean():
    """High-multiplier channels must produce higher-engagement users on average."""
    np.random.seed(42)
    high = sample_engagement_potential(5000, np.full(5000, 1.4))
    np.random.seed(42)
    low  = sample_engagement_potential(5000, np.full(5000, 0.7))
    assert high.mean() > low.mean() * 1.5


def test_sessions_grow_with_engagement():
    """Sessions_d7 must be monotonic-ish in engagement."""
    engagement = np.array([0.1, 0.5, 1.0, 2.0, 5.0, 10.0])
    sessions = derive_sessions_d7(engagement)
    # Not strictly monotone (Poisson noise), but the trend must hold at the ends.
    assert sessions[-1] > sessions[0]


def test_ad_views_grow_with_sessions():
    """More sessions → more ad impressions (Poisson-driven)."""
    sessions_low  = np.full(1000, 2)
    sessions_high = np.full(1000, 20)
    assert derive_ad_views_d7(sessions_high).mean() > derive_ad_views_d7(sessions_low).mean()


def test_purchase_probability_grows_with_engagement():
    """Sigmoid: higher engagement ⇒ higher purchase probability."""
    engagement_low  = np.full(5000, 0.5)
    engagement_high = np.full(5000, 5.0)
    low_rate  = decide_purchase(engagement_low).mean()
    high_rate = decide_purchase(engagement_high).mean()
    assert high_rate > low_rate


def test_purchase_probability_is_binary():
    """decide_purchase returns 0/1 ints only."""
    out = decide_purchase(np.linspace(0.1, 10, 100))
    assert set(np.unique(out)).issubset({0, 1})


def test_conversion_close_to_target():
    """At the industry benchmark engagement mix, conversion ≈ target ±30%."""
    # Baseline engagement distribution: mean ≈ 1.0 (LogNormal(0,1) has median 1).
    engagement = sample_engagement_potential(20_000, np.ones(20_000))
    conversion = decide_purchase(engagement).mean()
    # NOTE: this is a very loose bound — the point is to catch order-of-magnitude
    # regressions (e.g. if someone flips the sigmoid sign), not fine calibration.
    assert 0.005 < conversion < 0.20, f"conversion out of sanity range: {conversion:.4f}"


class TestIndustryConstants:
    """These pull from `benchmarks.py` — no state, only guard against typos."""

    def test_rewarded_completion_reasonable(self):
        # Rewarded-video completion in the wild is 75-95%.
        from src.routers.decide import REWARDED_COMPLETION_RATE
        assert 0.7 <= REWARDED_COMPLETION_RATE <= 0.95

    def test_iap_min_p_payer_reasonable(self):
        """The IAP eligibility floor should be low enough to fire on real payers,
        high enough to gate marginal ones. 0.10-0.40 is the sensible band."""
        from src.routers.decide import IAP_MIN_P_PAYER
        assert 0.10 <= IAP_MIN_P_PAYER <= 0.40
