"""Unit tests for src/synthetic/benchmarks.py.

These are pure Python — no DB, no models, no network. Always safe to run.
Goal: catch silent regressions when tweaking industry constants.
"""
from __future__ import annotations

import pytest

from src.synthetic import benchmarks as B


class TestRetentionTargets:
    def test_shape(self):
        assert set(B.RETENTION_TARGETS) >= {"D1", "D7", "D30"}

    def test_monotone_decreasing(self):
        """D1 > D7 > D30 > D60 > D90 — retention only goes down over time."""
        assert B.RETENTION_TARGETS["D1"] > B.RETENTION_TARGETS["D7"]
        assert B.RETENTION_TARGETS["D7"] > B.RETENTION_TARGETS["D30"]

    def test_within_industry_bounds(self):
        """Hybrid-casual D1 usually 25-50%. Anything outside means we broke the calibration."""
        assert 0.20 < B.RETENTION_TARGETS["D1"] < 0.55


class TestSegmentDistribution:
    def test_probabilities_sum_to_one(self):
        assert abs(sum(B.SEGMENT_DISTRIBUTION.values()) - 1.0) < 1e-9

    def test_free_is_dominant(self):
        """The pareto structure: majority never pays."""
        assert B.SEGMENT_DISTRIBUTION["free"] >= 0.80

    def test_whale_is_rare(self):
        """Whales should be ≤ 3% (industry: ~1%)."""
        assert B.SEGMENT_DISTRIBUTION["whale"] <= 0.03


class TestSegmentLTVRanges:
    """Segment ranges are allowed to overlap (a $28 payer could be tagged as
    either dolphin or whale). But the ORDERING must hold — whale range starts
    higher and reaches higher than dolphin, dolphin higher than minnow."""

    def test_whale_range_shifts_higher_than_dolphin(self):
        whale_min, whale_max = B.SEGMENT_LTV_RANGE_USD["whale"]
        dolphin_min, dolphin_max = B.SEGMENT_LTV_RANGE_USD["dolphin"]
        assert whale_min >= dolphin_min
        assert whale_max >= dolphin_max

    def test_dolphin_range_shifts_higher_than_minnow(self):
        dolphin_min, dolphin_max = B.SEGMENT_LTV_RANGE_USD["dolphin"]
        minnow_min, minnow_max = B.SEGMENT_LTV_RANGE_USD["minnow"]
        assert dolphin_min >= minnow_min
        assert dolphin_max >= minnow_max

    def test_free_is_zero(self):
        assert B.SEGMENT_LTV_RANGE_USD["free"] == (0.0, 0.0)


class TestIAPWeights:
    @pytest.mark.parametrize("segment", ["whale", "dolphin", "minnow"])
    def test_weights_sum_to_one(self, segment):
        w = B.IAP_WEIGHTS_BY_SEGMENT[segment]
        assert abs(sum(w) - 1.0) < 1e-6

    def test_whales_prefer_expensive_tiers(self):
        """Whale distribution mass should be shifted toward the top price."""
        whale = B.IAP_WEIGHTS_BY_SEGMENT["whale"]
        minnow = B.IAP_WEIGHTS_BY_SEGMENT["minnow"]
        # Last two entries are the priciest packs ($19.99, $49.99).
        assert sum(whale[-2:]) > sum(minnow[-2:])


class TestECPM:
    def test_rewarded_beats_interstitial(self):
        """Industry pattern: opt-in rewarded video has the highest eCPM."""
        assert B.ECPM_USD["rewarded_video"] > B.ECPM_USD["interstitial"]

    def test_interstitial_beats_banner(self):
        assert B.ECPM_USD["interstitial"] > B.ECPM_USD["banner"]


class TestChannelMultipliers:
    def test_organic_beats_paid_engagement(self):
        """Organic users tend to be more engaged than TikTok impulse traffic."""
        assert (
            B.CHANNEL_ENGAGEMENT_MULTIPLIER["Organic"]
            > B.CHANNEL_ENGAGEMENT_MULTIPLIER["TikTok Ads"]
        )

    def test_organic_beats_tiktok_ltv(self):
        """Same for LTV — TikTok impulses spend less per user."""
        assert (
            B.CHANNEL_LTV_MULTIPLIER["Organic"]
            > B.CHANNEL_LTV_MULTIPLIER["TikTok Ads"]
        )


class TestConversionRate:
    def test_within_industry_range(self):
        """Hybrid-casual conversion is 5-15%. Outside this = calibration broken."""
        assert 0.05 < B.CONVERSION_RATE < 0.15


class TestWhaleCohortConfig:
    def test_cohort_size_reasonable(self):
        """v1 had 150 whales; v2 grew to 500. Anything under 200 breaks test stats."""
        assert B.WHALE_COHORT_SIZE >= 200

    def test_nonpayer_ratio_prevents_trivial_f1(self):
        """Some non-payers must be injected, else F1 = 1.0 trivially."""
        assert 0.05 <= B.WHALE_COHORT_NONPAYER_RATIO <= 0.30
