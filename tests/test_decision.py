"""Unit tests for the shared decision math (no DB, no models needed).

`compute_expected_revenues` is the single source of truth used by BOTH
/decide/action and /personalized/offer — these tests pin its behaviour.
"""
from __future__ import annotations

from src.routers.decide import (
    IAP_MIN_P_PAYER,
    SKIP_EV_FLOOR,
    ad_fatigue_penalty_for,
    compute_expected_revenues,
)


def _pred(p_payer: float, pltv: float, gated: bool = False) -> dict:
    return {"p_payer": p_payer, "pLTV": pltv, "gate_applied": gated}


class TestFatigueTiers:
    def test_graduated(self):
        assert ad_fatigue_penalty_for(10) == 1.0
        assert ad_fatigue_penalty_for(31) == 0.50
        assert ad_fatigue_penalty_for(61) == 0.15

    def test_boundaries_are_strict(self):
        assert ad_fatigue_penalty_for(30) == 1.0
        assert ad_fatigue_penalty_for(60) == 0.50


class TestSkipReachability:
    """v2's floor (0.0005) made SKIP dead code — worst-case ad EV was 0.0048.
    The retention-protection branch must actually be reachable now."""

    def test_heavy_fatigue_non_iap_user_gets_skip(self):
        d = compute_expected_revenues(_pred(0.05, 0.0, gated=True), "app_open", 100)
        assert d["best"] == "SKIP"

    def test_after_loss_rewarded_survives_heavy_fatigue(self):
        """Right after a loss the rewarded offer (extra life) stays worth it."""
        d = compute_expected_revenues(_pred(0.05, 0.0, gated=True), "after_loss", 100)
        assert d["best"] == "SHOW_AD_REWARDED"

    def test_moderate_fatigue_still_serves_ads(self):
        d = compute_expected_revenues(_pred(0.05, 0.0, gated=True), "app_open", 40)
        assert d["best"] == "SHOW_AD_REWARDED"

    def test_iap_user_never_skipped_by_fatigue(self):
        """Fatigue penalises ads, not IAP — an eligible payer still sees IAP."""
        d = compute_expected_revenues(_pred(0.60, 25.0), "app_open", 100)
        assert d["best"] == "SHOW_IAP"


class TestIAPEligibility:
    def test_gate_blocks_iap(self):
        d = compute_expected_revenues(_pred(0.60, 25.0, gated=True), "app_open", 0)
        assert d["iap_eligible"] is False
        assert d["er"]["SHOW_IAP"] == 0.0

    def test_low_p_payer_blocks_iap(self):
        d = compute_expected_revenues(_pred(IAP_MIN_P_PAYER - 0.01, 25.0), "app_open", 0)
        assert d["iap_eligible"] is False

    def test_eligible_payer_gets_iap(self):
        d = compute_expected_revenues(_pred(0.60, 25.0), "level_complete", 0)
        assert d["iap_eligible"] is True
        assert d["best"] == "SHOW_IAP"
        # Context multiplier must be included in the EV (v2's reasoning string
        # omitted it, disagreeing with the expected_revenue field).
        assert abs(d["er"]["SHOW_IAP"] - 0.60 * 9.99 * 1.20) < 1e-9


class TestFloorConstant:
    def test_floor_sits_between_heavy_fatigue_evs(self):
        """The floor must separate 'heavy fatigue, neutral context' (skip)
        from 'heavy fatigue, after_loss' (still show rewarded)."""
        neutral = compute_expected_revenues(_pred(0.05, 0.0, True), "app_open", 100)
        loss    = compute_expected_revenues(_pred(0.05, 0.0, True), "after_loss", 100)
        assert max(v for k, v in neutral["er"].items() if k != "SKIP") < SKIP_EV_FLOOR
        assert loss["er"]["SHOW_AD_REWARDED"] > SKIP_EV_FLOOR
