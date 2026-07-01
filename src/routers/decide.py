"""
/decide router — Decision Engine (Service 2).

POST /decide/action
    Body: { "user_id": "abc", "context": "level_complete" | "app_open" | "after_loss" }
    Returns: best action ∈ {SHOW_IAP, SHOW_AD_REWARDED, SHOW_AD_INTERSTITIAL, SKIP}
             by argmax(expected_revenue) with context modifiers + ad fatigue.

Logic (per action):
  - SHOW_IAP                 → p_payer × offer_price[tier]
  - SHOW_AD_REWARDED         → eCPM_rewarded × completion_rate
  - SHOW_AD_INTERSTITIAL     → eCPM_interstitial × 1.0 (passive)
  - SKIP                     → 0 (preserves retention; floor when all else hurts)

Context multipliers (industry intuition):
  - "level_complete" : IAP +20%       (positive moment, buyer mood)
  - "after_loss"     : REWARDED +30%  (motivated to continue, e.g. extra life)
  - "app_open"       : balanced

Ad fatigue penalty: if ad_views_d7 > 30, ads get a 0.5x penalty.

Output is logged to AgentAction so we can later replay + A/B compare.
"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, HTTPException
from sqlmodel import Session

from src.database import engine as db_engine
from src.ml.inference import predict_pltv
from src.models import AgentAction
from src.synthetic import benchmarks as B

router = APIRouter(prefix="/decide", tags=["decide"])


# Per-impression ad revenue ($) — eCPM is per 1000 impressions
AD_REVENUE_PER_IMPRESSION = {
    "SHOW_AD_REWARDED":     B.ECPM_USD["rewarded_video"] / 1000.0,
    "SHOW_AD_INTERSTITIAL": B.ECPM_USD["interstitial"]   / 1000.0,
}

REWARDED_COMPLETION_RATE = 0.85   # industry typical

# IAP offer tier matched to pLTV bucket
IAP_OFFER_TIERS = [
    {"min_pltv":  5.0, "price": 9.99, "name": "starter_pack_premium"},
    {"min_pltv":  2.0, "price": 4.99, "name": "starter_pack_basic"},
    {"min_pltv":  0.5, "price": 1.99, "name": "remove_ads_lite"},
    {"min_pltv":  0.0, "price": 0.99, "name": "micro_offer"},
]

CONTEXT_MULTIPLIERS = {
    "level_complete": {"SHOW_IAP": 1.20, "SHOW_AD_REWARDED": 1.00, "SHOW_AD_INTERSTITIAL": 1.00},
    "after_loss":     {"SHOW_IAP": 0.90, "SHOW_AD_REWARDED": 1.30, "SHOW_AD_INTERSTITIAL": 0.80},
    "app_open":       {"SHOW_IAP": 1.00, "SHOW_AD_REWARDED": 1.00, "SHOW_AD_INTERSTITIAL": 1.00},
}

# Minimum p_payer required to attempt an IAP. Below this we don't trust the model
# enough to attempt monetization via IAP — we fall back to ads (or SKIP).
# The gate on pLTV protects the LTV prediction; this threshold protects the IAP
# decision from over-firing on marginal cases.
IAP_MIN_P_PAYER = 0.25

ContextStr = Literal["level_complete", "after_loss", "app_open"]


def match_iap_tier(pltv: float) -> dict:
    """Pick the highest-price tier the user can sustain by pLTV."""
    for tier in IAP_OFFER_TIERS:
        if pltv >= tier["min_pltv"]:
            return tier
    return IAP_OFFER_TIERS[-1]


@router.post("/action")
def decide_action(payload: dict):
    user_id = payload.get("user_id")
    context = payload.get("context", "app_open")
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id required")
    if context not in CONTEXT_MULTIPLIERS:
        raise HTTPException(
            status_code=422,
            detail=f"context must be one of {list(CONTEXT_MULTIPLIERS)}",
        )

    pred = predict_pltv(user_id)
    raw = pred["raw_features"].iloc[0]
    ad_views_d7 = int(raw["_ad_views_d7"])
    ad_fatigue_penalty = 0.5 if ad_views_d7 > 30 else 1.0

    mult = CONTEXT_MULTIPLIERS[context]
    iap_tier = match_iap_tier(pred["pLTV"])

    # ── IAP eligibility check ────────────────────────────────────────────────
    # Skip IAP if either: (a) the pLTV gate was applied ("we don't trust this
    # user is a payer") OR (b) p_payer is below the IAP-specific threshold.
    # This aligns the IAP decision with the gate mechanism upstream.
    iap_eligible = (
        not pred.get("gate_applied", False)
        and pred["p_payer"] >= IAP_MIN_P_PAYER
    )

    # Expected revenue per action
    er = {
        "SHOW_IAP": (
            pred["p_payer"] * iap_tier["price"] * mult["SHOW_IAP"]
            if iap_eligible else 0.0
        ),
        "SHOW_AD_REWARDED": (
            AD_REVENUE_PER_IMPRESSION["SHOW_AD_REWARDED"]
            * REWARDED_COMPLETION_RATE
            * mult["SHOW_AD_REWARDED"]
            * ad_fatigue_penalty
        ),
        "SHOW_AD_INTERSTITIAL": (
            AD_REVENUE_PER_IMPRESSION["SHOW_AD_INTERSTITIAL"]
            * mult["SHOW_AD_INTERSTITIAL"]
            * ad_fatigue_penalty
        ),
        "SKIP": 0.0,
    }

    # SKIP wins only when all paid actions are below a floor — protect retention
    # rather than annoy a fatigued user with no upside.
    if max(v for k, v in er.items() if k != "SKIP") < 0.0005:
        best_action = "SKIP"
    else:
        best_action = max(er, key=er.get)

    # Build reasoning string
    if best_action == "SHOW_IAP":
        pltv_qualifier = (
            "High" if pred["pLTV"] >= 5 else
            "Medium" if pred["pLTV"] >= 2 else
            "Low" if pred["pLTV"] >= 0.5 else
            "Marginal"
        )
        reasoning = (
            f"{pltv_qualifier} pLTV (${pred['pLTV']:.2f}) with p_payer={pred['p_payer']:.2f} "
            f"≥ IAP threshold {IAP_MIN_P_PAYER}; picks {iap_tier['name']} at "
            f"${iap_tier['price']} (expected {pred['p_payer']*iap_tier['price']:.3f})."
        )
        offer = iap_tier["name"]
    elif best_action == "SHOW_AD_REWARDED":
        why_not_iap = (
            "IAP gated (p_payer too low)" if not iap_eligible
            else "IAP EV lower than rewarded ad EV"
        )
        reasoning = (
            f"Rewarded ad wins: p_payer={pred['p_payer']:.2f}, pLTV=${pred['pLTV']:.2f}. "
            f"{why_not_iap}"
            + (" — ad fatigue active" if ad_fatigue_penalty < 1 else "")
        )
        offer = None
    elif best_action == "SHOW_AD_INTERSTITIAL":
        reasoning = (
            f"Interstitial wins on this context (passive impression). "
            f"p_payer={pred['p_payer']:.2f} below IAP threshold {IAP_MIN_P_PAYER}."
        )
        offer = None
    else:
        reasoning = "All actions below value floor — skip to protect retention"
        offer = None

    alternatives = [
        {"action": a, "expected_revenue": round(v, 5)}
        for a, v in sorted(er.items(), key=lambda kv: kv[1], reverse=True)
        if a != best_action
    ]

    # Audit log
    with Session(db_engine) as session:
        session.add(AgentAction(
            player_id=user_id,
            action_taken=best_action,
            offer_shown=offer,
            outcome=None,
            revenue_generated=None,
            model_version=pred["model_version"],
            ab_test_group=None,
        ))
        session.commit()

    return {
        "user_id":          user_id,
        "context":          context,
        "action":           best_action,
        "offer":            offer,
        "expected_revenue": round(er[best_action], 5),
        "reasoning":        reasoning,
        "alternatives":     alternatives,
        "diagnostics": {
            "p_payer":             round(pred["p_payer"], 4),
            "pLTV":                round(pred["pLTV"], 2),
            "ad_views_d7":         ad_views_d7,
            "ad_fatigue_penalty":  ad_fatigue_penalty,
            "iap_eligible":        iap_eligible,
            "iap_min_p_payer":     IAP_MIN_P_PAYER,
            "gate_applied":        pred.get("gate_applied", False),
        },
        "model_version": pred["model_version"],
    }
