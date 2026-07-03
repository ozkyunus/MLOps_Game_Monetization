"""
/personalized router — LLM Offer Copy (Service 5).

POST /personalized/offer
    Body: { "user_id": "abc", "context": "level_complete" | "after_loss" | "app_open" }
    Returns: personalized offer with title / body / CTA.

Pipeline:
  1. predict_pltv     → pLTV, p_payer, value segment
  2. decide_action    → SHOW_IAP / SHOW_AD_REWARDED / SHOW_AD_INTERSTITIAL / SKIP
  3. LLM (Gemini)     → context-aware copy (only for IAP / rewarded ad actions)
                        falls back to deterministic template if no API key.

Cache: keyed by (value_segment, action, context) so the same archetype of
user gets the same copy on retry — avoids spamming the LLM and keeps the
demo deterministic.
"""
from __future__ import annotations

import logging
import os
from functools import lru_cache

from fastapi import APIRouter
from pydantic import BaseModel, Field
from sqlmodel import Session

from src.database import engine as db_engine
from src.ml.inference import predict_pltv
from src.models import AgentAction, DecisionRequest
from src.routers.decide import compute_expected_revenues
from src.routers.propensity import value_segment

logger = logging.getLogger("personalized")

router = APIRouter(prefix="/personalized", tags=["personalized"])


# ── LLM output schema ────────────────────────────────────────────────────────

class OfferCopy(BaseModel):
    title: str = Field(description="Punchy headline, max 6 words")
    body: str  = Field(description="One sentence describing the offer / value, max 25 words")
    cta:   str = Field(description="Call-to-action button text, max 4 words")


# ── LLM helper (cached chain) ────────────────────────────────────────────────

@lru_cache(maxsize=1)
def _llm_chain():
    """Build the LangChain runnable once. Returns None if no API key."""
    api_key = os.getenv("GOOGLE_API_KEY", "").strip()
    if not api_key or api_key.startswith("your-"):
        return None

    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        google_api_key=api_key,
        temperature=0.7,
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system",
         "You are a mobile-game offer copywriter. Produce ONE offer in JSON with "
         "fields: title (≤6 words), body (≤25 words), cta (≤4 words). "
         "Tone matches the player segment. No emojis unless asked. "
         "Output ONLY valid JSON, no markdown fences."),
        ("user",
         "Player segment: {segment}\n"
         "Action: {action}\n"
         "Context: {context}\n"
         "Offer name: {offer}\n"
         "Offer price: ${price}\n"
         "pLTV (predicted $): {pltv}\n"
         "Write the offer copy."),
    ])
    return prompt | llm.with_structured_output(OfferCopy)


# ── Deterministic fallback templates ─────────────────────────────────────────

FALLBACK_TEMPLATES = {
    ("whale_candidate", "SHOW_IAP", "level_complete"): OfferCopy(
        title="Champion's Premium Pack",
        body="You're crushing it. Unlock the premium tier with exclusive skins, 10x gems, and a legendary bonus chest.",
        cta="Claim Now",
    ),
    ("whale_candidate", "SHOW_IAP", "after_loss"): OfferCopy(
        title="Bounce Back Bundle",
        body="One setback won't stop a champion. Grab the premium revival kit and dominate the next round.",
        cta="Power Up",
    ),
    ("dolphin_candidate", "SHOW_IAP", "level_complete"): OfferCopy(
        title="Starter Hero Pack",
        body="Great run! Boost your progress with extra coins, gems, and a guaranteed rare drop.",
        cta="Unlock Pack",
    ),
    ("minnow_candidate", "SHOW_IAP", "app_open"): OfferCopy(
        title="Remove Ads — One Time",
        body="Enjoy uninterrupted play. Permanent ad removal at a price that won't break the bank.",
        cta="Go Ad-Free",
    ),
    ("low_value", "SHOW_IAP", "app_open"): OfferCopy(
        title="Try a Micro Boost",
        body="A small upgrade to test the waters. Pick up coins and a 24-hour XP boost.",
        cta="Try It",
    ),
    # IAP generic fallbacks (used when segment-specific not defined)
    ("*", "SHOW_IAP", "level_complete"): OfferCopy(
        title="Level-Up Bundle",
        body="Great progress! Grab a value pack to power through the next challenges.",
        cta="Get Pack",
    ),
    ("*", "SHOW_IAP", "after_loss"): OfferCopy(
        title="Comeback Boost",
        body="One more try with a boost? Grab the deal and turn it around.",
        cta="Try Again",
    ),
    ("*", "SHOW_IAP", "app_open"): OfferCopy(
        title="Welcome Back Deal",
        body="A limited-time bundle just for you. Coins, gems, and a surprise bonus inside.",
        cta="Claim Deal",
    ),
    # Ad rewards
    ("*", "SHOW_AD_REWARDED", "after_loss"): OfferCopy(
        title="Get a Second Chance",
        body="Watch a quick video to revive and continue right where you left off.",
        cta="Watch Ad",
    ),
    ("*", "SHOW_AD_REWARDED", "level_complete"): OfferCopy(
        title="Double Your Reward",
        body="Tap to watch a short ad and double the coins from this level.",
        cta="Watch & Double",
    ),
    ("*", "SHOW_AD_REWARDED", "app_open"): OfferCopy(
        title="Daily Reward Boost",
        body="Watch a quick video to triple today's daily login bonus.",
        cta="Watch Now",
    ),
    ("*", "SHOW_AD_INTERSTITIAL", "*"): OfferCopy(
        title="Sponsored Break",
        body="A brief message from our sponsor — back to the game in a few seconds.",
        cta="Skip in 5s",
    ),
    ("*", "SKIP", "*"): OfferCopy(
        title="Keep Playing",
        body="Enjoy uninterrupted play.",
        cta="",
    ),
}


def _fallback_copy(segment: str, action: str, context: str) -> OfferCopy:
    for key in [
        (segment, action, context),
        ("*",     action, context),
        ("*",     action, "*"),
    ]:
        if key in FALLBACK_TEMPLATES:
            return FALLBACK_TEMPLATES[key]
    return OfferCopy(title="Special Offer", body="Don't miss out.", cta="Tap Here")


# ── Decision math is SHARED with /decide (compute_expected_revenues) ─────────
# v2 kept a hand-synced copy of the EV formulas + a copy-pasted value_segment
# here; both are now imported so the two surfaces cannot drift apart.

def _decide_action(pred: dict, context: str) -> dict:
    raw = pred["raw_features"].iloc[0]
    ad_views_d7 = int(raw["_ad_views_d7"])
    d = compute_expected_revenues(pred, context, ad_views_d7)
    return {
        "action": d["best"],
        "expected_revenue": d["er"][d["best"]],
        "offer_tier": d["iap_tier"],
    }


# ── LLM copy generator with cache ────────────────────────────────────────────

_COPY_CACHE: dict[tuple, OfferCopy] = {}


def _generate_copy(segment: str, action: str, context: str, offer_tier: dict,
                   pltv: float) -> tuple[OfferCopy, str]:
    """Returns (copy, source) where source ∈ {'llm', 'fallback', 'cache'}.

    Only REAL LLM results are cached. v2 also cached fallbacks, so a single
    transient Gemini error (e.g. a 429 at cold start) poisoned the cache for
    that (segment, action, context) key permanently — every later request
    returned the fallback labelled "cache" and the LLM was never retried.
    """
    cache_key = (segment, action, context)
    if cache_key in _COPY_CACHE:
        return _COPY_CACHE[cache_key], "cache"

    chain = _llm_chain()
    if chain is None or action in ("SHOW_AD_INTERSTITIAL", "SKIP"):
        # Deterministic template — cheap to recompute, no reason to cache.
        return _fallback_copy(segment, action, context), "fallback"

    try:
        copy = chain.invoke({
            "segment": segment,
            "action":  action,
            "context": context,
            "offer":   offer_tier["name"] if action == "SHOW_IAP" else f"{action.lower()}",
            "price":   offer_tier["price"] if action == "SHOW_IAP" else 0.0,
            "pltv":    round(pltv, 2),
        })
        _COPY_CACHE[cache_key] = copy
        return copy, "llm"
    except Exception:
        logger.exception(
            "LLM copy generation failed for %s — serving fallback (NOT cached)",
            cache_key,
        )
        return _fallback_copy(segment, action, context), "fallback"


# ── Endpoint ─────────────────────────────────────────────────────────────────

@router.post("/offer")
def personalized_offer(payload: DecisionRequest):
    user_id = payload.user_id
    context = payload.context

    pred = predict_pltv(user_id)
    segment = value_segment(pred["pLTV"])
    decision = _decide_action(pred, context)
    copy, source = _generate_copy(
        segment, decision["action"], context, decision["offer_tier"], pred["pLTV"]
    )

    # Audit log — this surface actually RENDERS an offer to the player, so it
    # must land in AgentAction for replay/attribution just like /decide does.
    # (v2 logged only /decide; offers served here were invisible to analysis.)
    with Session(db_engine) as session:
        session.add(AgentAction(
            player_id=user_id,
            action_taken=decision["action"],
            offer_shown=(
                decision["offer_tier"]["name"]
                if decision["action"] == "SHOW_IAP" else None
            ),
            outcome=None,
            revenue_generated=None,
            model_version=pred["model_version"],
            ab_test_group=None,
        ))
        session.commit()

    return {
        "user_id":          user_id,
        "context":          context,
        "value_segment":    segment,
        "action":           decision["action"],
        "offer_name":       decision["offer_tier"]["name"] if decision["action"] == "SHOW_IAP" else None,
        "offer_price_usd":  decision["offer_tier"]["price"] if decision["action"] == "SHOW_IAP" else None,
        "expected_revenue": round(decision["expected_revenue"], 5),
        "copy": {
            "title": copy.title,
            "body":  copy.body,
            "cta":   copy.cta,
        },
        "copy_source":    source,
        "diagnostics": {
            "p_payer": round(pred["p_payer"], 4),
            "pLTV":    round(pred["pLTV"], 2),
        },
        "model_version": pred["model_version"],
    }


@router.get("/cache")
def show_cache():
    """Inspect cached LLM copy per (segment, action, context)."""
    return {
        f"{seg}|{act}|{ctx}": copy.model_dump()
        for (seg, act, ctx), copy in _COPY_CACHE.items()
    }
