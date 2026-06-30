"""
Synthetic event generator — calibrated to real backbone.

Generates event-level behavioral data for each user in raw_user_source:
  - session_events       : daily user sessions (D0-D7)
  - level_events          : in-game progression
  - ad_impression_events  : ad shown/watched/skipped
  - iap_offer_events      : IAP popup shown/tapped/dismissed
  - page_view_events      : screen navigation

Calibrated against real data:
  - Conversion rate (purchases/installs) → ~9%
  - LTV distribution (purchases.ltv) → lognormal-ish
  - Channel quality differences (FB vs Google vs Organic)
  - Geo and platform conditional patterns

Anomaly injection (for drift demo):
  - Week 4+: CPI drift, retention drop, engagement decrease

Run: uv run python scripts/generate_synthetic_data.py
"""
import numpy as np
import pandas as pd


# -------------------- Configuration --------------------

CHANNEL_ENGAGEMENT_MULTIPLIER = {
    "Organic": 1.3,        # Organic users en aktif (kendileri buldular)
    "Facebook Ads": 1.0,   # Baseline
    "Google Ads": 0.95,    # Hafif daha düşük (intent-based)
    "TikTok Ads": 0.85,    # En düşük engagement (sweep traffic)
}

SEGMENT_DISTRIBUTION = {
    "whale": 0.01,    # %1 — toplam revenue'nun %50'si
    "dolphin": 0.04,  # %4 — toplam revenue'nun %30'u
    "minnow": 0.10,   # %10 — toplam revenue'nun %20'si
    "free": 0.85,     # %85 — hiç ödemez
}


# -------------------- Generators --------------------

def generate_session_events(users_df: pd.DataFrame, n_days: int = 7) -> pd.DataFrame:
    """TODO Gün 2: power-law sessions per channel, day-of-week effect."""
    raise NotImplementedError("Implement in Day 2")


def generate_level_events(users_df: pd.DataFrame, sessions_df: pd.DataFrame) -> pd.DataFrame:
    """TODO Gün 2: level progression with realistic difficulty curve."""
    raise NotImplementedError("Implement in Day 2")


def generate_ad_events(users_df, sessions_df, segments) -> pd.DataFrame:
    """TODO Gün 2: ad impressions per session, format mix (rewarded/interstitial/banner)."""
    raise NotImplementedError("Implement in Day 2")


def generate_iap_events(users_df, sessions_df, segments) -> pd.DataFrame:
    """TODO Gün 2: IAP popups + tap rate calibrated to real conversion (~9%)."""
    raise NotImplementedError("Implement in Day 2")


def generate_page_events(users_df, sessions_df) -> pd.DataFrame:
    """TODO Gün 2: page view events (home, level_complete, shop, daily_reward, etc.)."""
    raise NotImplementedError("Implement in Day 2")


# -------------------- Main --------------------

if __name__ == "__main__":
    print("Synthetic generator skeleton — implement in Day 2.")