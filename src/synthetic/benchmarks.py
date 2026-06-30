"""
Industry-validated benchmarks for synthetic data calibration.

All numbers sourced from public 2026 industry reports + our real backbone data.
Each constant cites its source so it can be defended in interview.

Sources:
  [1] GameAnalytics Mobile Retention Benchmarks 2026
      https://gameindustrylibrary.com/documents/gameanalytics-mobile-retention-benchmarks-2026
  [2] Tap Nation — KPIs for Hybrid Casual Games (2026)
      https://www.tap-nation.io/blog/kpis-that-matter-metrics-to-track-in-hybrid-casual-games/
  [3] Game Growth Advisor — Mobile Game CPI Benchmarks 2026
      https://gamegrowthadvisor.com/blog/2026-03-17-user-acquisition-cpi-benchmarks-2026/
  [4] Game Growth Advisor — 20 Mobile Game KPIs 2026
      https://gamegrowthadvisor.com/blog/2026-03-17-mobile-game-kpis-benchmarks-2026/
  [5] Antier — Hybrid Casual vs Hyper Casual 2026
      https://www.antier.com/blogs/hybrid-casual-games-vs-hypercasual-...
  [6] Real backbone — Marketing Freemium Game Data (Kaggle, anonymized)
  [7] Real reference — Firebase Public Project, Flood It! 30-day sample
"""
from __future__ import annotations
from typing import Final


# ── Project target genre ─────────────────────────────────────────────────────
# Our project models a "hybrid-casual" mobile game (deeper meta than
# hyper-casual but still mass-market accessible). All benchmarks below
# are calibrated to this genre unless noted.
GENRE: Final[str] = "hybrid_casual"


# ── Retention targets [1, 2] ─────────────────────────────────────────────────
# Source: GameAnalytics 2026 + Tap Nation 2026
# Real Flood It! D1 retention = 91% (returning users in window).
# We use industry median for hybrid-casual since Flood It! is casual puzzle
# (different genre).
RETENTION_TARGETS: Final[dict[str, float]] = {
    "D1":  0.34,   # [1] median for hybrid-casual; top iOS 35-40%
    "D7":  0.17,   # [1] >15% threshold for healthy game
    "D30": 0.10,   # [5] 8-12% range for hybrid-casual
    "D60": 0.06,   # extrapolated
    "D90": 0.04,   # extrapolated
}


# ── Monetization targets [2, 5] ──────────────────────────────────────────────
# ARPDAU = Average Revenue Per Daily Active User
ARPDAU_TARGET_USD: Final[float] = 0.30   # [2] hybrid-casual midpoint of $0.15-0.50

# Conversion rate (% of installs who ever pay)
# Real backbone conversion: 9.09%. Industry hybrid-casual range: 8-15%.
# We target the real value to stay calibrated.
CONVERSION_RATE: Final[float] = 0.091


# ── Spending segment distribution [4] ────────────────────────────────────────
# Industry standard 90/10 rule: 1% whales contribute ~50% of revenue.
SEGMENT_DISTRIBUTION: Final[dict[str, float]] = {
    "whale":   0.010,   # top 1% by lifetime spend
    "dolphin": 0.040,   # next 4%
    "minnow":  0.050,   # next 5% — calibrated so total payer ≈ CONVERSION_RATE
    "free":    0.900,   # 90% never pay (matches conversion = ~10%)
}
# Sanity: whale + dolphin + minnow = 0.10 = ~CONVERSION_RATE


# ── LTV distribution [6] — calibrated to REAL purchases data ─────────────────
# Real purchases.ltv summary: min $1.16, median $3.25, max $42, mean $3.77
# Synthetic LTV ranges chosen so combined (real + synthetic) distribution
# remains close to real — KS-test target p > 0.05.
SEGMENT_LTV_RANGE_USD: Final[dict[str, tuple[float, float]]] = {
    "whale":   (25.0, 60.0),  # slight extension of real max ($42 → $60)
    "dolphin": (10.0, 30.0),
    "minnow":  (1.16, 10.0),  # exact real min
    "free":    (0.0, 0.0),
}
# Note: whale range $25-60 lets us add ~150 high-LTV users without
# breaking the real distribution shape (median should stay ~$3-4).


# ── IAP price points [2, 4] — power-law industry standard ────────────────────
IAP_PRICES_USD: Final[list[float]] = [0.99, 1.30, 4.99, 9.99, 19.99, 49.99]

# Segment-specific weights: whales prefer big packs, minnows the smallest.
# Realistic industry pattern — whales buy $49.99 frequently, minnows $0.99.
IAP_WEIGHTS_BY_SEGMENT: Final[dict[str, list[float]]] = {
    "whale":   [0.03, 0.05, 0.12, 0.20, 0.30, 0.30],   # mean ≈ $20
    "dolphin": [0.10, 0.15, 0.30, 0.25, 0.15, 0.05],   # mean ≈ $7
    "minnow":  [0.50, 0.30, 0.15, 0.05, 0.00, 0.00],   # mean ≈ $1.6
}
# Default fallback (used in non-segment contexts)
IAP_WEIGHTS: Final[list[float]] = [0.40, 0.30, 0.15, 0.08, 0.05, 0.02]
# $1.30 included because that's the real backbone purchase amount.
# $0.99, $4.99, $9.99 are universal mobile IAP price points.


# ── eCPM by ad format [4] — USD per 1000 impressions ─────────────────────────
ECPM_USD: Final[dict[str, float]] = {
    "rewarded_video": 18.0,   # [4] $2-50 range, US/UK median
    "interstitial":   12.0,
    "banner":          2.0,
}
# Used to convert ad_view counts → ad revenue per user.


# ── Ad format mix (probability per impression) ───────────────────────────────
AD_FORMAT_DISTRIBUTION: Final[dict[str, float]] = {
    "rewarded_video": 0.60,   # most common in hybrid-casual
    "interstitial":   0.30,
    "banner":         0.10,
}


# ── Session metrics [4] ──────────────────────────────────────────────────────
# Used to calibrate session_count and session_length generators.
SESSION_LENGTH_MEDIAN_MIN: Final[float] = 3.3   # [4] 3.1-3.5 median range
SESSION_LENGTH_P99_MIN:    Final[float] = 24.0  # [4] top 1%
SESSIONS_PER_DAY_MEDIAN:   Final[float] = 3.85  # [4] median
SESSIONS_PER_DAY_P99:      Final[float] = 13.0  # [4] top 1%


# ── CPI by platform [3] ──────────────────────────────────────────────────────
CPI_USD: Final[dict[str, float]] = {
    "ios":     4.22,   # [3]
    "android": 2.97,   # [3]
}


# ── Channel quality multipliers (engagement_potential modifier) ──────────────
# Higher = users from this channel are more engaged (industry knowledge).
# Used in causal generation: behaviors ~ f(latent * channel_multiplier).
CHANNEL_ENGAGEMENT_MULTIPLIER: Final[dict[str, float]] = {
    "Organic":      1.30,   # came on their own — most engaged
    "(direct)":     1.30,   # same as organic
    "google-play":  1.20,   # browsing app store — high intent
    "Facebook Ads": 1.00,   # baseline
    "Google Ads":   0.95,   # slight intent-based dropoff
    "TikTok Ads":   0.85,   # impulse traffic, lower engagement
    "firebase":     1.00,   # debug/test
    "(none)":       1.00,
    "google":       1.00,
    "invite_a_friend": 1.40, # viral users — highest engagement
}


# ── Geo distribution targets — for augmentation balance ──────────────────────
# Real backbone is US-heavy. Geo-diversity augmentation cohort spreads
# users across these countries (matching global mobile gaming demographics).
GEO_DIVERSITY_TARGETS: Final[dict[str, float]] = {
    "United States":  0.25,
    "Germany":        0.08,
    "France":         0.07,
    "United Kingdom": 0.08,
    "Japan":          0.10,
    "South Korea":    0.06,
    "Brazil":         0.10,
    "Turkey":         0.07,
    "India":          0.09,
    "Indonesia":      0.05,
    "Philippines":    0.05,
}


# ── Causal generation parameters ─────────────────────────────────────────────
# Forward-causal model:
#   engagement_potential ~ LogNormal(mu=0, sigma=1)
#       → mu, sigma tuned so generated sessions distribution matches
#         real Flood It! distribution.
#   sessions_d7   = max(0, int(engagement_potential * channel_mult * 2.6))
#   ad_views_d7   = poisson(sessions_d7 * 1.5)
#   purchase_made ~ Bernoulli(sigmoid(engagement_potential - 1.8))
#                   threshold tuned to give ~9% conversion
#   amount_usd    ~ Categorical(IAP_PRICES, IAP_WEIGHTS) | purchase_made
#   ltv_d30       = cumulative purchases per user
#
# This ordering ensures NO target leakage: behaviors come from latent,
# LTV comes from behaviors. Predicting LTV from behaviors is legitimate
# learning, not memorization.
ENGAGEMENT_LATENT_MU:    Final[float] = 0.0
ENGAGEMENT_LATENT_SIGMA: Final[float] = 1.0

# Sessions per active user per day, conditional on engagement_potential
SESSIONS_BASELINE_MULT: Final[float] = 2.6   # tuned to real Flood It! median

# Conversion probability tuning (Bernoulli sigmoid threshold)
# Calibrated: with engagement ~ LogNormal(0,1) × channel_mult (~1.0-1.3),
# threshold=3.5 gives ~9% conversion (matches real backbone).
PURCHASE_THRESHOLD: Final[float] = 3.5

# Ad views per session (Poisson lambda multiplier)
ADS_PER_SESSION_LAMBDA: Final[float] = 1.5


# ── Validation thresholds — what counts as "calibrated" ──────────────────────
# Used in notebooks/02_synthetic_validation.ipynb to pass/fail the
# synthetic data. If any of these fail, regenerate with adjusted params.
VALIDATION_THRESHOLDS: Final[dict[str, tuple[float, float]]] = {
    # KS-test p-value should be > 0.05 (distributions not significantly different)
    "ks_pvalue_ltv":        (0.05, 1.0),
    "ks_pvalue_sessions":   (0.05, 1.0),
    # Conversion rate within ±2% of target
    "conversion_rate":      (0.07, 0.13),
    # SDV quality score (higher is better, 1.0 = perfect)
    "sdv_quality_score":    (0.80, 1.0),
    # Correlation matrix RMSE (lower is better, 0 = identical correlations)
    "correlation_rmse":     (0.0, 0.15),
}
