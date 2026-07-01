"""
User & purchase augmentation — forward-causal generation calibrated to real backbone.

Generation order (forward-causal, NO target leakage):
  1. Generate user profile (country, channel, platform) — SDV + constraints
  2. Sample engagement_potential ~ LogNormal (latent variable)
  3. Behaviors = f(engagement_potential * channel_multiplier)
  4. Purchase decision ~ Bernoulli(sigmoid(engagement_potential - threshold))
  5. LTV = sum of purchases (Categorical IAP prices)

Key property: behaviors → LTV (never LTV → behaviors).
Models learning LTV from behaviors is legitimate, not memorization.

Three cohorts produced:
  - whale_cohort   : ~150 high-engagement users (whale + dolphin segments)
  - geo_diversity  : ~800 users across underrepresented geos
  - healthy_cohort : ~1500 users from a "later launch" date window

Outputs written to Postgres:
  - synth_users      : user-level profile + behavior aggregates
  - synth_purchases  : transaction-level data linked to synth_users

Calibration validated in: notebooks/02_synthetic_validation.ipynb
"""
from __future__ import annotations

import os
import random
import uuid
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer
from sqlalchemy import create_engine

from src.synthetic import benchmarks as B

# ── Setup ────────────────────────────────────────────────────────────────────

load_dotenv()


def get_engine():
    """Lazy DB engine — created on first call, not at import time.
    Keeps `import src.synthetic.user_augmentation` cheap and CI-safe when
    `SQLALCHEMY_DATABASE_URL` is unset."""
    return create_engine(os.getenv("SQLALCHEMY_DATABASE_URL"))


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── Forward-causal helpers ───────────────────────────────────────────────────

def sample_engagement_potential(n: int, channel_mult: np.ndarray) -> np.ndarray:
    """Latent engagement variable. Higher = more active/more likely to pay.
    LogNormal so right-skewed (whale tail).
    Multiplied by channel quality factor (organic users more engaged).
    """
    base = np.random.lognormal(
        mean=B.ENGAGEMENT_LATENT_MU,
        sigma=B.ENGAGEMENT_LATENT_SIGMA,
        size=n,
    )
    return base * channel_mult


def derive_sessions_d7(engagement: np.ndarray) -> np.ndarray:
    """sessions_d7 = engagement * baseline, with realistic noise."""
    raw = engagement * B.SESSIONS_BASELINE_MULT
    noise = np.random.poisson(lam=1.0, size=len(engagement))
    return np.maximum(0, (raw + noise).astype(int))


def derive_ad_views_d7(sessions_d7: np.ndarray) -> np.ndarray:
    """ad_views_d7 ~ Poisson(sessions * lambda).
    Free users see MORE ads than payers (industry pattern).
    """
    return np.random.poisson(
        lam=sessions_d7 * B.ADS_PER_SESSION_LAMBDA,
    ).astype(int)


def decide_purchase(engagement: np.ndarray) -> np.ndarray:
    """Bernoulli sigmoid — calibrated to give ~9% conversion."""
    prob = 1.0 / (1.0 + np.exp(-(engagement - B.PURCHASE_THRESHOLD)))
    return (np.random.random(size=len(engagement)) < prob).astype(int)


def assign_segment(engagement: np.ndarray, will_purchase: np.ndarray) -> np.ndarray:
    """Segment derived from engagement (causal, not arbitrary)."""
    segments = np.full(len(engagement), "free", dtype=object)
    payer_mask = will_purchase == 1

    # Among payers: top 10% → whale, next 40% → dolphin, rest → minnow
    if payer_mask.sum() > 0:
        payer_engagement = engagement[payer_mask]
        whale_thr = np.quantile(payer_engagement, 0.90)
        dolphin_thr = np.quantile(payer_engagement, 0.50)

        for _i, idx in enumerate(np.where(payer_mask)[0]):
            e = engagement[idx]
            if e >= whale_thr:
                segments[idx] = "whale"
            elif e >= dolphin_thr:
                segments[idx] = "dolphin"
            else:
                segments[idx] = "minnow"
    return segments


def generate_purchases(
    users_df: pd.DataFrame,
    txn_id_start: int = 200000,
) -> pd.DataFrame:
    """For each paying user, generate purchase events with cumulative LTV.
    Total LTV bounded by segment range (calibrated to real distribution).
    Uses segment-specific IAP weights AND channel LTV multiplier so:
      - whales buy bigger packs (segment-specific IAP weights)
      - TikTok users spend less per session (channel LTV multiplier ~0.6)
      - Organic users spend baseline (multiplier 1.0)
    """
    rows = []
    txn_counter = txn_id_start

    # Transaction count per segment: whales buy more often
    n_txn_map = {"whale": (8, 25), "dolphin": (3, 10), "minnow": (1, 3)}

    for _, user in users_df.iterrows():
        seg = user["_segment"]
        if seg == "free":
            continue

        ltv_min, ltv_max = B.SEGMENT_LTV_RANGE_USD[seg]
        weights = B.IAP_WEIGHTS_BY_SEGMENT[seg]
        # Apply channel LTV multiplier — narrows whale range for low-quality
        # channels (TikTok), keeps Organic at full range.
        chan_ltv_mult = B.CHANNEL_LTV_MULTIPLIER.get(user.get("type", "Organic"), 1.0)
        ltv_max_chan = ltv_max * chan_ltv_mult
        ltv_min_chan = ltv_min * chan_ltv_mult
        n_txn = random.randint(*n_txn_map[seg])

        cumulative = 0.0
        install_dt = pd.to_datetime(user["install_date"])

        for _ in range(n_txn):
            amount = float(np.random.choice(B.IAP_PRICES_USD, p=weights))
            # Cap last txn so we don't blow past channel-adjusted ltv_max
            if cumulative + amount > ltv_max_chan:
                if cumulative >= ltv_min_chan:
                    break
                amount = max(0.99, ltv_max_chan - cumulative)
            cumulative += amount
            txn_counter += 1
            days_after = random.randint(0, 30)
            rows.append({
                "transaction_id": f"PUR-S{txn_counter}",
                "user_id":     user["user_id"],
                "date":        install_dt + timedelta(days=days_after),
                "value_in_USD": round(amount, 2),
                "ltv":         round(cumulative, 2),
                "_segment":    seg,
                "_cohort":     user["_cohort"],
            })

    return pd.DataFrame(rows)


# ── Cohort generators ────────────────────────────────────────────────────────

def generate_whale_cohort(synth: GaussianCopulaSynthesizer, n: int | None = None) -> pd.DataFrame:
    """High-engagement users sampled with iOS bias + premium geos.

    Design notes:
      - n defaults to B.WHALE_COHORT_SIZE (500) — large enough for ≥75 whale
        samples in stratified test (was 8 in v1, statistically unusable).
      - ~15% non-payers injected (B.WHALE_COHORT_NONPAYER_RATIO) — prevents
        the "100% payer trivial F1" problem.
      - augmented_whale is TRAIN-ONLY (filtered out of val/test in train
        scripts). Evaluating on a designed-positive cohort would mask
        real-world performance.
    """
    n = n if n is not None else B.WHALE_COHORT_SIZE
    df = synth.sample(num_rows=n)
    df["user_id"] = [str(uuid.uuid4()).upper().replace("-", "") for _ in range(n)]
    df["country"] = np.random.choice(
        ["United States", "Japan", "South Korea", "United Kingdom", "Germany"],
        size=n, p=[0.45, 0.20, 0.15, 0.10, 0.10],
    )
    df["platform"] = np.random.choice(["ios", "android"], size=n, p=[0.65, 0.35])
    df["_cohort"] = "augmented_whale"

    # Use the channel from synth output (organic-heavy via the SDV sampling),
    # then apply our (calibrated stronger) channel multiplier.
    chan_mult = df["type"].map(B.CHANNEL_ENGAGEMENT_MULTIPLIER).fillna(1.0).values
    # Whales get a 3x boost on top of channel quality (was 2x, but with
    # the new segment ladder we need top-quantile engagement to land in
    # whale tier consistently).
    df["_engagement_potential"] = sample_engagement_potential(n, chan_mult) * 3.0
    df["_sessions_d7"] = derive_sessions_d7(df["_engagement_potential"].values)
    df["_ad_views_d7"] = derive_ad_views_d7(df["_sessions_d7"].values)

    # Forced segment composition (instead of engagement-quantile assignment):
    # 60% whale, 20% dolphin, 5% minnow, 15% non-payer.
    # This is what "whale cohort" should mean — designed to add whale rows
    # to the dataset, not just "high-engagement users".
    n_whale   = int(n * 0.60)
    n_dolphin = int(n * 0.20)
    n_minnow  = int(n * 0.05)
    n - n_whale - n_dolphin - n_minnow

    idx = np.random.permutation(n)
    seg_array = np.empty(n, dtype=object)
    will_purchase = np.zeros(n, dtype=int)

    seg_array[idx[:n_whale]] = "whale"
    will_purchase[idx[:n_whale]] = 1

    seg_array[idx[n_whale:n_whale + n_dolphin]] = "dolphin"
    will_purchase[idx[n_whale:n_whale + n_dolphin]] = 1

    seg_array[idx[n_whale + n_dolphin:n_whale + n_dolphin + n_minnow]] = "minnow"
    will_purchase[idx[n_whale + n_dolphin:n_whale + n_dolphin + n_minnow]] = 1

    seg_array[idx[n_whale + n_dolphin + n_minnow:]] = "free"

    df["_segment"] = seg_array
    df["_will_purchase"] = will_purchase
    return df


def generate_geo_diversity_cohort(
    synth: GaussianCopulaSynthesizer, n: int = 800
) -> pd.DataFrame:
    """Underrepresented geos with natural conversion rate."""
    df = synth.sample(num_rows=n)
    df["user_id"] = [str(uuid.uuid4()).upper().replace("-", "") for _ in range(n)]
    # Sample geo from diversity distribution
    countries = list(B.GEO_DIVERSITY_TARGETS.keys())
    probs = list(B.GEO_DIVERSITY_TARGETS.values())
    df["country"] = np.random.choice(countries, size=n, p=probs)
    df["_cohort"] = "augmented_geo"

    # Standard engagement distribution
    channel_mult = df["type"].map(B.CHANNEL_ENGAGEMENT_MULTIPLIER).fillna(1.0).values
    df["_engagement_potential"] = sample_engagement_potential(n, channel_mult)
    df["_sessions_d7"] = derive_sessions_d7(df["_engagement_potential"].values)
    df["_ad_views_d7"] = derive_ad_views_d7(df["_sessions_d7"].values)
    df["_will_purchase"] = decide_purchase(df["_engagement_potential"].values)
    df["_segment"] = assign_segment(df["_engagement_potential"].values, df["_will_purchase"].values)
    return df


def generate_healthy_cohort(
    synth: GaussianCopulaSynthesizer, n: int = 1500
) -> pd.DataFrame:
    """A 'later launch' cohort with slightly better engagement (industry-typical).

    install_date kept in Aug 2024 range (matching real backbone) — earlier
    versions used Sep dates which broke any temporal-aware downstream task.
    """
    df = synth.sample(num_rows=n)
    df["user_id"] = [str(uuid.uuid4()).upper().replace("-", "") for _ in range(n)]
    df["install_date"] = pd.to_datetime(
        np.random.choice(pd.date_range("2024-08-01", "2024-08-30"), size=n)
    )
    df["_cohort"] = "augmented_healthy"

    # Slightly higher channel multiplier (better-targeted UA campaign simulation)
    channel_mult = (
        df["type"].map(B.CHANNEL_ENGAGEMENT_MULTIPLIER).fillna(1.0).values * 1.15
    )
    df["_engagement_potential"] = sample_engagement_potential(n, channel_mult)
    df["_sessions_d7"] = derive_sessions_d7(df["_engagement_potential"].values)
    df["_ad_views_d7"] = derive_ad_views_d7(df["_sessions_d7"].values)
    df["_will_purchase"] = decide_purchase(df["_engagement_potential"].values)
    df["_segment"] = assign_segment(df["_engagement_potential"].values, df["_will_purchase"].values)
    return df


# ── Main entry ───────────────────────────────────────────────────────────────

def run() -> None:
    engine = get_engine()
    print("Loading raw_user_source from Postgres...")
    real_users = pd.read_sql("SELECT * FROM raw_user_source", engine)
    print(f"  Real users: {len(real_users):,}")

    # Train SDV
    print("\nTraining SDV GaussianCopulaSynthesizer on real backbone...")
    metadata = SingleTableMetadata()
    metadata.detect_from_dataframe(real_users)
    metadata.update_column(column_name="user_id", sdtype="id")
    synth = GaussianCopulaSynthesizer(metadata, enforce_min_max_values=True)
    synth.fit(real_users)
    print("  ✓ SDV trained")

    # Generate cohorts
    print("\nGenerating cohorts (forward-causal):")
    whales = generate_whale_cohort(synth)  # uses B.WHALE_COHORT_SIZE
    print(f"  ✓ whale_cohort:   {len(whales):>5,} users  | segments: {dict(whales['_segment'].value_counts())}")

    geos = generate_geo_diversity_cohort(synth, n=800)
    print(f"  ✓ geo_diversity:  {len(geos):>5,} users  | segments: {dict(geos['_segment'].value_counts())}")

    healthy = generate_healthy_cohort(synth, n=1500)
    print(f"  ✓ healthy_cohort: {len(healthy):>5,} users  | segments: {dict(healthy['_segment'].value_counts())}")

    synth_users = pd.concat([whales, geos, healthy], ignore_index=True)
    print(f"\nTotal synthetic users: {len(synth_users):,}")

    # Generate matching purchases
    print("\nGenerating purchases for paying users (forward-causal)...")
    synth_purchases = generate_purchases(synth_users)
    print(f"  ✓ Synthetic purchases: {len(synth_purchases):,}")
    print(f"  Unique paying users:   {synth_purchases['user_id'].nunique():,}")
    sim_conv = synth_purchases["user_id"].nunique() / len(synth_users)
    print(f"  Simulated conversion:  {sim_conv * 100:.2f}% (target: {B.CONVERSION_RATE * 100:.2f}%)")

    # LTV summary (calibration check)
    print("\nLTV by segment (synthetic):")
    print(synth_purchases.groupby("_segment")["ltv"].describe().round(2)[
        ["count", "mean", "min", "50%", "max"]
    ])

    # Write to Postgres
    print("\nWriting to Postgres...")
    synth_users.to_sql("synth_users", engine, if_exists="replace", index=False)
    synth_purchases.to_sql("synth_purchases", engine, if_exists="replace", index=False)
    print(f"  ✓ synth_users:     {len(synth_users):,} rows")
    print(f"  ✓ synth_purchases: {len(synth_purchases):,} rows")

    print("\n" + "=" * 60)
    print("Augmentation complete. Run validation:")
    print("  notebooks/02_synthetic_validation.ipynb")
    print("=" * 60)


if __name__ == "__main__":
    run()
