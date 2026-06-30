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
import uuid
import random
from datetime import timedelta

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from sdv.metadata import SingleTableMetadata
from sdv.single_table import GaussianCopulaSynthesizer

from src.synthetic import benchmarks as B


# ── Setup ────────────────────────────────────────────────────────────────────

load_dotenv()
engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL"))

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

        for i, idx in enumerate(np.where(payer_mask)[0]):
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
    Uses segment-specific IAP weights so whales buy bigger packs.
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
        n_txn = random.randint(*n_txn_map[seg])

        cumulative = 0.0
        install_dt = pd.to_datetime(user["install_date"])

        for _ in range(n_txn):
            amount = float(np.random.choice(B.IAP_PRICES_USD, p=weights))
            # Cap last txn so we don't blow past ltv_max
            if cumulative + amount > ltv_max:
                if cumulative >= ltv_min:
                    break
                amount = ltv_max - cumulative
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

def generate_whale_cohort(synth: GaussianCopulaSynthesizer, n: int = 150) -> pd.DataFrame:
    """High-engagement users sampled with iOS bias + premium geos."""
    df = synth.sample(num_rows=n)
    df["user_id"] = [str(uuid.uuid4()).upper().replace("-", "") for _ in range(n)]
    df["country"] = np.random.choice(
        ["United States", "Japan", "South Korea", "United Kingdom", "Germany"],
        size=n, p=[0.45, 0.20, 0.15, 0.10, 0.10],
    )
    df["platform"] = np.random.choice(["ios", "android"], size=n, p=[0.65, 0.35])
    df["_cohort"] = "augmented_whale"

    # Whales are forced high-engagement (multiplier boosted)
    channel_mult = np.full(n, 1.3)  # high baseline (organic-like)
    df["_engagement_potential"] = sample_engagement_potential(n, channel_mult) * 2.0  # 2x boost
    df["_sessions_d7"] = derive_sessions_d7(df["_engagement_potential"].values)
    df["_ad_views_d7"] = derive_ad_views_d7(df["_sessions_d7"].values)
    # Whales virtually always purchase
    df["_will_purchase"] = 1
    # All payers — assign within whale/dolphin/minnow by engagement
    df["_segment"] = assign_segment(df["_engagement_potential"].values, df["_will_purchase"].values)
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
    """A 'later launch' cohort with slightly better engagement (industry-typical)."""
    df = synth.sample(num_rows=n)
    df["user_id"] = [str(uuid.uuid4()).upper().replace("-", "") for _ in range(n)]
    # Override install_date to be later (different cohort)
    df["install_date"] = pd.to_datetime(
        np.random.choice(pd.date_range("2024-09-15", "2024-09-30"), size=n)
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
    whales = generate_whale_cohort(synth, n=150)
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
