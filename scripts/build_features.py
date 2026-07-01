"""
Build the ML-ready feature table: user_features_d7.

Combines real backbone + synthetic augmentation into one row-per-user table.

Honesty fixes vs v1 (see README → Limitations section for full discussion):
  - Real users: engagement_potential is now sampled INDEPENDENTLY from the
    industry distribution, NOT derived from LTV. This eliminates the
    target leakage that was inflating model AUC.
    Consequence: real users become harder to predict (which is the honest
    state of affairs given we have no behavioral event data for them).
  - target_ltv_d30 → target_ltv (the underlying real data is a D7 snapshot,
    not a true D30 measurement; the v1 name was misleading).
  - Channel-LTV multiplier flows through to real users too (so the model
    sees channel-driven LTV variance).

Steps:
  1. Load real users + real purchases from Postgres.
  2. Generate independent random behavior for real users (forward-causal
     from population engagement distribution).
  3. Load synth users (already have engagement/sessions/etc).
  4. Concatenate, engineer derived features, attach LTV target.
  5. Write to Postgres as `user_features_d7`.

Output: ~17,955 rows × ~20 columns (with whale 500 instead of 150).
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.synthetic import benchmarks as B
from src.synthetic.user_augmentation import (
    derive_ad_views_d7,
    derive_sessions_d7,
    sample_engagement_potential,
)

load_dotenv()
engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL"))

SEED = 42
np.random.seed(SEED)


def derive_behavior_for_real_users(real_users: pd.DataFrame, real_purchases: pd.DataFrame) -> pd.DataFrame:
    """For real users, derive D7 behavior features INDEPENDENTLY of LTV.

    This is the honest replacement for v1's leaky `engagement = log1p(LTV) + noise`.
    Now engagement is sampled from the same population distribution used for
    synth users, multiplied only by the channel quality factor.

    Consequence: model cannot predict real-user LTV from behavior (because
    behavior is uninformative). It can still use demographics (channel,
    country, platform). This is what real production data looks like for
    a brand-new user with no behavioral history.
    """
    user_ltv = (
        real_purchases.groupby("user_id")["ltv"]
        .max()
        .rename("_real_ltv")
        .reset_index()
    )
    df = real_users.merge(user_ltv, on="user_id", how="left")
    df["_real_ltv"] = df["_real_ltv"].fillna(0.0)

    # INDEPENDENT engagement — sampled from population distribution +
    # channel quality multiplier. No LTV dependence.
    chan_mult = df["type"].map(B.CHANNEL_ENGAGEMENT_MULTIPLIER).fillna(1.0).values
    df["_engagement_potential"] = sample_engagement_potential(len(df), chan_mult)

    # Forward-causal behaviors from independent engagement
    df["_sessions_d7"] = derive_sessions_d7(df["_engagement_potential"].values)
    df["_ad_views_d7"] = derive_ad_views_d7(df["_sessions_d7"].values)

    # Segment from LTV (industry-standard thresholds)
    df["_segment"] = "free"
    df.loc[df["_real_ltv"] > 0,  "_segment"] = "minnow"
    df.loc[df["_real_ltv"] > 10, "_segment"] = "dolphin"
    df.loc[df["_real_ltv"] > 20, "_segment"] = "whale"

    df["_cohort"]        = "real"
    df["_will_purchase"] = (df["_real_ltv"] > 0).astype(int)
    return df


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure real + synth have same string representation.

    Critical fix: country was 'US' in real backbone and 'United States' in
    synth — the model was learning this string difference as a cohort
    indicator (v1 country importance was 0.39, mostly cohort signal). We
    normalize so the model has to rely on actual demographic signal.
    """
    channel_norm = {
        "organic": "Organic",
        "(direct)": "Organic",
        "(none)": "Organic",
        "Tik Tok Ads": "TikTok Ads",
        "google": "Google Ads",
    }
    df["channel"] = df["type"].map(channel_norm).fillna(df["type"])

    country_norm = {
        "US":      "United States",
        "USA":     "United States",
        "U.S.":    "United States",
        "U.S.A.":  "United States",
        "UK":      "United Kingdom",
        "GB":      "United Kingdom",
        "Korea":   "South Korea",
    }
    df["country"] = df["country"].replace(country_norm)
    return df


def build_features() -> pd.DataFrame:
    print("Loading source tables from Postgres...")
    real_users      = pd.read_sql("SELECT * FROM raw_user_source", engine)
    real_purchases  = pd.read_sql("SELECT * FROM raw_purchases",    engine)
    synth_users     = pd.read_sql("SELECT * FROM synth_users",      engine)
    synth_purchases = pd.read_sql("SELECT * FROM synth_purchases",  engine)
    print(f"  Real users: {len(real_users):,} | Synth users: {len(synth_users):,}")

    # ── Derive INDEPENDENT behavior for real users ───────────────────────────
    print("\nGenerating independent behavior for real users (no LTV leakage)...")
    real_enriched = derive_behavior_for_real_users(real_users, real_purchases)

    # ── Compute LTV target for synth users ───────────────────────────────────
    synth_ltv = (
        synth_purchases.groupby("user_id")["ltv"]
        .max()
        .rename("_real_ltv")
        .reset_index()
    )
    synth_enriched = synth_users.merge(synth_ltv, on="user_id", how="left")
    synth_enriched["_real_ltv"] = synth_enriched["_real_ltv"].fillna(0.0)

    # ── Combine ──────────────────────────────────────────────────────────────
    print("\nCombining real + synth users...")
    common_cols = [
        "user_id", "install_date", "country", "platform", "type",
        "_engagement_potential", "_sessions_d7", "_ad_views_d7",
        "_segment", "_cohort", "_will_purchase", "_real_ltv",
    ]
    real_subset  = real_enriched[common_cols].copy()
    synth_subset = synth_enriched[common_cols].copy()
    combined = pd.concat([real_subset, synth_subset], ignore_index=True)
    print(f"  Combined: {len(combined):,} rows")

    # ── Normalize channel names ──────────────────────────────────────────────
    combined = normalize_columns(combined)

    # ── Engineer derived features ────────────────────────────────────────────
    print("\nEngineering derived features...")
    combined["install_date"] = pd.to_datetime(combined["install_date"], errors="coerce")
    combined["install_dow"]     = combined["install_date"].dt.dayofweek
    combined["install_month"]   = combined["install_date"].dt.month
    combined["install_is_weekend"] = (combined["install_dow"] >= 5).astype(int)

    combined["ads_per_session"]   = combined["_ad_views_d7"] / combined["_sessions_d7"].replace(0, 1)
    combined["sessions_per_day"]  = combined["_sessions_d7"] / 7.0

    combined["engagement_bucket"] = pd.qcut(
        combined["_engagement_potential"],
        q=[0, 0.25, 0.75, 1.0],
        labels=["low", "mid", "high"],
    )

    combined["channel_platform"] = combined["channel"] + "_" + combined["platform"].fillna("unknown")

    # ── Targets ──────────────────────────────────────────────────────────────
    # target_ltv: observed lifetime value at observation window. Real users:
    # D7 snapshot from raw data. Synth users: D30 cumulative by construction.
    # Mixed semantics — see README for the disclaimer downstream consumers see.
    combined["target_ltv"]       = combined["_real_ltv"]
    combined["target_is_payer"]  = combined["_will_purchase"]

    feature_cols = [
        "user_id",
        # Demographics / acquisition
        "country", "platform", "channel", "type",
        "install_date", "install_dow", "install_month", "install_is_weekend",
        # Behavior (D7 snapshot — for synth, causally generated; for real, independent)
        "_engagement_potential", "_sessions_d7", "_ad_views_d7",
        "ads_per_session", "sessions_per_day", "engagement_bucket",
        # Interaction
        "channel_platform",
        # Cohort labels (for filtering / analysis, not training)
        "_segment", "_cohort",
        # Targets
        "target_ltv", "target_is_payer",
    ]
    combined = combined[feature_cols]

    return combined


def main():
    df = build_features()

    print("\n── Feature table summary ──────────────────────────────")
    print(f"Shape: {df.shape}")
    print("\nCohort distribution:")
    print(df["_cohort"].value_counts())
    print("\nSegment distribution:")
    print(df["_segment"].value_counts())
    print(f"\nPayer rate: {df['target_is_payer'].mean()*100:.2f}%")
    print("LTV stats (payers only):")
    print(df.loc[df['target_is_payer']==1, "target_ltv"].describe().round(2))

    print("\n── Channel-LTV signal (was missing in v1) ──")
    print(
        df[df["target_is_payer"] == 1]
        .groupby("channel")["target_ltv"]
        .agg(["count", "mean", "median"])
        .round(2)
        .sort_values("mean", ascending=False)
    )

    print("\nWriting to Postgres → user_features_d7 ...")
    df.to_sql("user_features_d7", engine, if_exists="replace", index=False)
    print(f"  ✓ {len(df):,} rows written")

    print("\n" + "=" * 60)
    print("Feature table ready. Train models against:")
    print("  SELECT * FROM user_features_d7")
    print("=" * 60)


if __name__ == "__main__":
    main()
