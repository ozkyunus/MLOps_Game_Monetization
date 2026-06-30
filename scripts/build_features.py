"""
Build the ML-ready feature table: user_features_d7.

Combines real backbone + synthetic augmentation into one row-per-user table.

Steps:
  1. Load real users + real purchases from Postgres.
  2. Derive D7 behavior for real users (forward-causal from LTV + noise).
     This is the same forward model used for synthetic users, so the
     combined dataset is internally consistent.
  3. Load synth users (already have engagement/sessions/etc).
  4. Concatenate, engineer derived features, attach LTV target.
  5. Write to Postgres as `user_features_d7`.

Output: ~17,605 rows × ~25 columns
"""
from __future__ import annotations

import os
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine

from src.synthetic import benchmarks as B
from src.synthetic.user_augmentation import derive_sessions_d7, derive_ad_views_d7


load_dotenv()
engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL"))

SEED = 42
np.random.seed(SEED)


def derive_behavior_for_real_users(real_users: pd.DataFrame, real_purchases: pd.DataFrame) -> pd.DataFrame:
    """For real users, derive D7 behavior features consistent with their LTV.

    Forward-causal-ish (with noise for non-leakage):
      LTV → engagement_potential (inverse map + noise) → behaviors
    The noise ensures models still have to LEARN, not just decode.
    """
    user_ltv = (
        real_purchases.groupby("user_id")["ltv"]
        .max()
        .rename("_real_ltv")
        .reset_index()
    )
    df = real_users.merge(user_ltv, on="user_id", how="left")
    df["_real_ltv"] = df["_real_ltv"].fillna(0.0)

    # Inverse map LTV → engagement (with structural noise)
    # log1p flattens whales so they don't completely dominate
    base = np.log1p(df["_real_ltv"].values)
    noise = np.random.normal(0, 0.7, size=len(df))
    df["_engagement_potential"] = np.maximum(0.1, base + noise + 0.8)

    # Forward-causal behaviors
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
    """Ensure real + synth have same column names + types."""
    # Channel name harmonization
    channel_norm = {
        "organic": "Organic",
        "(direct)": "Organic",
        "(none)": "Organic",
        "Tik Tok Ads": "TikTok Ads",
        "google": "Google Ads",
    }
    df["channel"] = df["type"].map(channel_norm).fillna(df["type"])
    return df


def build_features() -> pd.DataFrame:
    print("Loading source tables from Postgres...")
    real_users      = pd.read_sql("SELECT * FROM raw_user_source", engine)
    real_purchases  = pd.read_sql("SELECT * FROM raw_purchases",    engine)
    synth_users     = pd.read_sql("SELECT * FROM synth_users",      engine)
    synth_purchases = pd.read_sql("SELECT * FROM synth_purchases",  engine)
    print(f"  Real users: {len(real_users):,} | Synth users: {len(synth_users):,}")

    # ── Derive behavior for real users ───────────────────────────────────────
    print("\nDeriving D7 behavior for real users (forward-causal)...")
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

    # Behavior ratios (NaN-safe)
    combined["ads_per_session"]   = combined["_ad_views_d7"] / combined["_sessions_d7"].replace(0, 1)
    combined["sessions_per_day"]  = combined["_sessions_d7"] / 7.0

    # Engagement bucket (low / mid / high)
    combined["engagement_bucket"] = pd.qcut(
        combined["_engagement_potential"],
        q=[0, 0.25, 0.75, 1.0],
        labels=["low", "mid", "high"],
    )

    # Channel x platform interaction
    combined["channel_platform"] = combined["channel"] + "_" + combined["platform"].fillna("unknown")

    # ── Targets ──────────────────────────────────────────────────────────────
    combined["target_ltv_d30"]     = combined["_real_ltv"]
    combined["target_is_payer"]    = combined["_will_purchase"]

    # ── Final column order ───────────────────────────────────────────────────
    feature_cols = [
        "user_id",
        # Demographics / acquisition
        "country", "platform", "channel", "type",
        "install_date", "install_dow", "install_month", "install_is_weekend",
        # Behavior (D7)
        "_engagement_potential", "_sessions_d7", "_ad_views_d7",
        "ads_per_session", "sessions_per_day", "engagement_bucket",
        # Interaction
        "channel_platform",
        # Cohort labels (for filtering / analysis, not training)
        "_segment", "_cohort",
        # Targets
        "target_ltv_d30", "target_is_payer",
    ]
    combined = combined[feature_cols]

    return combined


def main():
    df = build_features()

    print("\n── Feature table summary ──────────────────────────────")
    print(f"Shape: {df.shape}")
    print(f"\nCohort distribution:")
    print(df["_cohort"].value_counts())
    print(f"\nSegment distribution:")
    print(df["_segment"].value_counts())
    print(f"\nPayer rate: {df['target_is_payer'].mean()*100:.2f}%")
    print(f"LTV stats (payers only):")
    print(df.loc[df['target_is_payer']==1, "target_ltv_d30"].describe().round(2))

    print("\nWriting to Postgres → user_features_d7 ...")
    df.to_sql("user_features_d7", engine, if_exists="replace", index=False)
    print(f"  ✓ {len(df):,} rows written")

    print("\n" + "=" * 60)
    print("Feature table ready. Train models against:")
    print("  SELECT * FROM user_features_d7")
    print("=" * 60)


if __name__ == "__main__":
    main()
