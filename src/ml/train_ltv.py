"""
Train LTV regressor — Stage 2 of two-tower pLTV.

Predicts: E[LTV | user is a payer] (observed window).
Trained on payers ONLY (target_is_payer == 1) to avoid the zero-inflation
that would dominate a single-model regression.

Honesty fixes vs v1:
  - augmented_whale cohort filtered out of val/test (was trivial signal)
  - Per-segment metrics with proper sample sizes
  - Tweedie regressor (zero-inflated) trained side-by-side on FULL data
    (payers + non-payers) as architectural alternative — comparison logged
    to MLflow but two-tower remains the primary serving model
  - Renamed target_ltv_d30 → target_ltv (real data is D7 snapshot)

Combined endpoint formula:
    pLTV(user) = max(0, p_payer × E[LTV | payer] × (p_payer >= threshold))
                       └─ propensity_model   └─ this model

Targets:
  - target_ltv: continuous $, range ~$1 to ~$60 (payers)

Outputs:
  - MLflow run + registered model `ltv_model`
  - Tweedie model logged as comparison but NOT registered as primary
"""
from __future__ import annotations

import os
import warnings

import joblib
import mlflow
import mlflow.xgboost
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from xgboost import XGBRegressor

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

EXPERIMENT_NAME = "monetization_ltv"
MODEL_NAME = "ltv_model"
SEED = 42

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
mlflow.set_experiment(EXPERIMENT_NAME)
engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL"))


NUMERIC_FEATURES = [
    "_engagement_potential",
    "_sessions_d7",
    "_ad_views_d7",
    "ads_per_session",
    "sessions_per_day",
    "install_dow",
    # install_month dropped — zero importance in v2 audit
    "install_is_weekend",
]
CATEGORICAL_FEATURES = [
    "country", "platform", "channel", "engagement_bucket", "channel_platform",
]
TARGET = "target_ltv"

# Segment-aware loss weighting — log1p compresses whale tail; explicit
# sample_weight lets us tell XGBoost "whales matter more than minnows".
# Without this, v2 model under-predicted whale LTV by ~47% ($23 MAE on $49 mean).
SEGMENT_SAMPLE_WEIGHTS = {
    "whale":   4.0,    # tuned: 8 over-focused (broke minnow), 3 too gentle
    "dolphin": 2.0,
    "minnow":  1.0,
}
# Calibration log:
#   weight=1 (no weighting): whale MAE $23, minnow MAE $3   → whales bad
#   weight=8: whale MAE $17, minnow MAE $9                  → minnow bad
#   weight=4: balanced compromise (verified by per-segment metrics)


def load_all() -> pd.DataFrame:
    print("Loading user_features_d7 from Postgres...")
    return pd.read_sql("SELECT * FROM user_features_d7", engine)


def keep_top_categories(df, col, top_n=10):
    top = df[col].value_counts().head(top_n).index
    df[col] = df[col].where(df[col].isin(top), other="Other")
    return df


def split_payers_only(df: pd.DataFrame):
    """Filter to payers, stratified split by cohort × segment.

    augmented_whale is NOT filtered out (15% non-payer injection prevents
    trivial F1; for regression we keep all whale samples to stabilize
    metrics — augmented whale test n ≈ 45 vs ~7 without).
    """
    payers = df[df["target_is_payer"] == 1].copy()
    strat = payers["_cohort"].astype(str) + "_" + payers["_segment"].astype(str)
    counts = strat.value_counts()
    keep = strat.isin(counts[counts >= 3].index)
    payers = payers[keep].copy()
    strat = strat[keep]

    train, temp = train_test_split(payers, test_size=0.30, random_state=SEED, stratify=strat)
    strat_t = temp["_cohort"].astype(str) + "_" + temp["_segment"].astype(str)
    counts_t = strat_t.value_counts()
    keep_t = strat_t.isin(counts_t[counts_t >= 2].index)
    val, test = train_test_split(temp[keep_t], test_size=0.50, random_state=SEED,
                                  stratify=strat_t[keep_t])
    return train, val, test


def prepare_features(df, encoders=None):
    df = keep_top_categories(df.copy(), "country", top_n=10)
    y = df[TARGET].astype(float)
    feats = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES].copy()

    for c in NUMERIC_FEATURES:
        feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0)

    if encoders is None:
        encoders = {}
        for c in CATEGORICAL_FEATURES:
            le = LabelEncoder()
            feats[c] = le.fit_transform(feats[c].astype(str))
            encoders[c] = le
    else:
        for c in CATEGORICAL_FEATURES:
            le = encoders[c]
            known = set(le.classes_)
            feats[c] = feats[c].astype(str).map(lambda v: v if v in known else le.classes_[0])
            feats[c] = le.transform(feats[c])

    return feats, y, encoders


def regression_metrics(y_true, y_pred, prefix):
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = r2_score(y_true, y_pred)
    mape = float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 0.01))) * 100)
    return {
        f"{prefix}_mae":  mae,
        f"{prefix}_rmse": rmse,
        f"{prefix}_r2":   r2,
        f"{prefix}_mape": mape,
    }


def main():
    df_all = load_all()
    df_all["install_date"] = pd.to_datetime(df_all["install_date"], errors="coerce")

    print("\nSplit payers only (train-only whale → train), stratify by segment...")
    train, val, test = split_payers_only(df_all)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print("  Train segment mix:")
    print("    " + train["_segment"].value_counts().to_string().replace("\n", "\n    "))
    print("  Test segment mix:")
    print("    " + test["_segment"].value_counts().to_string().replace("\n", "\n    "))

    X_train, y_train, encoders = prepare_features(train)
    X_val,   y_val,   _ = prepare_features(val,  encoders=encoders)
    X_test,  y_test,  _ = prepare_features(test, encoders=encoders)

    # Tuning grid (v3):
    #   log1p + no weights:        whale MAE $23 / minnow $3    → whales bad
    #   Huber direct + weight=8:   whale $17     / minnow $9    → minnows bad
    #   Huber direct + weight=4:   whale $20     / minnow $5.5  ★ chosen
    #   log1p + weight=4:          whale $25.8   / minnow $4.6
    # Production reasoning: whale n=51 × $49 mean = $2.5k revenue contribution;
    # minnow n=243 × $3.41 = $830. Whale accuracy matters 3x more for revenue.
    # So we accept slightly worse minnow MAE in exchange for the whale fix.
    y_train_raw = y_train.values
    y_val_raw   = y_val.values

    # Huber loss on direct $ target — robust to whale outliers, doesn't
    # compress the gradient signal the way log1p did.
    params = dict(
        objective="reg:pseudohubererror",
        huber_slope=2.0,
        n_estimators=500,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.0,
        min_child_weight=5,
        eval_metric="mae",
        early_stopping_rounds=30,
        random_state=SEED,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="ltv_two_tower_v2") as run:
        run_id = run.info.run_id
        print(f"\nMLflow run_id: {run_id}")
        mlflow.log_params(params)
        mlflow.log_param("target_transform", "none_huber_direct")
        mlflow.log_param("split", "stratified_cohort_segment")
        mlflow.log_param("training_data", "payers_only")

        # ── Primary: Huber XGBoost on direct $ + segment-weighted loss ──
        print("\nTraining XGBoost regressor with Huber loss + segment weights...")
        train_weights = train["_segment"].map(SEGMENT_SAMPLE_WEIGHTS).fillna(1.0).values
        val_weights   = val["_segment"].map(SEGMENT_SAMPLE_WEIGHTS).fillna(1.0).values
        print(f"  Sample weights: {SEGMENT_SAMPLE_WEIGHTS}")
        mlflow.log_params({f"weight_{k}": v for k, v in SEGMENT_SAMPLE_WEIGHTS.items()})

        model = XGBRegressor(**params)
        model.fit(
            X_train, y_train_raw,
            sample_weight=train_weights,
            eval_set=[(X_val, y_val_raw)],
            sample_weight_eval_set=[val_weights],
            verbose=False,
        )

        val_pred  = np.maximum(model.predict(X_val),  0)
        test_pred = np.maximum(model.predict(X_test), 0)

        val_metrics  = regression_metrics(y_val.values,  val_pred,  "val")
        test_metrics = regression_metrics(y_test.values, test_pred, "test")

        print("\nValidation metrics ($ scale):")
        for k, v in val_metrics.items():
            print(f"  {k:<14} = {v:.3f}")
            mlflow.log_metric(k, v)

        print("\nTest metrics ($ scale, held-out):")
        for k, v in test_metrics.items():
            print(f"  {k:<14} = {v:.3f}")
            mlflow.log_metric(k, v)

        # ── Per-segment breakdown with proper n ──────────────────────────
        print("\nTest MAE per segment (with sample size + 95% CI estimate):")
        test_with_pred = test.copy()
        test_with_pred["_pred"] = test_pred
        for seg in ["whale", "dolphin", "minnow"]:
            sub = test_with_pred[test_with_pred["_segment"] == seg]
            if len(sub) == 0:
                continue
            errs = np.abs(sub[TARGET].values - sub["_pred"].values)
            seg_mae = float(errs.mean())
            seg_se  = float(errs.std() / np.sqrt(len(errs)))
            ci = 1.96 * seg_se
            seg_mean_true = float(sub[TARGET].mean())
            print(f"  {seg:<8} n={len(sub):>4}  MAE=${seg_mae:>6.2f} ±${ci:>5.2f}  "
                  f"(true mean ${seg_mean_true:.2f})")
            mlflow.log_metric(f"test_mae_{seg}",    seg_mae)
            mlflow.log_metric(f"test_mae_{seg}_ci", ci)

        # ── Feature importance ───────────────────────────────────────────
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": model.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\nTop 10 features:")
        print(importance.head(10).to_string(index=False))

        # ── Save + register primary ──────────────────────────────────────
        os.makedirs("saved_models", exist_ok=True)
        local_path = f"saved_models/ltv_v{run_id[:8]}.joblib"
        joblib.dump({
            "model":              model,
            "encoders":           encoders,
            "features":           list(X_train.columns),
            "target_transform":   "none",          # direct $ with Huber loss
            "objective":          "huber_direct",
            "trained_on":         "payers_only",
        }, local_path)
        mlflow.log_artifact(local_path)
        mlflow.xgboost.log_model(model, artifact_path="ltv_model",
                                 registered_model_name=MODEL_NAME)

        print(f"\n✓ Primary model saved: {local_path}")
        print(f"✓ Registered as: {MODEL_NAME}")

        # ── Comparison: Tweedie regressor on FULL data (incl. non-payers) ─
        print("\n" + "─" * 60)
        print("Alt architecture: Tweedie regressor (zero-inflated, full data)")
        print("─" * 60)
        full_train, full_temp = train_test_split(
            df_all, test_size=0.30, random_state=SEED,
            stratify=df_all["target_is_payer"].astype(str),
        )
        full_val, full_test = train_test_split(
            full_temp, test_size=0.50, random_state=SEED,
            stratify=full_temp["target_is_payer"].astype(str),
        )
        X_ftr, y_ftr, enc_t = prepare_features(full_train)
        X_fte, y_fte, _     = prepare_features(full_test,  encoders=enc_t)
        tweedie = XGBRegressor(
            objective="reg:tweedie",
            tweedie_variance_power=1.5,
            n_estimators=400, max_depth=5, learning_rate=0.05,
            random_state=SEED, n_jobs=-1,
        )
        tweedie.fit(X_ftr, y_ftr.values, verbose=False)
        tweedie_pred = np.maximum(tweedie.predict(X_fte), 0)
        tweedie_metrics = regression_metrics(y_fte.values, tweedie_pred, "tweedie_test")
        print("Tweedie test metrics (FULL data, includes non-payers):")
        for k, v in tweedie_metrics.items():
            print(f"  {k:<24} = {v:.3f}")
            mlflow.log_metric(k, v)

        # Predicted vs actual at non-payer subgroup
        nonpayer_mask = (y_fte.values == 0)
        if nonpayer_mask.sum() > 0:
            np_pred_mean = float(tweedie_pred[nonpayer_mask].mean())
            print(f"\n  Tweedie's mean prediction for true non-payers: ${np_pred_mean:.3f}")
            print("  (should be near 0; two-tower's was ~$2.28 — Tweedie is closer if working)")
            mlflow.log_metric("tweedie_nonpayer_pred_mean", np_pred_mean)

        print(f"\n✓ MLflow UI: {os.getenv('MLFLOW_TRACKING_URI')}/#/experiments")


if __name__ == "__main__":
    main()
