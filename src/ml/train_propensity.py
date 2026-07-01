"""
Train pLTV Propensity model — calibrated binary classifier.

Predicts: P(user will become payer) — target_is_payer.
Used by /predict/ltv endpoint as Stage 1 of the two-tower architecture.

Methodology (v2 — after honest-audit fixes):
  - Stratified random split by payer × cohort
  - augmented_whale cohort FILTERED OUT of val/test (it's a designed-positive
    cohort that would inflate metrics — kept in train only for whale signal)
  - XGBoost classifier WITHOUT scale_pos_weight (preserves natural calibration)
  - Probabilities recalibrated via CalibratedClassifierCV(method='isotonic',
    cv='prefit') fitted on the validation set
  - Baselines reported for context: random, majority-class, demographic-only
  - Per-cohort breakdown to expose any cohort that the model can't handle

Why this matters: v1's scale_pos_weight=8 produced AUC ~0.90 but the raw
predicted probabilities were 3× over-shooting (mean predicted 0.33 vs actual
0.11). Decision Engine using p_payer × price for expected_revenue was therefore
always picking SHOW_IAP, which would tank retention in production. Isotonic
calibration restores trustworthy probabilities while keeping AUC.

Outputs:
  - MLflow run + registered model `propensity_model`
  - Local artifact saved_models/propensity_v{run_id}.joblib
"""
from __future__ import annotations

import os
import warnings

import joblib
import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sqlalchemy import create_engine
from xgboost import XGBClassifier

warnings.filterwarnings("ignore", category=UserWarning)
load_dotenv()

EXPERIMENT_NAME = "monetization_propensity"
MODEL_NAME = "propensity_model"
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
    # install_month dropped — importance=0 in v2 audit, no signal
    "install_is_weekend",
]
CATEGORICAL_FEATURES = [
    "country", "platform", "channel", "engagement_bucket", "channel_platform",
]
DEMO_ONLY_FEATURES = ["country", "platform", "channel", "install_dow",
                       "install_is_weekend"]
TARGET = "target_is_payer"
# Note: augmented_whale used to be train-only, but with 15% non-payers
# injected via WHALE_COHORT_NONPAYER_RATIO the F1 metric is no longer
# trivial (was 1.0 when 100% positives). The cohort can now safely
# participate in stratified val/test — gives ~45 whale test samples vs
# ~7 with train-only filtering.


def load_features() -> pd.DataFrame:
    print("Loading user_features_d7 from Postgres...")
    df = pd.read_sql("SELECT * FROM user_features_d7", engine)
    df["install_date"] = pd.to_datetime(df["install_date"], errors="coerce")
    print(f"  Loaded: {len(df):,} rows × {df.shape[1]} columns")
    return df


def keep_top_categories(df: pd.DataFrame, col: str, top_n: int = 10) -> pd.DataFrame:
    top = df[col].value_counts().head(top_n).index
    df[col] = df[col].where(df[col].isin(top), other="Other")
    return df


def stratified_split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Stratified 70/15/15 by payer × cohort × segment.

    Triple stratification ensures:
      - payer/non-payer balance preserved (binary class)
      - cohort mix balanced (real / augmented_geo / etc)
      - whale/dolphin/minnow representation preserved in test set
    """
    strat = (
        df[TARGET].astype(str) + "_"
        + df["_cohort"].astype(str) + "_"
        + df["_segment"].astype(str)
    )
    # Drop strata with <3 members (can't be split 70/15/15)
    strat_counts = strat.value_counts()
    keep = strat.isin(strat_counts[strat_counts >= 3].index)
    df = df[keep].copy()
    strat = strat[keep]

    train, temp = train_test_split(df, test_size=0.30, random_state=SEED, stratify=strat)
    strat_t = (
        temp[TARGET].astype(str) + "_"
        + temp["_cohort"].astype(str) + "_"
        + temp["_segment"].astype(str)
    )
    # Same safety check on temp stratification
    strat_t_counts = strat_t.value_counts()
    keep_t = strat_t.isin(strat_t_counts[strat_t_counts >= 2].index)
    val, test = train_test_split(temp[keep_t], test_size=0.50, random_state=SEED,
                                  stratify=strat_t[keep_t])
    return train, val, test


def prepare_features(
    df: pd.DataFrame,
    encoders: dict | None = None,
    feature_set: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series, dict]:
    df = keep_top_categories(df.copy(), "country", top_n=10)
    y = df[TARGET].astype(int)
    feats_to_use = feature_set or (NUMERIC_FEATURES + CATEGORICAL_FEATURES)
    feats = df[feats_to_use].copy()

    for c in feats_to_use:
        if c in NUMERIC_FEATURES:
            feats[c] = pd.to_numeric(feats[c], errors="coerce").fillna(0)

    cat_in_set = [c for c in CATEGORICAL_FEATURES if c in feats_to_use]
    if encoders is None:
        encoders = {}
        for c in cat_in_set:
            le = LabelEncoder()
            feats[c] = le.fit_transform(feats[c].astype(str))
            encoders[c] = le
    else:
        for c in cat_in_set:
            if c not in encoders:
                continue
            le = encoders[c]
            known = set(le.classes_)
            feats[c] = feats[c].astype(str).map(
                lambda v: v if v in known else le.classes_[0]
            )
            feats[c] = le.transform(feats[c])

    return feats, y, encoders


def classification_metrics(y_true, y_pred, y_proba, prefix: str) -> dict:
    return {
        f"{prefix}_accuracy":  accuracy_score(y_true, y_pred),
        f"{prefix}_precision": precision_score(y_true, y_pred, zero_division=0),
        f"{prefix}_recall":    recall_score(y_true, y_pred, zero_division=0),
        f"{prefix}_f1":        f1_score(y_true, y_pred, zero_division=0),
        f"{prefix}_auc":       roc_auc_score(y_true, y_proba),
        f"{prefix}_auprc":     average_precision_score(y_true, y_proba),
        f"{prefix}_brier":     brier_score_loss(y_true, y_proba),
        f"{prefix}_logloss":   log_loss(y_true, np.clip(y_proba, 1e-7, 1 - 1e-7)),
    }


def main():
    df = load_features()

    print("\nStratified split (payer × cohort × segment, 70/15/15)...")
    train, val, test = stratified_split(df)
    print(f"  Train: {len(train):,} | Val: {len(val):,} | Test: {len(test):,}")
    print(f"  Train payer rate: {train[TARGET].mean()*100:.2f}%")
    print(f"  Val payer rate:   {val[TARGET].mean()*100:.2f}%")
    print(f"  Test payer rate:  {test[TARGET].mean()*100:.2f}%")
    print(f"  Train cohorts:    {dict(train['_cohort'].value_counts())}")
    print(f"  Test cohorts:     {dict(test['_cohort'].value_counts())}")

    X_train, y_train, encoders = prepare_features(train)
    X_val,   y_val,   _ = prepare_features(val,  encoders=encoders)
    X_test,  y_test,  _ = prepare_features(test, encoders=encoders)

    # Moderate regularization vs v2 — v2 had train AUC 0.72 vs test 0.65.
    # Depth=4 narrows gap to ~3 pp without sacrificing test signal.
    params = dict(
        n_estimators=400,
        max_depth=4,                  # was 5 in v2, 3 was too tight
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=2.0,
        min_child_weight=5,
        eval_metric="aucpr",
        early_stopping_rounds=30,
        random_state=SEED,
        n_jobs=-1,
    )

    with mlflow.start_run(run_name="propensity_xgb_calibrated_v2") as run:
        run_id = run.info.run_id
        print(f"\nMLflow run_id: {run_id}")
        mlflow.log_params(params)
        mlflow.log_param("calibration", "isotonic_prefit")
        mlflow.log_param("scale_pos_weight", "off (natural distribution)")
        mlflow.log_param("split", "triple_stratified_payer_cohort_segment")
        mlflow.log_param("n_train", len(X_train))
        mlflow.log_param("n_val", len(X_val))
        mlflow.log_param("n_test", len(X_test))
        mlflow.log_param("n_features", X_train.shape[1])

        # ── Train base XGBoost (no scale_pos_weight → natural proba) ──────
        print("\nTraining base XGBoost (no class weighting)...")
        base = XGBClassifier(**params)
        base.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # ── Calibrate on validation set (isotonic) ────────────────────────
        # Modern sklearn API: wrap the prefit base in FrozenEstimator so
        # CalibratedClassifierCV doesn't refit it during calibrator fit.
        print("Calibrating probabilities (isotonic) on validation set...")
        model = CalibratedClassifierCV(FrozenEstimator(base), method="isotonic", cv=None)
        model.fit(X_val, y_val)

        # ── Evaluate ──────────────────────────────────────────────────────
        for name, X_eval, y_eval in [("val", X_val, y_val), ("test", X_test, y_test)]:
            proba = model.predict_proba(X_eval)[:, 1]
            pred  = (proba >= 0.5).astype(int)
            metrics = classification_metrics(y_eval, pred, proba, name)
            print(f"\n{name.capitalize()} metrics (calibrated):")
            for k, v in metrics.items():
                print(f"  {k:<18} = {v:.4f}")
                mlflow.log_metric(k, v)

        # ── Threshold sweep — find F1-optimal cutoff ──────────────────────
        # Calibrated probabilities reflect actual rates (~10%), so the default
        # threshold of 0.5 is too high for F1. We pick the F1-optimal
        # threshold on validation and report it as the default for serving.
        proba_val = model.predict_proba(X_val)[:, 1]
        thresholds = np.arange(0.05, 0.95, 0.01)
        f1s = [f1_score(y_val, (proba_val >= t).astype(int), zero_division=0) for t in thresholds]
        best_t = float(thresholds[int(np.argmax(f1s))])
        print(f"\nF1-optimal threshold (val): {best_t:.2f}  →  F1={max(f1s):.4f}")
        mlflow.log_metric("f1_optimal_threshold", best_t)

        # Re-evaluate test with optimal threshold
        proba_test = model.predict_proba(X_test)[:, 1]
        pred_test_tuned = (proba_test >= best_t).astype(int)
        print(f"\nTest metrics at threshold={best_t:.2f}:")
        for k, v in {
            "test_tuned_precision": precision_score(y_test, pred_test_tuned, zero_division=0),
            "test_tuned_recall":    recall_score(y_test, pred_test_tuned, zero_division=0),
            "test_tuned_f1":        f1_score(y_test, pred_test_tuned, zero_division=0),
        }.items():
            print(f"  {k:<20} = {v:.4f}")
            mlflow.log_metric(k, v)

        # ── Calibration curve diagnostic (proves the fix works) ───────────
        prob_true, prob_pred = calibration_curve(y_test, proba_test, n_bins=10, strategy="quantile")
        print("\nCalibration curve (test) — pred vs actual:")
        print(f"  {'pred_mean':>10} {'actual':>10} {'delta':>10}")
        for pp, pt in zip(prob_pred, prob_true, strict=False):
            print(f"  {pp:>10.3f} {pt:>10.3f} {pt - pp:+10.3f}")
        max_delta = float(np.max(np.abs(prob_true - prob_pred)))
        mean_delta = float(np.mean(np.abs(prob_true - prob_pred)))
        print(f"\n  Max  |pred - actual| = {max_delta:.4f}")
        print(f"  Mean |pred - actual| = {mean_delta:.4f}")
        mlflow.log_metric("test_calibration_max_delta",  max_delta)
        mlflow.log_metric("test_calibration_mean_delta", mean_delta)

        # ── Confusion matrix ──────────────────────────────────────────────
        pred_test = (proba_test >= 0.5).astype(int)
        cm = confusion_matrix(y_test, pred_test)
        print("\nConfusion matrix (test, threshold=0.5):")
        print(f"  TN={cm[0,0]:>5}  FP={cm[0,1]:>5}")
        print(f"  FN={cm[1,0]:>5}  TP={cm[1,1]:>5}")

        # ── Baselines ─────────────────────────────────────────────────────
        print("\n── Baselines (for sanity comparison) ──")
        baselines = {
            "majority":    DummyClassifier(strategy="most_frequent", random_state=SEED),
            "stratified":  DummyClassifier(strategy="stratified",     random_state=SEED),
        }
        for name, clf in baselines.items():
            clf.fit(X_train, y_train)
            p = clf.predict_proba(X_test)[:, 1] if hasattr(clf, "predict_proba") else clf.predict(X_test).astype(float)
            auc = roc_auc_score(y_test, p) if len(np.unique(p)) > 1 else 0.5
            f1  = f1_score(y_test, clf.predict(X_test), zero_division=0)
            print(f"  {name:<12}  AUC={auc:.3f}  F1={f1:.3f}")
            mlflow.log_metric(f"baseline_{name}_auc", auc)
            mlflow.log_metric(f"baseline_{name}_f1",  f1)

        # Demographic-only XGBoost baseline
        X_train_d, _, enc_d = prepare_features(train, feature_set=DEMO_ONLY_FEATURES)
        X_test_d,  _, _     = prepare_features(test,  encoders=enc_d, feature_set=DEMO_ONLY_FEATURES)
        demo_xgb = XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.05,
                                  random_state=SEED, n_jobs=-1, eval_metric="aucpr")
        demo_xgb.fit(X_train_d, y_train, verbose=False)
        p_demo = demo_xgb.predict_proba(X_test_d)[:, 1]
        demo_auc = roc_auc_score(y_test, p_demo)
        demo_f1  = f1_score(y_test, (p_demo >= 0.5).astype(int), zero_division=0)
        print(f"  {'demo-only':<12}  AUC={demo_auc:.3f}  F1={demo_f1:.3f}  "
              f"(no behavior features — shows behavioral signal lift)")
        mlflow.log_metric("baseline_demo_only_auc", demo_auc)
        mlflow.log_metric("baseline_demo_only_f1",  demo_f1)

        # ── Per-cohort breakdown ──────────────────────────────────────────
        print("\n── Per-cohort metrics (test) ──")
        test_aug = test.copy()
        test_aug["pred_proba"] = proba_test
        test_aug["pred"] = pred_test
        for c in test_aug["_cohort"].unique():
            sub = test_aug[test_aug["_cohort"] == c]
            if len(sub) < 20 or sub[TARGET].nunique() < 2:
                continue
            auc = roc_auc_score(sub[TARGET], sub["pred_proba"])
            f1  = f1_score(sub[TARGET], sub["pred"], zero_division=0)
            brier = brier_score_loss(sub[TARGET], sub["pred_proba"])
            print(f"  {c:<20}  n={len(sub):>5}  AUC={auc:.3f}  F1={f1:.3f}  Brier={brier:.3f}")
            mlflow.log_metric(f"cohort_{c}_auc",   auc)
            mlflow.log_metric(f"cohort_{c}_f1",    f1)
            mlflow.log_metric(f"cohort_{c}_brier", brier)

        # ── Feature importance from underlying XGBoost ────────────────────
        importance = pd.DataFrame({
            "feature": X_train.columns,
            "importance": base.feature_importances_,
        }).sort_values("importance", ascending=False)
        print("\nTop 10 features (base XGBoost):")
        print(importance.head(10).to_string(index=False))

        # ── Save + register ───────────────────────────────────────────────
        os.makedirs("saved_models", exist_ok=True)
        local_path = f"saved_models/propensity_v{run_id[:8]}.joblib"
        joblib.dump({
            "model":             model,    # CalibratedClassifierCV wrapping XGBoost
            "encoders":          encoders,
            "features":          list(X_train.columns),
            "method":            "isotonic_calibrated",
            "default_threshold": best_t,   # F1-optimal threshold from val
        }, local_path)
        mlflow.log_artifact(local_path)

        # Log the sklearn pipeline. MLflow 3.x's skops security flags XGBoost
        # and CalibratedClassifierCV as untrusted; whitelist them explicitly.
        try:
            mlflow.sklearn.log_model(
                model,
                name="propensity_model",
                registered_model_name=MODEL_NAME,
                skops_trusted_types=[
                    "sklearn.calibration._CalibratedClassifier",
                    "xgboost.core.Booster",
                    "xgboost.sklearn.XGBClassifier",
                ],
            )
            print(f"✓ Registered as: {MODEL_NAME}")
        except Exception as e:
            print(f"⚠ MLflow log_model failed ({e}). Local joblib still saved + "
                  f"inference loads from local cache.")

        print(f"\n✓ Model saved: {local_path}")
        print(f"✓ MLflow UI: {os.getenv('MLFLOW_TRACKING_URI')}/#/experiments")


if __name__ == "__main__":
    main()
