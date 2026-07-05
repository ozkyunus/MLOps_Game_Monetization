"""
Shared inference helpers — two-tower pLTV with calibrated propensity + gating.

Used by /propensity, /decide, /personalized routers.

Key v2 fixes:
  - Loads CalibratedClassifierCV-wrapped propensity (proba is trustworthy)
  - Default classification threshold from training (F1-optimal on val)
  - Non-payer gating: if p_payer < gate_threshold, pLTV is forced to 0
    (LTV model is trained on payers only — predictions on non-payers are
    out-of-distribution noise).
"""
from __future__ import annotations

import os
from functools import lru_cache

import joblib
import mlflow
import numpy as np
import pandas as pd
from fastapi import HTTPException
from sqlalchemy import create_engine, text

from src.ml.preprocessing import select_raw_inputs

mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))


# Lazy feature engine — created on first DB query. Keeps import cheap so
# constant-only tests (test_synthetic.TestIndustryConstants) don't need a
# live database.
_feature_engine = None


def _get_feature_engine():
    global _feature_engine
    if _feature_engine is None:
        _feature_engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL"))
    return _feature_engine


# Cohort-aware non-payer gate.
#
# A single gate at 0.20 made the model 95% silent — Decision Engine couldn't
# act on most real-cohort users (who legitimately have low p_payer because we
# have no telemetry for them). Production deployment would have real telemetry
# → real users would behave like synth cohort. But for THIS dataset, real-cohort
# predictions are uninformative, so we use a separate (lower) gate for them.
#
# Reasoning:
#   - synth cohorts have full behavioral signal → 0.20 is appropriate
#   - real cohort has random behavior → most p_payer ~ base rate (0.09)
#       lowering to 0.10 catches the marginal users without going below baseline
NONPAYER_GATE_DEFAULT = 0.20
COHORT_GATE_OVERRIDE = {
    "real": 0.10,
    # synth cohorts use default 0.20
}


@lru_cache(maxsize=1)
def load_models() -> dict:
    """Load propensity + ltv — local-first, MLflow registry as metadata.

    MLflow 3.x's skops security has been flaky for our XGBoost+Calibrated
    bundles. We treat the local saved_models/*.joblib (with mtime-sorted
    'latest' selection) as the source of truth, and use MLflow registry
    only for run_id / version metadata when available. Either way the
    model loads.
    """
    prop_path, prop_version, prop_run = _resolve_latest("propensity")
    ltv_path,  ltv_version,  ltv_run  = _resolve_latest("ltv")

    prop_bundle = joblib.load(prop_path)
    ltv_bundle  = joblib.load(ltv_path)

    return {
        "propensity": {**prop_bundle, "version": prop_version, "run_id": prop_run, "path": prop_path},
        "ltv":        {**ltv_bundle,  "version": ltv_version,  "run_id": ltv_run,  "path": ltv_path},
    }


def _resolve_latest(prefix: str) -> tuple[str, str, str]:
    """Find newest saved_models/{prefix}_v*.joblib by mtime.

    Returns (path, version, run_id) where version/run_id come from MLflow
    if the run_id matches, else 'local-{timestamp}' as fallback.
    """
    import glob
    candidates = glob.glob(f"saved_models/{prefix}_v*.joblib")
    if not candidates:
        # Models are not code: the CI-built image ships WITHOUT joblib
        # artifacts (100MB+, gitignored). In that case the MLflow registry
        # is the model's home — pull the newest registered version's
        # artifact into saved_models/ and continue as usual.
        candidates = _pull_from_registry(prefix)
    if not candidates:
        raise FileNotFoundError(
            f"No saved_models/{prefix}_v*.joblib found locally or in the "
            f"MLflow registry. Run `uv run python -m src.ml.train_{prefix}` "
            f"against the tracking server first."
        )
    # Newest by mtime
    latest = max(candidates, key=os.path.getmtime)
    run_id_short = latest.split(f"{prefix}_v")[-1].replace(".joblib", "")

    # Try to find the matching MLflow run_id and registry version
    try:
        client = mlflow.MlflowClient()
        all_versions = client.search_model_versions(f"name='{prefix}_model'")
        match = [v for v in all_versions if v.run_id.startswith(run_id_short)]
        if match:
            v = match[0]
            return latest, v.version, v.run_id
    except Exception:
        pass

    return latest, f"local-{run_id_short}", run_id_short


def _pull_from_registry(prefix: str) -> list[str]:
    """Download the newest registered {prefix}_model joblib from MLflow.

    The training scripts log the bundle via mlflow.log_artifact(), so the
    artifact lives at the run root named {prefix}_v{run8}.joblib and is
    served through the tracking server's artifact proxy.
    """
    try:
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{prefix}_model'")
        if not versions:
            return []
        newest = max(versions, key=lambda m: int(m.version))
        os.makedirs("saved_models", exist_ok=True)
        for art in client.list_artifacts(newest.run_id):
            if art.path.startswith(f"{prefix}_v") and art.path.endswith(".joblib"):
                local = mlflow.artifacts.download_artifacts(
                    run_id=newest.run_id, artifact_path=art.path,
                    dst_path="saved_models")
                print(f"✓ pulled {art.path} from MLflow registry "
                      f"(model v{newest.version})")
                return [local]
    except Exception as exc:
        print(f"⚠ MLflow registry pull failed for {prefix}: {exc}")
    return []


def fetch_user_features(user_id: str) -> pd.DataFrame:
    q = text("SELECT * FROM user_features_d7 WHERE user_id = :uid LIMIT 1")
    df = pd.read_sql(q, _get_feature_engine(), params={"uid": user_id})
    if df.empty:
        raise HTTPException(status_code=404, detail=f"user_id '{user_id}' not in feature table")
    return df


def model_inputs(df: pd.DataFrame, bundle: dict) -> pd.DataFrame:
    """Select raw input columns for the bundled Pipeline.

    All preprocessing (top-N bucketing, one-hot, imputation) lives INSIDE the
    fitted Pipeline shipped in the bundle — nothing is re-derived here. A
    missing column is schema drift and fails loudly instead of being
    zero-filled into silently-garbage predictions.
    """
    try:
        return select_raw_inputs(df, bundle["features"])
    except KeyError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


def predict_pltv(user_id: str, gate_threshold: float | None = None) -> dict:
    """Run the two-tower inference with cohort-aware gating.

    If gate_threshold is None (default), the gate is chosen from the user's
    cohort: synth users use 0.20, real users use 0.10. This is the honest
    operating point — real users would have higher gates in production once
    real telemetry replaced the random behavior features.

    Pass an explicit gate_threshold (e.g. 0.20) to override and use a single
    cutoff for all users.
    """
    models = load_models()
    raw = fetch_user_features(user_id)
    cohort = str(raw["_cohort"].iloc[0])
    gate_used = gate_threshold if gate_threshold is not None else COHORT_GATE_OVERRIDE.get(
        cohort, NONPAYER_GATE_DEFAULT
    )

    # Stage 1 — calibrated propensity (bundled Pipeline: prep + calibrated clf)
    p_feats = model_inputs(raw, models["propensity"])
    p_payer = float(models["propensity"]["model"].predict_proba(p_feats)[0, 1])
    threshold = models["propensity"].get("default_threshold", 0.5)
    is_predicted_payer = bool(p_payer >= threshold)

    # Stage 2 — LTV regressor (bundled Pipeline: prep + regressor)
    l_feats = model_inputs(raw, models["ltv"])
    ltv_pred = float(models["ltv"]["model"].predict(l_feats)[0])
    # No silent default: guessing "log1p" for a direct-$ model would expm1()
    # a $30 prediction into ~$10^13. A bundle missing the key must fail loudly.
    transform = models["ltv"].get("target_transform")
    if transform is None:
        raise HTTPException(
            status_code=500,
            detail="LTV bundle missing 'target_transform' — retrain the LTV model",
        )
    if transform == "log1p":
        expected_ltv_if_payer = float(np.expm1(max(ltv_pred, 0.0)))
    else:  # "none" / direct $ regression
        expected_ltv_if_payer = float(max(ltv_pred, 0.0))

    pltv_raw = p_payer * expected_ltv_if_payer
    gate_applied = p_payer < gate_used
    pltv = 0.0 if gate_applied else pltv_raw

    return {
        "user_id":                user_id,
        "cohort":                 cohort,
        "p_payer":                p_payer,
        "default_threshold":      threshold,
        "is_predicted_payer":     is_predicted_payer,
        "expected_ltv_if_payer":  expected_ltv_if_payer,
        "pLTV":                   pltv,
        "pLTV_raw":               pltv_raw,
        "gate_applied":           gate_applied,
        "gate_threshold":         gate_used,
        "gate_source":            "explicit" if gate_threshold is not None else "cohort_default",
        "raw_features":           raw,
        "model_version": (
            f"propensity:v{models['propensity']['version']}"
            f"+ltv:v{models['ltv']['version']}"
        ),
    }
