"""
/propensity router — two-tower pLTV inference (v2 — calibrated + gated).

POST /propensity/predict
    Body: { "user_id": "abc123", "gate_threshold": 0.20 (optional) }
    Returns: gated pLTV = P(payer) × E[LTV | payer]  if P(payer) ≥ gate, else 0.
    Probabilities are isotonic-calibrated → safe to use as scalar inputs.

GET /propensity/models
    Inspect which model versions are currently served.
"""
from __future__ import annotations

import json

from fastapi import APIRouter, HTTPException
from sqlmodel import Session

from src.database import engine as db_engine
from src.ml.inference import load_models, predict_pltv
from src.models import PredictionLog

router = APIRouter(prefix="/propensity", tags=["propensity"])


def value_segment(pltv: float) -> str:
    if pltv >= 5.0:  return "whale_candidate"
    if pltv >= 2.0:  return "dolphin_candidate"
    if pltv >= 0.5:  return "minnow_candidate"
    return "low_value"


@router.post("/predict")
def predict(payload: dict):
    user_id = payload.get("user_id")
    # If client doesn't pass gate_threshold, let inference pick the cohort-
    # aware default (real=0.10, synth=0.20). Only override if explicitly given.
    raw_gate = payload.get("gate_threshold")
    gate = float(raw_gate) if raw_gate is not None else None
    if not user_id:
        raise HTTPException(status_code=422, detail="user_id required")

    result = predict_pltv(user_id, gate_threshold=gate)
    raw = result["raw_features"]
    segment = value_segment(result["pLTV"])

    with Session(db_engine) as session:
        log = PredictionLog(
            player_id=user_id,
            service="propensity_ltv",
            model_version=result["model_version"],
            prediction_value=result["pLTV"],
            prediction_label=segment,
            confidence=result["p_payer"],
            features_snapshot=json.dumps({
                "_engagement_potential": float(raw["_engagement_potential"].iloc[0]),
                "_sessions_d7":          int(raw["_sessions_d7"].iloc[0]),
                "_segment_real":         str(raw["_segment"].iloc[0]),
                "_cohort":               str(raw["_cohort"].iloc[0]),
                "country":               str(raw["country"].iloc[0]),
                "channel":               str(raw["channel"].iloc[0]),
                "gate_applied":          result["gate_applied"],
            }),
        )
        session.add(log)
        session.commit()

    return {
        "user_id":               user_id,
        "p_payer":               round(result["p_payer"], 4),
        "is_predicted_payer":    result["is_predicted_payer"],
        "default_threshold":     round(result["default_threshold"], 3),
        "expected_ltv_if_payer": round(result["expected_ltv_if_payer"], 2),
        "pLTV":                  round(result["pLTV"], 2),
        "pLTV_raw":              round(result["pLTV_raw"], 2),
        "gate_applied":          result["gate_applied"],
        "gate_threshold":        round(result["gate_threshold"], 2),
        "value_segment":         segment,
        "model_version":         result["model_version"],
        "_disclaimer": (
            "LTV target = observed lifetime value over the available window "
            "(D7 for real backbone, D30 for synthetic users) — not a true D30 "
            "measurement. See README → Limitations for details."
        ),
    }


@router.get("/models")
def model_info():
    models = load_models()
    return {
        "propensity": {
            "version":           models["propensity"]["version"],
            "run_id":            models["propensity"]["run_id"],
            "features":          models["propensity"]["features"],
            "method":            models["propensity"].get("method"),
            "default_threshold": models["propensity"].get("default_threshold"),
        },
        "ltv": {
            "version":           models["ltv"]["version"],
            "run_id":            models["ltv"]["run_id"],
            "features":          models["ltv"]["features"],
            "trained_on":        models["ltv"].get("trained_on"),
        },
    }
