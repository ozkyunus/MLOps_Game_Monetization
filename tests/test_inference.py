"""Integration tests for src/ml/inference.py.

Requires Postgres + saved_models/*.joblib. Auto-skipped otherwise (see conftest).
"""
from __future__ import annotations

import pytest


def test_load_models_returns_both_towers(needs_models):
    from src.ml.inference import load_models
    models = load_models()
    assert "propensity" in models
    assert "ltv" in models
    for k in ("model", "encoders", "features"):
        assert k in models["propensity"], f"propensity bundle missing '{k}'"
        assert k in models["ltv"],       f"ltv bundle missing '{k}'"


def test_propensity_features_match_ltv(needs_models):
    """Both models expect the same feature schema — otherwise inference blows up."""
    from src.ml.inference import load_models
    models = load_models()
    assert set(models["propensity"]["features"]) == set(models["ltv"]["features"])


def test_predict_pltv_returns_full_shape(sample_user_id):
    from src.ml.inference import predict_pltv
    result = predict_pltv(sample_user_id)

    # Required keys — anything missing breaks the /propensity/predict router.
    required = {
        "user_id", "cohort", "p_payer", "expected_ltv_if_payer",
        "pLTV", "pLTV_raw", "gate_applied", "gate_threshold",
        "default_threshold", "is_predicted_payer", "model_version",
    }
    assert required.issubset(result.keys()), f"missing keys: {required - result.keys()}"

    # Types + numeric ranges — catch dtype drift.
    assert 0.0 <= result["p_payer"] <= 1.0
    assert result["expected_ltv_if_payer"] >= 0
    assert result["pLTV"] >= 0
    assert result["pLTV_raw"] >= 0


def test_predict_pltv_gate_zeroes_low_probability(sample_user_id):
    """With a high gate threshold every real-cohort user should be gated to 0."""
    from src.ml.inference import predict_pltv
    result = predict_pltv(sample_user_id, gate_threshold=0.99)
    assert result["gate_applied"] is True
    assert result["pLTV"] == 0.0
    # But pLTV_raw (before the gate) still reflects the model output.
    assert result["pLTV_raw"] >= 0


def test_predict_pltv_cohort_aware_default(sample_user_id):
    """Auto gate threshold varies by cohort: 0.20 for synth, 0.10 for real."""
    from src.ml.inference import predict_pltv
    result = predict_pltv(sample_user_id)   # augmented_whale/whale
    assert result["gate_source"] == "cohort_default"
    assert result["gate_threshold"] == pytest.approx(0.20)   # synth cohort


def test_predict_pltv_unknown_user_raises():
    """Feature fetcher must raise HTTPException (404) for unseen user_ids."""
    from fastapi import HTTPException
    from src.ml.inference import predict_pltv
    with pytest.raises(HTTPException) as ex:
        predict_pltv("this-user-does-not-exist-in-the-db-abc123")
    assert ex.value.status_code == 404
