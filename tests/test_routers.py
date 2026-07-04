"""API smoke tests — hit every router through FastAPI's TestClient.

Requires the full runtime stack (Postgres + models). Auto-skipped otherwise.
Tests are shallow on purpose: they verify status codes, response shape, and
that the disclaimer / diagnostics fields survive. Deep behaviour is covered
by test_inference.py.
"""
from __future__ import annotations

import pytest


def test_healthz(api_client):
    r = api_client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_root_lists_services(api_client):
    r = api_client.get("/")
    assert r.status_code == 200
    payload = r.json()
    # Exact match — the root endpoint once advertised nonexistent services
    # ("ad-cap", "segment") and a loose >= assertion masked it.
    assert set(payload["services"]) == {
        "propensity", "decide", "channel", "cohort", "personalized", "copilot",
    }


def test_readyz_reports_ready(api_client):
    """With Postgres up and models on disk (this test's preconditions),
    readiness must be green."""
    r = api_client.get("/readyz")
    assert r.status_code == 200
    assert r.json()["ready"] is True


def test_admin_reload_models(api_client):
    r = api_client.post("/admin/reload-models")
    assert r.status_code == 200
    body = r.json()
    assert body["reloaded"] is True
    assert body["propensity_version"] and body["ltv_version"]


def test_openapi_shows_all_endpoints(api_client):
    """OpenAPI must document every endpoint we ship — otherwise Swagger lies."""
    r = api_client.get("/openapi.json")
    assert r.status_code == 200
    paths = set(r.json()["paths"])
    for endpoint in [
        "/propensity/predict", "/propensity/models",
        "/decide/action",
        "/channel/roi",
        "/cohort/retention",
        "/personalized/offer",
    ]:
        assert endpoint in paths, f"{endpoint} missing from OpenAPI"


# ── /propensity ──────────────────────────────────────────────────────────────

def test_propensity_predict_ok(api_client, sample_user_id):
    r = api_client.post("/propensity/predict", json={"user_id": sample_user_id})
    assert r.status_code == 200, r.text
    body = r.json()
    for k in ["p_payer", "expected_ltv_if_payer", "pLTV", "value_segment",
              "gate_applied", "_disclaimer"]:
        assert k in body


def test_propensity_predict_requires_user_id(api_client):
    r = api_client.post("/propensity/predict", json={})
    assert r.status_code == 422


def test_propensity_predict_unknown_user(api_client):
    r = api_client.post("/propensity/predict", json={"user_id": "nope-nope-nope"})
    assert r.status_code == 404


def test_propensity_gate_threshold_bounds(api_client, sample_user_id):
    """gate_threshold=-1 used to be accepted and silently disabled gating."""
    r = api_client.post(
        "/propensity/predict",
        json={"user_id": sample_user_id, "gate_threshold": -1},
    )
    assert r.status_code == 422
    r = api_client.post(
        "/propensity/predict",
        json={"user_id": sample_user_id, "gate_threshold": 1.5},
    )
    assert r.status_code == 422


def test_propensity_models_ok(api_client):
    r = api_client.get("/propensity/models")
    assert r.status_code == 200
    body = r.json()
    assert "propensity" in body and "ltv" in body
    assert "default_threshold" in body["propensity"]


# ── /decide ──────────────────────────────────────────────────────────────────

def test_decide_action_ok(api_client, sample_user_id):
    r = api_client.post(
        "/decide/action",
        json={"user_id": sample_user_id, "context": "app_open"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["action"] in {
        "SHOW_IAP", "SHOW_AD_REWARDED", "SHOW_AD_INTERSTITIAL", "SKIP",
    }
    # Diagnostics must expose the IAP eligibility layer.
    assert "iap_eligible" in body["diagnostics"]
    assert "iap_min_p_payer" in body["diagnostics"]


def test_decide_action_invalid_context(api_client, sample_user_id):
    r = api_client.post(
        "/decide/action",
        json={"user_id": sample_user_id, "context": "hyperspace"},
    )
    assert r.status_code == 422


# ── /channel/roi ─────────────────────────────────────────────────────────────

def test_channel_roi_ok(api_client):
    r = api_client.get("/channel/roi")
    assert r.status_code == 200
    body = r.json()
    assert "overall" in body and "by_channel" in body
    # Honesty disclaimer must not disappear silently.
    assert "_disclaimer" in body


# ── /cohort/retention ────────────────────────────────────────────────────────

@pytest.mark.parametrize("dim", ["channel", "country", "platform"])
def test_cohort_retention_dims(api_client, dim):
    r = api_client.get(f"/cohort/retention?dim={dim}")
    assert r.status_code == 200
    body = r.json()
    assert body["dim"] == dim
    for cohort in body["cohorts"]:
        assert 0 <= cohort["d1_retention"] <= 1
        assert 0 <= cohort["d7_retention"] <= 1
        assert 0 <= cohort["d30_retention"] <= 1


def test_cohort_retention_invalid_dim(api_client):
    r = api_client.get("/cohort/retention?dim=bogus")
    assert r.status_code == 422


# ── /personalized/offer ──────────────────────────────────────────────────────

def test_personalized_offer_ok(api_client, sample_user_id):
    r = api_client.post(
        "/personalized/offer",
        json={"user_id": sample_user_id, "context": "app_open"},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    for k in ["copy", "copy_source", "value_segment", "action"]:
        assert k in body
    for k in ["title", "body", "cta"]:
        assert k in body["copy"]
    # Copy source badge must be one of the three known kinds.
    assert body["copy_source"] in {"llm", "cache", "fallback"}
