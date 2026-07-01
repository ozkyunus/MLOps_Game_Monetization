"""Pytest fixtures + skip logic.

Tests split into two tiers:
  1. Unit — pure Python, no external services (always run, incl. CI)
  2. Integration — needs Postgres + saved_models/*.joblib.
     Auto-skipped when infra is missing (so CI runs green even without
     bringing up docker-compose).

Set `RUN_INTEGRATION=1` to force integration tests to error instead of skip
(useful locally to verify the stack is really up).

Import-order note: xgboost is imported here BEFORE anything else pulls in
torch / SDV / MLflow. Some macOS wheels segfault during pickle load when
another native library (torch's OpenMP runtime, in particular) initialises
first. Pre-warming xgboost sidesteps that.
"""
from __future__ import annotations

import xgboost  # noqa: F401  # MUST be first — see module docstring

import glob
import os
from pathlib import Path

import pytest
from dotenv import load_dotenv

# Load .env so tests inherit the same DB URL / MLflow URI as the app.
load_dotenv(Path(__file__).resolve().parent.parent / ".env")


# ── Infra availability helpers ───────────────────────────────────────────────

def _postgres_reachable() -> bool:
    """Quick probe — returns True iff we can open a Postgres connection."""
    url = os.getenv("SQLALCHEMY_DATABASE_URL")
    if not url:
        return False
    try:
        from sqlalchemy import create_engine, text
        engine = create_engine(url, pool_pre_ping=True, connect_args={"connect_timeout": 2})
        with engine.connect() as c:
            c.execute(text("SELECT 1"))
        return True
    except Exception:
        return False


def _models_available() -> bool:
    """True iff we have at least one propensity + ltv joblib on disk."""
    root = Path(__file__).resolve().parent.parent
    prop = glob.glob(str(root / "saved_models" / "propensity_v*.joblib"))
    ltv  = glob.glob(str(root / "saved_models" / "ltv_v*.joblib"))
    return bool(prop and ltv)


# Compute once per test session (probes are ~ms).
POSTGRES_UP    = _postgres_reachable()
MODELS_ON_DISK = _models_available()
STRICT         = os.getenv("RUN_INTEGRATION") == "1"


# ── Fixtures ─────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def needs_postgres():
    """Skip test if Postgres unreachable — unless RUN_INTEGRATION=1 (then error)."""
    if not POSTGRES_UP:
        (pytest.fail if STRICT else pytest.skip)(
            "Postgres not reachable — start `docker compose up -d postgres` or "
            "set RUN_INTEGRATION=1 to fail instead of skip."
        )


@pytest.fixture(scope="session")
def needs_models():
    """Skip test if no saved joblib models."""
    if not MODELS_ON_DISK:
        (pytest.fail if STRICT else pytest.skip)(
            "No models in saved_models/. Run the training scripts first."
        )


@pytest.fixture(scope="session")
def sample_user_id(needs_postgres) -> str:
    """A real user_id from the DB to feed into inference/router tests."""
    from sqlalchemy import create_engine, text
    engine = create_engine(os.getenv("SQLALCHEMY_DATABASE_URL"))
    with engine.connect() as c:
        row = c.execute(text(
            "SELECT user_id FROM user_features_d7 "
            "WHERE _cohort='augmented_whale' AND _segment='whale' LIMIT 1"
        )).fetchone()
    if not row:
        pytest.skip("No augmented_whale/whale user found in DB.")
    return row[0]


@pytest.fixture(scope="session")
def api_client(needs_postgres, needs_models):
    """TestClient wired to the real FastAPI app (with lifespan)."""
    from fastapi.testclient import TestClient
    from src.main import app
    with TestClient(app) as client:
        yield client
