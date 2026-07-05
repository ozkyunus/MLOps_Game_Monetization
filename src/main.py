"""
Player Monetization Intelligence Platform — FastAPI entry point.

Services (one router each):
  1. /propensity     — calibrated p_payer + gated pLTV
  2. /decide         — IAP vs Ad vs Skip decision engine
  3. /channel        — per-channel CPI / ROAS / payback
  4. /cohort         — D1/D7/D30 proxy retention vs benchmarks
  5. /personalized   — LLM-generated offer copy
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from sqlalchemy import text

from src import models  # noqa: F401 — register SQLModel tables
from src.database import create_db_and_tables, get_engine
from src.ml.inference import load_models
from src.routers import channel, cohort, copilot, decide, personalized, propensity


@asynccontextmanager
async def lifespan(app: FastAPI):
    # DDL at startup, not at import — `import src.main` must not need a DB
    # (tests, tooling, and OpenAPI generation all import the module).
    create_db_and_tables()
    # Warm the model cache NOW so the first /readyz probe isn't doing cold
    # joblib loads + MLflow lookups under a 5s probe timeout (that pattern
    # kept fresh pods NotReady for minutes on the K8s cluster).
    try:
        load_models()
    except Exception as exc:
        # Don't block startup — /readyz will keep reporting not-ready with
        # the real reason until models become loadable.
        print(f"⚠ model warm-up failed at startup: {exc}")
    yield


app = FastAPI(
    title="Player Monetization Intelligence Platform",
    description="MLOps capstone — predict, decide, monetize for mobile games.",
    version="0.1.0",
    lifespan=lifespan,
)

# /metrics for Prometheus (ops golden signals: rate, latency, errors,
# in-flight). Custom LLM metrics live in src/metrics.py.
from prometheus_fastapi_instrumentator import Instrumentator  # noqa: E402

Instrumentator(
    excluded_handlers=["/metrics", "/healthz", "/readyz"],
).instrument(app).expose(app)

app.include_router(propensity.router)
app.include_router(decide.router)
app.include_router(channel.router)
app.include_router(cohort.router)
app.include_router(personalized.router)
app.include_router(copilot.router)


@app.get("/")
def root():
    return {
        "message": "Player Monetization Intelligence Platform",
        "version": "0.1.0",
        "docs": "/docs",
        "services": ["propensity", "decide", "channel", "cohort", "personalized", "copilot"],
    }


@app.get("/healthz")
def healthz():
    """Liveness — process is up. Kept dependency-free on purpose (the Docker
    healthcheck hits this; a DB blip should not put the container in a
    restart loop). Use /readyz for dependency checks."""
    return {"status": "ok"}


@app.get("/readyz")
def readyz():
    """Readiness — can this instance actually serve predictions?
    Checks Postgres connectivity and that both model bundles load."""
    problems = []
    try:
        with get_engine().connect() as conn:
            conn.execute(text("SELECT 1"))
    except Exception as exc:
        problems.append(f"postgres: {exc.__class__.__name__}")
    try:
        load_models()
    except Exception as exc:
        problems.append(f"models: {exc.__class__.__name__}: {exc}")
    if problems:
        raise HTTPException(status_code=503, detail={"ready": False, "problems": problems})
    return {"ready": True}


@app.post("/admin/reload-models")
def reload_models():
    """Clear the model cache and load the newest bundles from saved_models/.

    Without this, `load_models()`'s lru_cache meant a retrain was invisible
    to a running server until full restart ("loads newest joblib" was only
    true once per process lifetime).
    """
    load_models.cache_clear()
    m = load_models()
    return {
        "reloaded": True,
        "propensity_version": m["propensity"]["version"],
        "ltv_version":        m["ltv"]["version"],
    }
