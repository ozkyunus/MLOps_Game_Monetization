"""
Player Monetization Intelligence Platform — FastAPI entry point.

Services (one router each):
  1. /propensity     — calibrated p_payer + gated pLTV
  2. /decide         — IAP vs Ad vs Skip decision engine
  3. /channel        — per-channel CPI / ROAS / payback
  4. /cohort         — D1/D7/D30 proxy retention vs benchmarks
  5. /personalized   — LLM-generated offer copy
"""
from fastapi import FastAPI

from src import models  # noqa: F401 — register SQLModel tables
from src.database import create_db_and_tables
from src.routers import channel, cohort, decide, personalized, propensity

app = FastAPI(
    title="Player Monetization Intelligence Platform",
    description="MLOps capstone — predict, decide, monetize for mobile games.",
    version="0.1.0",
)

create_db_and_tables()
app.include_router(propensity.router)
app.include_router(decide.router)
app.include_router(channel.router)
app.include_router(cohort.router)
app.include_router(personalized.router)


@app.get("/")
def root():
    return {
        "message": "Player Monetization Intelligence Platform",
        "version": "0.1.0",
        "docs": "/docs",
        "services": ["propensity", "decide", "channel", "cohort", "personalized"],
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}
