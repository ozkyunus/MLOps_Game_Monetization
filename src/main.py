"""
Player Monetization Intelligence Platform — FastAPI entry point.

Services (each will be a separate router):
  1. /propensity     — purchase probability + expected amount
  2. /decide         — IAP vs Ad vs Skip decision engine
  3. /ad-cap         — per-user ad fatigue + recommended daily cap
  4. /segment        — clustering-based player segment
  5. /personalized   — LLM-generated offer copy

Auth + user management via /auth.
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
        "services": ["propensity", "decide", "ad-cap", "segment", "personalized"],
    }


@app.get("/healthz")
def healthz():
    return {"status": "ok"}


# Routers will be wired in Day 2-4:
# from src.routers import auth, propensity, decide, ad_cap, segment, personalized
# app.include_router(auth.router)
# app.include_router(propensity.router)
# app.include_router(decide.router)
# app.include_router(ad_cap.router)
# app.include_router(segment.router)
# app.include_router(personalized.router)
