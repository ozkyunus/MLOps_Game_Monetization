"""
SQLModel tables + Pydantic schemas for Player Monetization Intelligence Platform.

Tables:
- User           : auth (admin / analyst)
- Player         : player-level features (denormalized from datasets)
- PredictionLog  : every prediction served (Servis 1, 2, 3)
- ModelVersion   : MLflow registry mirror — which model is in production
- DriftLog       : weekly drift scores
- AgentAction    : decision engine actions taken (for replay / A/B analysis)
"""

from datetime import UTC, datetime
from typing import Literal

from sqlmodel import Field, SQLModel, text

# -------------------- Auth --------------------

class User(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: str | None = None
    password_hash: str
    role: str = Field(default="analyst")  # admin / analyst


class CreateUser(SQLModel):
    username: str
    password: str
    email: str | None = None


class Login(SQLModel):
    username: str
    password: str


class ShowUser(SQLModel):
    id: int
    username: str
    email: str | None = None
    role: str


# -------------------- Player (denormalized features) --------------------

class Player(SQLModel, table=True):
    player_id: str = Field(primary_key=True)  # supports UUIDs from IAP dataset
    age: float | None = None
    gender: str | None = None
    country: str | None = None
    device: str | None = None
    game_genre: str | None = None
    # Activity
    sessions_per_week: float | None = None
    avg_session_duration: float | None = None
    play_time_hours: float | None = None
    player_level: int | None = None
    achievements_unlocked: int | None = None
    # Monetization
    in_app_purchases: bool | None = None
    in_app_purchase_amount: float | None = None  # USD
    first_purchase_days_after_install: float | None = None
    payment_method: str | None = None
    last_purchase_date: datetime | None = None
    spending_segment: str | None = None  # Minnow/Dolphin/Whale
    # Engagement / churn
    engagement_level: str | None = None  # Low/Medium/High
    retention_1: bool | None = None
    retention_7: bool | None = None
    # Synthetic ad data (populated by data_simulator)
    ad_views_total: int | None = None
    ad_views_last_7d: int | None = None
    ad_tolerance_score: float | None = None
    # Bookkeeping
    created_at: datetime | None = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )
    updated_at: datetime | None = Field(
        default_factory=lambda: datetime.now(UTC),
    )


# -------------------- Predictions --------------------

class PredictionLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    player_id: str = Field(index=True)
    service: str = Field(index=True)  # propensity / ltv / decision / ad_fatigue / segmentation
    model_version: str
    prediction_value: float | None = None  # numeric output (score, amount)
    prediction_label: str | None = None    # categorical output (segment, action)
    confidence: float | None = None
    features_snapshot: str | None = None   # JSON string for audit
    created_at: datetime | None = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )


# Request schemas — wired into the routers so FastAPI validates payloads
# (v2 accepted raw dicts: {"user_id": 123} reached the SQL layer and 500'd,
# gate_threshold=-1 silently disabled gating, and Swagger showed no schema).

class PropensityRequest(SQLModel):
    user_id: str
    # None → cohort-aware default gate (real=0.10, synth=0.20).
    gate_threshold: float | None = Field(default=None, ge=0.0, le=1.0)


class DecisionRequest(SQLModel):
    user_id: str
    context: Literal["level_complete", "after_loss", "app_open"] = "app_open"


class CopilotRequest(SQLModel):
    question: str
    conversation_id: str | None = None


# -------------------- Copilot telemetry --------------------
# One row per copilot turn — the LLM analogue of PredictionLog. Feeds the
# eval harness, the Grafana LLM dashboard, and cost tracking.

class CopilotLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    conversation_id: str | None = Field(default=None, index=True)
    question: str
    answer: str
    refused: bool = False
    sources: str | None = None        # JSON list of {doc_id,title,section}
    tool_calls: str | None = None     # JSON list of {tool,args,ms}
    prompt_version: str
    model_name: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: float = 0.0
    retrieval_ms: float = 0.0
    created_at: datetime | None = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )


# -------------------- Model registry mirror --------------------

class ModelVersion(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    service: str = Field(index=True)
    version: str
    mlflow_run_id: str
    metrics_json: str   # JSON: {"f1": 0.82, "auc": 0.91, ...}
    is_production: bool = Field(default=False, index=True)
    promoted_at: datetime | None = None
    created_at: datetime | None = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )


# -------------------- Drift monitoring --------------------

class DriftLog(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    service: str = Field(index=True)
    feature_name: str
    ks_statistic: float
    p_value: float
    drift_detected: bool
    week_number: int
    created_at: datetime | None = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )


# -------------------- Agent / decision action audit --------------------

class AgentAction(SQLModel, table=True):
    id: int | None = Field(default=None, primary_key=True)
    player_id: str = Field(index=True)
    action_taken: str
    offer_shown: str | None = None
    outcome: str | None = None  # CONVERTED / DISMISSED / NO_RESPONSE
    revenue_generated: float | None = None
    model_version: str
    ab_test_group: str | None = None
    created_at: datetime | None = Field(
        default_factory=lambda: datetime.now(UTC),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )
