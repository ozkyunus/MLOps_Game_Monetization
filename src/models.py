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

from datetime import datetime, timezone
from typing import Optional, Literal
from sqlmodel import SQLModel, Field, text


# -------------------- Auth --------------------

class User(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    username: str = Field(index=True, unique=True)
    email: Optional[str] = None
    password_hash: str
    role: str = Field(default="analyst")  # admin / analyst


class CreateUser(SQLModel):
    username: str
    password: str
    email: Optional[str] = None


class Login(SQLModel):
    username: str
    password: str


class ShowUser(SQLModel):
    id: int
    username: str
    email: Optional[str] = None
    role: str


# -------------------- Player (denormalized features) --------------------

class Player(SQLModel, table=True):
    player_id: str = Field(primary_key=True)  # supports UUIDs from IAP dataset
    age: Optional[float] = None
    gender: Optional[str] = None
    country: Optional[str] = None
    device: Optional[str] = None
    game_genre: Optional[str] = None
    # Activity
    sessions_per_week: Optional[float] = None
    avg_session_duration: Optional[float] = None
    play_time_hours: Optional[float] = None
    player_level: Optional[int] = None
    achievements_unlocked: Optional[int] = None
    # Monetization
    in_app_purchases: Optional[bool] = None
    in_app_purchase_amount: Optional[float] = None  # USD
    first_purchase_days_after_install: Optional[float] = None
    payment_method: Optional[str] = None
    last_purchase_date: Optional[datetime] = None
    spending_segment: Optional[str] = None  # Minnow/Dolphin/Whale
    # Engagement / churn
    engagement_level: Optional[str] = None  # Low/Medium/High
    retention_1: Optional[bool] = None
    retention_7: Optional[bool] = None
    # Synthetic ad data (populated by data_simulator)
    ad_views_total: Optional[int] = None
    ad_views_last_7d: Optional[int] = None
    ad_tolerance_score: Optional[float] = None
    # Bookkeeping
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )
    updated_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
    )


# -------------------- Predictions --------------------

class PredictionLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: str = Field(index=True)
    service: str = Field(index=True)  # propensity / ltv / decision / ad_fatigue / segmentation
    model_version: str
    prediction_value: Optional[float] = None  # numeric output (score, amount)
    prediction_label: Optional[str] = None    # categorical output (segment, action)
    confidence: Optional[float] = None
    features_snapshot: Optional[str] = None   # JSON string for audit
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )


class PropensityRequest(SQLModel):
    player_id: str


class PropensityResponse(SQLModel):
    player_id: str
    purchase_prob_30d: float
    expected_amount: Optional[float] = None
    expected_revenue: float
    segment: str
    model_version: str


class DecisionRequest(SQLModel):
    player_id: str
    context: Optional[str] = None  # "level_complete" / "app_open" / etc.


class DecisionResponse(SQLModel):
    player_id: str
    action: str  # SHOW_IAP / SHOW_AD_REWARDED / SHOW_AD_INTERSTITIAL / SKIP
    offer: Optional[str] = None
    expected_revenue: float
    reasoning: str
    alternatives: list[dict]
    model_version: str


# -------------------- Model registry mirror --------------------

class ModelVersion(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    service: str = Field(index=True)
    version: str
    mlflow_run_id: str
    metrics_json: str   # JSON: {"f1": 0.82, "auc": 0.91, ...}
    is_production: bool = Field(default=False, index=True)
    promoted_at: Optional[datetime] = None
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )


# -------------------- Drift monitoring --------------------

class DriftLog(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    service: str = Field(index=True)
    feature_name: str
    ks_statistic: float
    p_value: float
    drift_detected: bool
    week_number: int
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )


# -------------------- Agent / decision action audit --------------------

class AgentAction(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    player_id: str = Field(index=True)
    action_taken: str
    offer_shown: Optional[str] = None
    outcome: Optional[str] = None  # CONVERTED / DISMISSED / NO_RESPONSE
    revenue_generated: Optional[float] = None
    model_version: str
    ab_test_group: Optional[str] = None
    created_at: Optional[datetime] = Field(
        default_factory=lambda: datetime.now(timezone.utc),
        sa_column_kwargs={"server_default": text("CURRENT_TIMESTAMP")},
    )
