"""Shared database helpers for all dashboard pages.

Centralises engine creation and the sample-user picker query that was
previously copy-pasted across the Propensity / Decision Engine / Offer pages.
"""
from __future__ import annotations

import os

import pandas as pd
import streamlit as st
from _i18n import L
from sqlalchemy import create_engine


@st.cache_resource
def get_engine():
    """Return a shared SQLAlchemy engine, or None if the DB URL is not configured."""
    db_url = os.getenv("SQLALCHEMY_DATABASE_URL")
    if not db_url:
        return None
    return create_engine(db_url)


def require_db():
    """Return the shared engine, or render a friendly bilingual error and stop."""
    engine = get_engine()
    if engine is None:
        st.error(L(
            "Veritabanı yapılandırılmamış — .env dosyasında SQLALCHEMY_DATABASE_URL ayarla.",
            "Database not configured — set SQLALCHEMY_DATABASE_URL in .env.",
        ))
        st.stop()
    return engine


@st.cache_data(ttl=120)
def load_sample_users(_engine) -> pd.DataFrame:
    """Up to 5 sample users per (cohort, segment) pair for the sidebar pickers."""
    q = """
        WITH ranked AS (
            SELECT user_id, _cohort, _segment, target_ltv, target_is_payer,
                   ROW_NUMBER() OVER (PARTITION BY _cohort, _segment ORDER BY user_id) AS rn
            FROM user_features_d7
        )
        SELECT user_id, _cohort AS cohort, _segment AS segment,
               target_ltv AS true_ltv, target_is_payer AS true_payer
        FROM ranked WHERE rn <= 5
        ORDER BY _cohort, _segment
    """
    return pd.read_sql(q, _engine)
