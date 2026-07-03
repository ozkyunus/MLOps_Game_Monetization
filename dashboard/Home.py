"""
Player Monetization Intelligence Platform — Streamlit dashboard.

Home page: platform overview, health checks, and dataset statistics.
Language toggle (🇹🇷/🇬🇧) lives in the sidebar and persists across pages.
"""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from _data import get_engine
from _i18n import L, sidebar_lang_toggle
from dotenv import load_dotenv
from sqlalchemy import text

# ── Setup ────────────────────────────────────────────────────────────────────
load_dotenv()
API_URL    = os.getenv("API_URL", "http://localhost:8000")
MLFLOW_URL = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5001")

# Browser-facing URLs for link buttons (inside Docker the container-internal
# hostnames are not reachable from the user's browser).
PUBLIC_API_URL    = os.getenv("PUBLIC_API_URL", API_URL)
PUBLIC_MLFLOW_URL = os.getenv("PUBLIC_MLFLOW_URL", MLFLOW_URL)

st.set_page_config(
    page_title="Monetization Intelligence Platform",
    page_icon="🎮",
    layout="wide",
    initial_sidebar_state="expanded",
)

sidebar_lang_toggle()


# ── Helpers ──────────────────────────────────────────────────────────────────

@st.cache_data(ttl=60)
def check_api_health() -> dict:
    try:
        r = requests.get(f"{API_URL}/healthz", timeout=2)
        return {"up": r.status_code == 200}
    except Exception:
        return {"up": False}


@st.cache_data(ttl=60)
def check_mlflow_health() -> bool:
    try:
        r = requests.get(f"{MLFLOW_URL}/", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


@st.cache_data(ttl=60)
def load_platform_stats() -> dict:
    e = get_engine()
    stats = {}
    with e.connect() as c:
        stats["total_users"]   = c.execute(text("SELECT COUNT(*) FROM user_features_d7")).scalar()
        stats["total_payers"]  = c.execute(text("SELECT COUNT(*) FROM user_features_d7 WHERE target_is_payer=1")).scalar()
        stats["conversion"]    = stats["total_payers"] / max(stats["total_users"], 1)
        stats["avg_ltv_payer"] = float(c.execute(text(
            "SELECT AVG(target_ltv) FROM user_features_d7 WHERE target_is_payer=1"
        )).scalar() or 0)
        stats["n_predictions"]   = c.execute(text("SELECT COUNT(*) FROM predictionlog")).scalar()
        stats["n_agent_actions"] = c.execute(text("SELECT COUNT(*) FROM agentaction")).scalar()
    return stats


@st.cache_data(ttl=60)
def load_cohort_breakdown() -> pd.DataFrame:
    q = """
        SELECT _cohort AS cohort, COUNT(*) AS users,
               SUM(target_is_payer) AS payers,
               AVG(target_is_payer::float) AS payer_rate,
               AVG(CASE WHEN target_is_payer=1 THEN target_ltv END) AS avg_ltv_payers
        FROM user_features_d7 GROUP BY _cohort ORDER BY users DESC
    """
    return pd.read_sql(q, get_engine())


@st.cache_data(ttl=60)
def load_segment_breakdown() -> pd.DataFrame:
    q = """
        SELECT _segment AS segment, COUNT(*) AS n, AVG(target_ltv) AS mean_ltv
        FROM user_features_d7 GROUP BY _segment
        ORDER BY CASE _segment
            WHEN 'whale' THEN 1 WHEN 'dolphin' THEN 2
            WHEN 'minnow' THEN 3 WHEN 'free' THEN 4 END
    """
    return pd.read_sql(q, get_engine())


@st.cache_data(ttl=60)
def load_recent_predictions(limit=10) -> pd.DataFrame:
    q = f"""
        SELECT id, player_id, service, model_version,
               prediction_value, prediction_label, confidence, created_at
        FROM predictionlog ORDER BY id DESC LIMIT {limit}
    """
    return pd.read_sql(q, get_engine())


@st.cache_data(ttl=60)
def load_registered_models() -> pd.DataFrame:
    try:
        r = requests.get(f"{MLFLOW_URL}/api/2.0/mlflow/registered-models/search", timeout=3)
        rows = []
        for m in r.json().get("registered_models", []):
            for v in sorted(m.get("latest_versions", []), key=lambda x: int(x["version"]))[-5:]:
                rows.append({
                    "model":   m["name"],
                    "version": v["version"],
                    "run_id":  v.get("run_id", "")[:12],
                    "status":  v.get("status", ""),
                    "stage":   v.get("current_stage", ""),
                })
        return pd.DataFrame(rows)
    except Exception as ex:
        return pd.DataFrame([{"error": str(ex)}])


# ── Header ───────────────────────────────────────────────────────────────────
st.title("🎮 Player Monetization Intelligence Platform")
st.caption(L(
    "MLOps Capstone · Mobil oyun gelirleştirme için beş servisli ML tahmin platformu",
    "MLOps Capstone · Five-service inference platform for mobile game monetization",
))

with st.expander(L("ℹ️ Bu dashboard nedir?", "ℹ️ What is this dashboard?"), expanded=False):
    st.markdown(L(
        """
        Bu dashboard aşağıdaki 5 modeli ve servisi tek arayüzde birleştiriyor. **Kendi başına
        model çalıştırmıyor** — arkada çalışan **FastAPI servisine** (port 8000) HTTP çağrıları
        atıyor, cevapları ekrana yansıtıyor. Yani gerçek üretimde bir mobil oyun backend'inin
        API'yi nasıl kullanacağını **birebir simüle ediyor**.

        **Bu sayfada ne göreceksin:**
        - Platformun toplam istatistikleri (kullanıcı sayısı, ödeyici oranı, ortalama LTV)
        - Cohort ve segment kırılımı
        - MLflow'a kayıtlı modellerin listesi
        - Son yapılan tahminler (audit trail)

        Soldaki menüden 6 alt sayfaya geçebilirsin.
        """,
        """
        This dashboard combines 5 ML services in one UI. It **doesn't run any model itself** —
        it makes HTTP calls to the **FastAPI service** running on port 8000 and displays
        the responses. This is exactly how a production mobile game backend would consume
        the API.

        **What you'll see here:**
        - Platform-wide statistics (total users, payer rate, average LTV)
        - Cohort and segment breakdown
        - Registered MLflow models
        - Recent predictions (audit trail)

        Navigate to the 6 service pages from the left menu.
        """,
    ))


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(L("Altyapı", "Infrastructure"))
    api_status = check_api_health()
    mlflow_up  = check_mlflow_health()
    st.metric("FastAPI",  "🟢 UP" if api_status["up"] else "🔴 DOWN",
              help=L("Model serving servisi", "Model serving service"))
    st.metric("MLflow",   "🟢 UP" if mlflow_up else "🔴 DOWN",
              help=L("Model registry ve tracking", "Model registry and tracking"))
    if not api_status["up"]:
        st.warning(L(
            "API kapalı! Başlatmak için:\n\n`uv run uvicorn src.main:app --port 8000`",
            "API is down! Start with:\n\n`uv run uvicorn src.main:app --port 8000`",
        ))

    st.divider()
    st.header(L("Adresler", "Endpoints"))
    st.code(f"API    → {API_URL}", language="text")
    st.code(f"MLflow → {MLFLOW_URL}", language="text")
    st.link_button("FastAPI Docs (Swagger)", f"{PUBLIC_API_URL}/docs")
    st.link_button("MLflow UI", PUBLIC_MLFLOW_URL)

    st.divider()
    st.caption(L(
        "Servislere sol menüden geç →",
        "Navigate services from the left menu →",
    ))


# ── Overview KPIs ────────────────────────────────────────────────────────────
st.subheader(L("Genel İstatistikler", "Platform Overview"))
if get_engine() is None:
    st.error(L(
        "Veritabanı yapılandırılmamış — .env dosyasında SQLALCHEMY_DATABASE_URL ayarla.",
        "Database not configured — set SQLALCHEMY_DATABASE_URL in .env.",
    ))
    st.stop()
try:
    stats = load_platform_stats()
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric(L("Toplam Kullanıcı", "Total Users"),   f"{stats['total_users']:,}")
    c2.metric(L("Ödeyici", "Payers"),
              f"{stats['total_payers']:,}",
              f"{stats['conversion']*100:.1f}% " + L("dönüşüm", "conversion"))
    c3.metric(L("Ort. Ödeyici LTV", "Avg Payer LTV"), f"${stats['avg_ltv_payer']:.2f}")
    c4.metric(L("Yapılan Tahmin", "Predictions Served"),
              f"{stats['n_predictions']:,}",
              help=L("Şimdiye kadar servis edilen tahmin sayısı",
                     "Predictions served to date"))
    c5.metric(L("Kararlar", "Decisions Logged"),
              f"{stats['n_agent_actions']:,}",
              help=L("Decision Engine tarafından verilen kararlar",
                     "Decisions logged by the Decision Engine"))
except Exception as ex:
    st.error(L("Veritabanına ulaşılamıyor", "Database unreachable") + f": {ex}")
    st.stop()


st.divider()


# ── Cohort & Segment Overview ────────────────────────────────────────────────
col_left, col_right = st.columns(2)

with col_left:
    st.subheader(L("Kohortlar", "Users by Cohort"))
    st.caption(L(
        "Her kullanıcı bir cohort'a atanmış: real (gerçek kullanıcılar) ya da 3 sentetik cohort. "
        "Renk = ödeyici oranı.",
        "Each user belongs to one cohort: real, or one of 3 synthetic cohorts. "
        "Color intensity = payer rate.",
    ))
    cohort_df = load_cohort_breakdown()
    fig = px.bar(
        cohort_df, x="cohort", y="users",
        color="payer_rate", color_continuous_scale="Blues",
        text="users",
        hover_data={"payer_rate": ":.2%", "avg_ltv_payers": ":.2f"},
        labels={"cohort": L("Cohort", "Cohort"),
                "users": L("Kullanıcı", "Users"),
                "payer_rate": L("Ödeyici oranı", "Payer rate")},
    )
    fig.update_traces(textposition="outside")
    fig.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col_right:
    st.subheader(L("Segmentler", "Users by Segment"))
    st.caption(L(
        "Ödeme davranışına göre segment: whale ($25+), dolphin ($10-25), minnow ($1-10), "
        "free (hiç ödemeyen).",
        "Segment by spend behavior: whale ($25+), dolphin ($10-25), minnow ($1-10), "
        "free (never paid).",
    ))
    seg_df = load_segment_breakdown()
    fig2 = px.pie(
        seg_df, values="n", names="segment", hole=0.4,
        color_discrete_map={
            "whale":   "#1E395E",
            "dolphin": "#2D74B5",
            "minnow":  "#5B9BD5",
            "free":    "#BFBFBF",
        },
    )
    fig2.update_traces(textposition="outside", textinfo="label+percent")
    fig2.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig2, use_container_width=True)


# ── Registered Models ────────────────────────────────────────────────────────
st.divider()
st.subheader(L("🏛 Kayıtlı Modeller (MLflow)", "🏛 Registered Models (MLflow)"))
st.caption(L(
    "MLflow'a her train çalışmasında yeni bir model versiyonu eklenir. "
    "Serving her zaman en yeni versiyonu kullanır. Aşağıda her model için son 5 versiyon.",
    "MLflow adds a new version on every training run. Serving always uses the newest. "
    "Below are the last 5 versions per model.",
))
models_df = load_registered_models()
if "error" in models_df.columns:
    st.warning(L("MLflow sorgusu başarısız", "MLflow query failed") + f": {models_df.iloc[0]['error']}")
else:
    st.dataframe(models_df, use_container_width=True, hide_index=True)
    st.caption(L(
        "🔧 Serving `saved_models/` içindeki en yeni joblib'i alır. Bkz. `src/ml/inference.py`",
        "🔧 Serving picks the newest joblib in `saved_models/`. See `src/ml/inference.py`",
    ))


# ── Recent Predictions ──────────────────────────────────────────────────────
st.divider()
st.subheader(L("📝 Son Tahminler (Audit Trail)", "📝 Recent Predictions (Audit Trail)"))
st.caption(L(
    "Her `/propensity/predict` çağrısı `predictionlog` tablosuna kayıtlanır. "
    "Bu bir MLOps standardı — model drift analizi ve hata çözümü için gerekli.",
    "Every `/propensity/predict` call is logged to `predictionlog`. "
    "Standard MLOps practice — required for drift analysis and troubleshooting.",
))
recent = load_recent_predictions(limit=10)
if recent.empty:
    st.info(L(
        "Henüz tahmin yapılmadı. Soldaki 'Propensity' sayfasından deneyebilirsin →",
        "No predictions logged yet. Try the Propensity page from the left menu →",
    ))
else:
    st.dataframe(
        recent.assign(
            confidence=lambda d: d["confidence"].apply(lambda v: f"{v:.3f}" if pd.notna(v) else ""),
            prediction_value=lambda d: d["prediction_value"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else ""),
        ),
        use_container_width=True, hide_index=True,
    )


# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
with st.expander(L("ℹ️ Bu dashboard hakkında", "ℹ️ About this dashboard")):
    st.markdown(L(
        """
        Bu dashboard, üretimde bir mobil oyun backend'inin çağıracağı aynı FastAPI servisini
        tüketiyor. Her aksiyon gerçek bir endpoint'e gidiyor, her tahmin Postgres'e kayıtlanıyor —
        hiçbir şey mock değil.

        **Servisler** (soldaki menüden erişim):
        1. **Propensity** — Kalibre edilmiş olasılıklarla iki-kule pLTV
        2. **Decision Engine** — IAP vs Reklam vs Atla, beklenen gelir argmax'ı
        3. **Channel ROI** — Kanal başına CPI, gözlemlenen ROAS, geri ödeme süresi
        4. **Cohort Retention** — D1/D7/D30 proxy retention, endüstri kriterlerine karşı
        5. **Personalized Offer** — Gemini tarafından üretilen teklif metni (fallback template'lerle)
        6. **Model Registry** — MLflow versiyonları, metrikler, artefaktlar

        **Stack**: FastAPI · XGBoost · scikit-learn (CalibratedClassifierCV) · MLflow 3 ·
        PostgreSQL · Docker · SDV sentetik veri · LangChain + Gemini.

        Modellerin ne öngörebildiği ve öngöremediği hakkında dürüst değerlendirme için
        `README.md` → *Limitations & Lessons Learned* bölümüne bakın.
        """,
        """
        This dashboard consumes the same FastAPI service that a production mobile-game
        backend would call. Every action here hits a real endpoint and every prediction is
        logged to Postgres — nothing is mocked.

        **Services** (via left sidebar):
        1. **Propensity** — Two-tower pLTV with calibrated probabilities
        2. **Decision Engine** — IAP vs Ad vs Skip, expected-revenue argmax
        3. **Channel ROI** — Per-channel CPI, observed ROAS, payback estimate
        4. **Cohort Retention** — D1/D7/D30 proxy retention vs industry benchmarks
        5. **Personalized Offer** — Gemini-generated offer copy with fallback templates
        6. **Model Registry** — MLflow versions, metrics, artifacts

        **Stack**: FastAPI · XGBoost · scikit-learn (CalibratedClassifierCV) · MLflow 3 ·
        PostgreSQL · Docker · SDV synthetic data · LangChain + Gemini.

        See `README.md` → *Limitations & Lessons Learned* for the honest audit of what
        the models can and cannot predict.
        """,
    ))
