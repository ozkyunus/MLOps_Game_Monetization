"""Propensity + pLTV page — calls POST /propensity/predict."""
from __future__ import annotations

import os

import plotly.graph_objects as go
import requests
import streamlit as st
from _data import load_sample_users, require_db
from _i18n import L, sidebar_lang_toggle
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Propensity · pLTV", page_icon="🎯", layout="wide")
sidebar_lang_toggle()

st.title(L("🎯 Propensity ve pLTV Tahmini", "🎯 Propensity & pLTV Prediction"))
st.caption(L(
    "İki-kule mimarisi: kalibre edilmiş XGBoost sınıflandırıcı × Huber kayıp LTV regresörü, "
    "cohort-aware non-payer gate ile.",
    "Two-tower architecture: calibrated XGBoost classifier × Huber-loss LTV regressor, "
    "with cohort-aware non-payer gating.",
))

with st.expander(L("ℹ️ Bu sayfa ne yapıyor?", "ℹ️ What does this page do?")):
    st.markdown(L(
        """
        Bir kullanıcı seçtiğinde şu iki soruyu cevaplar:
        1. **P(payer)** — Bu kullanıcı ödeyici mi olacak? (kalibre edilmiş 0-1 olasılık)
        2. **E[LTV | payer]** — Eğer öderse ne kadar öder?

        Sonra bunları çarparak `pLTV = P(payer) × E[LTV|payer]` hesaplar. **Gate mekanizması**:
        P(payer) çok düşükse (real cohort'ta 0.10, synth cohort'ta 0.20 altında), pLTV = $0
        döndürülür — çünkü LTV modeli sadece ödeyicilerle eğitildi, non-payer tahminleri
        güvenilir değil.

        **Ne göreceksin:**
        - Ground truth (gerçek değerler) vs modelin tahmini
        - `pLTV_raw` (gate uygulanmamış) vs `pLTV` (servis edilen değer)
        - Kullanılan model versiyonu
        """,
        """
        For a chosen user, this page answers two questions:
        1. **P(payer)** — Will this user become a payer? (calibrated 0-1 probability)
        2. **E[LTV | payer]** — If they pay, how much?

        Then combines them: `pLTV = P(payer) × E[LTV|payer]`. A **gate** forces pLTV=0
        when P(payer) is too low (real cohort: 0.10, synth: 0.20), because the LTV model
        was trained only on payers and its predictions on non-payers are out-of-distribution.

        **What you'll see:**
        - Ground truth vs the model's prediction
        - `pLTV_raw` (before gate) vs `pLTV` (served value)
        - Which model version was used
        """,
    ))


# ── Data ─────────────────────────────────────────────────────────────────────
engine = require_db()
try:
    users_df = load_sample_users(engine)
except Exception as ex:
    st.error(L("Veritabanı sorgusu başarısız", "Database query failed") + f": {ex}")
    st.stop()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header(L("Kullanıcı Seç", "Pick a user"))
    filter_cohort = st.selectbox(
        L("Cohort filtresi", "Filter by cohort"),
        ["(all)"] + sorted(users_df["cohort"].unique().tolist()),
    )
    filter_segment = st.selectbox(
        L("Segment filtresi", "Filter by segment"),
        ["(all)"] + sorted(users_df["segment"].unique().tolist()),
    )

    filt = users_df.copy()
    if filter_cohort != "(all)":
        filt = filt[filt["cohort"] == filter_cohort]
    if filter_segment != "(all)":
        filt = filt[filt["segment"] == filter_segment]

    if filt.empty:
        st.warning(L("Bu filtrelere uyan kullanıcı yok.", "No users match those filters."))
        picked_user = None
    else:
        picked_user = st.selectbox(
            L(f"Kullanıcı ({len(filt)} adet)", f"User ({len(filt)} available)"),
            filt["user_id"].tolist(),
            format_func=lambda uid: f"{uid[:8]}… ({filt.loc[filt.user_id==uid, 'segment'].iloc[0]})",
        )

    st.divider()
    st.subheader(L("Gelişmiş", "Advanced"))
    manual_gate = st.checkbox(
        L("Gate threshold'u değiştir", "Override gate threshold"),
        value=False,
    )
    gate_value = None
    if manual_gate:
        gate_value = st.slider("Gate threshold", 0.05, 0.50, 0.20, 0.01)
        st.caption(L(
            "Bu değerin altındaki p_payer değerli kullanıcılara pLTV=$0 verilir.",
            "Users below this p_payer get pLTV=0",
        ))

    predict_btn = st.button(
        L("🚀 Tahmin Et", "🚀 Predict"),
        type="primary", use_container_width=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────
if picked_user is None:
    st.info(L(
        "Sol menüden bir kullanıcı seç.",
        "Pick a user from the sidebar to run a prediction.",
    ))
    st.stop()


st.subheader(L("Gerçek Değerler", "Ground Truth"))
st.caption(L(
    "Modelin tahminini karşılaştırmak için, seçtiğin kullanıcının veritabanındaki gerçek değerleri.",
    "The actual values for this user, so you can compare against the model's prediction.",
))
gt = users_df[users_df["user_id"] == picked_user].iloc[0]
gt_col1, gt_col2, gt_col3, gt_col4 = st.columns(4)
gt_col1.metric(L("Kullanıcı", "User"), picked_user[:12] + "…")
gt_col2.metric("Cohort", gt["cohort"])
gt_col3.metric(L("Segment (gerçek)", "Segment (true)"), gt["segment"])
gt_col4.metric(
    L("Gerçek LTV", "True LTV"),
    f"${gt['true_ltv']:.2f}",
    L("ödeyici", "payer") if gt["true_payer"] else L("ödemedi", "non-payer"),
    delta_color="off" if not gt["true_payer"] else "normal",
)

st.divider()

if predict_btn or "last_pred" in st.session_state:
    if predict_btn:
        payload = {"user_id": picked_user}
        if gate_value is not None:
            payload["gate_threshold"] = gate_value
        try:
            r = requests.post(f"{API_URL}/propensity/predict", json=payload, timeout=10)
            r.raise_for_status()
            st.session_state["last_pred"] = {
                "inputs": {"user_id": picked_user},
                "response": r.json(),
            }
        except Exception as ex:
            st.error(L("API çağrısı başarısız", "API call failed") + f": {ex}")
            st.stop()

    stored = st.session_state["last_pred"]
    pred = stored["response"]
    if stored["inputs"] != {"user_id": picked_user}:
        st.warning(L(
            "Gösterilen sonuç farklı bir seçim için üretildi — güncellemek için butona bas.",
            "Shown result was generated for a different selection — click the button to refresh.",
        ))

    st.subheader(L("Model Tahmini", "Model Output"))
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("P(payer)", f"{pred['p_payer']:.3f}",
              help=L("Ödeyici olma olasılığı (kalibre edilmiş)",
                     "Calibrated probability of paying"))
    c2.metric("E[LTV | payer]", f"${pred['expected_ltv_if_payer']:.2f}",
              help=L("Ödeyici olursa beklenen LTV", "Expected LTV if the user pays"))
    c3.metric(
        L("pLTV (servis edilen)", "pLTV (served)"),
        f"${pred['pLTV']:.2f}",
        delta=L("GATED → 0", "GATED → 0") if pred["gate_applied"] else L("servis edildi", "served"),
        delta_color="off" if pred["gate_applied"] else "normal",
        help=L("p_payer × E[LTV|payer], gate uygulandıktan sonra",
               "p_payer × E[LTV|payer], after gating"),
    )
    c4.metric(L("Değer Segmenti", "Value segment"), pred["value_segment"])

    st.subheader(L("Gate Görselleştirmesi", "Gating Visualization"))
    st.caption(L(
        "Sol: gate uygulanmamış ham pLTV. Sağ: gate sonrası servis edilen değer.",
        "Left: raw pLTV. Right: served value after gating.",
    ))

    fig = go.Figure()
    fig.add_trace(go.Bar(
        name=L("pLTV ham (gate'siz)", "pLTV raw (uncapped)"), x=["raw"], y=[pred["pLTV_raw"]],
        marker_color="#2D74B5",
        text=[f"${pred['pLTV_raw']:.2f}"], textposition="outside",
    ))
    fig.add_trace(go.Bar(
        name=L("pLTV (gate sonrası)", "pLTV served (after gate)"), x=["served"], y=[pred["pLTV"]],
        marker_color="#1E395E",
        text=[f"${pred['pLTV']:.2f}"], textposition="outside",
    ))
    fig.add_hline(
        y=0.01, line_dash="dash", line_color="gray",
        annotation_text=L(
            f"Gate: p_payer < {pred['gate_threshold']} → 0",
            f"Gate: p_payer < {pred['gate_threshold']} → 0",
        ),
        annotation_position="top right",
    )
    fig.update_layout(
        height=320, showlegend=True,
        yaxis_title="pLTV ($)",
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader(L("Tanılar", "Diagnostics"))
    d1, d2, d3 = st.columns(3)
    d1.write(f"**{L('Ödeyici tahmini', 'Predicted payer')}**")
    d1.write(
        (L("✅ Evet", "✅ Yes") if pred["is_predicted_payer"] else L("❌ Hayır", "❌ No"))
        + f"  \nF1-optimal threshold = {pred['default_threshold']}"
    )
    d2.write("**Gate**")
    d2.write(
        (L("🔒 Uygulandı (pLTV=0)", "🔒 Applied (pLTV=0)") if pred["gate_applied"]
         else L("🔓 Uygulanmadı", "🔓 Not applied"))
        + f"  \nthreshold = {pred['gate_threshold']}"
    )
    d3.write(f"**{L('Model versiyonu', 'Model version')}**")
    d3.write(f"`{pred['model_version']}`")

    with st.expander(L("🔎 Ham JSON cevabı", "🔎 Full response JSON")):
        st.json(pred)

    st.caption(pred.get("_disclaimer", ""))
