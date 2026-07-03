"""Decision Engine page — calls POST /decide/action."""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from _data import load_sample_users, require_db
from _i18n import L, sidebar_lang_toggle
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Decision Engine", page_icon="🤖", layout="wide")
sidebar_lang_toggle()

st.title(L("🤖 Karar Motoru", "🤖 Decision Engine"))
st.caption(L(
    "Kullanıcı-başına gerçek zamanlı karar: SHOW_IAP · SHOW_AD_REWARDED · "
    "SHOW_AD_INTERSTITIAL · SKIP. Beklenen gelirin argmax'ı + bağlam çarpanları + ad-fatigue penaltısı.",
    "Per-user real-time decision: SHOW_IAP · SHOW_AD_REWARDED · SHOW_AD_INTERSTITIAL · SKIP. "
    "Argmax over expected revenue with context modifiers and ad-fatigue penalty.",
))

with st.expander(L("ℹ️ Bu sayfa ne yapıyor?", "ℹ️ What does this page do?")):
    st.markdown(L(
        """
        Kullanıcı ve bir bağlam (context) seçtiğinde, 4 aksiyon arasından **beklenen
        gelirini en çoklaştıran**ı seçer:

        - **SHOW_IAP** — In-App Purchase teklifi göster (pLTV bazlı fiyat)
        - **SHOW_AD_REWARDED** — Ödüllü reklam göster
        - **SHOW_AD_INTERSTITIAL** — Geçiş reklamı göster
        - **SKIP** — Hiçbir şey gösterme (retention'ı korumak için)

        **Bağlam çarpanları**:
        - `level_complete` — kullanıcı bölüm bitirdi (IAP moodu +%20)
        - `after_loss` — kullanıcı öldü/kaybetti (rewarded ad moodu +%30)
        - `app_open` — sadece uygulama açıldı (nötr)

        **Ad-fatigue penaltısı**: son 7 günde 30'dan fazla reklam gördüyse, reklam
        opsiyonlarının beklenen geliri 0.5x azaltılır.
        """,
        """
        Given a user + context, picks one of 4 actions by argmax over expected revenue:

        - **SHOW_IAP** — In-app purchase offer (pLTV-tiered price)
        - **SHOW_AD_REWARDED** — Rewarded ad
        - **SHOW_AD_INTERSTITIAL** — Interstitial ad
        - **SKIP** — Show nothing (protect retention)

        **Context modifiers**:
        - `level_complete` — user finished a level (IAP boost +20%)
        - `after_loss` — user died/lost (rewarded ad boost +30%)
        - `app_open` — just an app open (neutral)

        **Ad-fatigue**: if the user has seen 30+ ads in the last 7 days, ad actions
        get a 0.5× penalty.
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
    st.header(L("Girdiler", "Inputs"))
    picked = st.selectbox(
        L("Kullanıcı", "User"),
        users_df["user_id"].tolist(),
        format_func=lambda uid: (
            f"{uid[:8]}… "
            f"({users_df.loc[users_df.user_id==uid, 'cohort'].iloc[0]}/"
            f"{users_df.loc[users_df.user_id==uid, 'segment'].iloc[0]})"
        ),
    )
    context = st.selectbox(
        L("Bağlam", "Context"),
        ["level_complete", "after_loss", "app_open"],
    )
    decide_btn = st.button(
        L("🎬 Karar Ver", "🎬 Decide action"),
        type="primary", use_container_width=True,
    )


# ── Main ─────────────────────────────────────────────────────────────────────
if not decide_btn and "last_decision" not in st.session_state:
    st.info(L(
        "Soldan kullanıcı + bağlam seç, sonra **Karar Ver** butonuna bas.",
        "Pick a user + context on the left, then hit **Decide action**.",
    ))
    st.stop()

if decide_btn:
    try:
        r = requests.post(
            f"{API_URL}/decide/action",
            json={"user_id": picked, "context": context},
            timeout=10,
        )
        r.raise_for_status()
        st.session_state["last_decision"] = {
            "inputs": {"user_id": picked, "context": context},
            "response": r.json(),
        }
    except Exception as ex:
        st.error(L("API çağrısı başarısız", "API call failed") + f": {ex}")
        st.stop()

stored = st.session_state["last_decision"]
d = stored["response"]
if stored["inputs"] != {"user_id": picked, "context": context}:
    st.warning(L(
        "Gösterilen sonuç farklı bir seçim için üretildi — güncellemek için butona bas.",
        "Shown result was generated for a different selection — click the button to refresh.",
    ))

action = d["action"]
action_style = {
    "SHOW_IAP":             ("💎", L("IAP teklifi", "IAP offer")),
    "SHOW_AD_REWARDED":     ("🎁", L("Ödüllü reklam", "Rewarded ad")),
    "SHOW_AD_INTERSTITIAL": ("📺", L("Geçiş reklamı", "Interstitial ad")),
    "SKIP":                 ("⏭",  L("Atla (retention koru)", "Skip (protect retention)")),
}
icon, label = action_style.get(action, ("❓", action))
st.subheader(f"{icon} {L('Karar', 'Decision')}: **{label}**")

c1, c2, c3, c4 = st.columns(4)
c1.metric(L("Bağlam", "Context"), d["context"])
c2.metric(L("Beklenen Gelir", "Expected revenue"), f"${d['expected_revenue']:.5f}")
c3.metric(L("Teklif", "Offer"), d.get("offer") or "—")
c4.metric(L("Model versiyonu", "Model version"), f"`{d['model_version']}`")

st.info(d["reasoning"])

st.subheader(L("Tanılar", "Diagnostics"))
diag = d["diagnostics"]
dc1, dc2, dc3, dc4 = st.columns(4)
dc1.metric("p_payer", f"{diag['p_payer']:.3f}",
           help=L("Ödeyici olma olasılığı", "Payer probability"))
dc2.metric("pLTV", f"${diag['pLTV']:.2f}",
           help=L("Tahmin edilen LTV", "Predicted LTV"))
dc3.metric(
    L("Reklam (D7)", "Ad views (D7)"),
    diag["ad_views_d7"],
    help=L("Son 7 günde görülen reklam sayısı", "Ads seen in the last 7 days"),
)
dc4.metric(
    L("Ad-fatigue penaltı", "Ad fatigue penalty"),
    f"{diag['ad_fatigue_penalty']:.2f}",
    L("iyi", "OK") if diag["ad_fatigue_penalty"] == 1.0 else L("penaltı aktif", "penalty active"),
    delta_color="off" if diag["ad_fatigue_penalty"] == 1.0 else "inverse",
)


st.subheader(L("Alternatif Aksiyonlar", "Alternative Actions"))
st.caption(L(
    "Her aksiyonun beklenen geliri. Koyu = seçilen aksiyon.",
    "Expected revenue per action. Dark = chosen action.",
))

alts = pd.DataFrame(d["alternatives"] + [{"action": action, "expected_revenue": d["expected_revenue"]}])
alts["chosen"] = alts["action"] == action
alts = alts.sort_values("expected_revenue", ascending=True)

fig = px.bar(
    alts, x="expected_revenue", y="action",
    color="chosen",
    color_discrete_map={True: "#1E395E", False: "#BFBFBF"},
    orientation="h",
    text=alts["expected_revenue"].apply(lambda v: f"${v:.5f}"),
    labels={"expected_revenue": L("Beklenen gelir ($)", "Expected revenue ($)"), "action": ""},
)
fig.update_traces(textposition="outside")
fig.update_layout(height=280, showlegend=False, margin=dict(l=0, r=0, t=10, b=0))
st.plotly_chart(fig, use_container_width=True)

with st.expander(L("🔎 Ham JSON cevabı", "🔎 Full response JSON")):
    st.json(d)
