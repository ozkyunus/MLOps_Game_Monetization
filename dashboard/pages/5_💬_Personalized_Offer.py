"""Personalized Offer page — calls POST /personalized/offer."""
from __future__ import annotations

import os

import requests
import streamlit as st
from _data import load_sample_users, require_db
from _i18n import L, sidebar_lang_toggle
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Personalized Offer", page_icon="💬", layout="wide")
sidebar_lang_toggle()

st.title(L("💬 Kişiselleştirilmiş Teklif", "💬 Personalized Offer Copy"))
st.caption(L(
    "LLM (Gemini) ile üretilmiş teklif metni + deterministik fallback template'ler. "
    "Kullanıcıya özel title / body / CTA.",
    "LLM-generated offer copy (Gemini) with deterministic fallback templates. "
    "Per-user title / body / CTA.",
))

with st.expander(L("ℹ️ Bu sayfa ne yapıyor?", "ℹ️ What does this page do?")):
    st.markdown(L(
        """
        Bir mobil oyunda kullanıcıya IAP teklifi (ör: "Premium Chest") gösterileceğini
        varsayalım. Herkese aynı metni göstermek yerine, **kullanıcının segmentine +
        bağlamına özel** bir metin üretebilir miyiz?

        **Pipeline**:
        1. Kullanıcının pLTV'sini hesapla
        2. Karar motoruyla aksiyon belirle
        3. Eğer IAP çıktıysa, segmentine + bağlamına uygun LLM prompt hazırla
        4. Gemini'yi çağır → structured JSON (title + body + CTA) al
        5. Kullanıcıya göster

        **Fallback template'ler**: GOOGLE_API_KEY yoksa, 12 el-yazılı template'ten uygun
        olanı seçilir. Böylece sistem her zaman çalışır.

        **copy_source badge**:
        - 🤖 `llm` — Gerçek Gemini çağrısı
        - 📝 `fallback` — Template kullanıldı
        - 📋 `cache` — Aynı (segment, action, context) daha önce hesaplandı
        """,
        """
        For an in-game IAP prompt (e.g. "Premium Chest"), instead of showing the same text
        to everyone, this generates copy tailored to the user's segment + context.

        **Pipeline**:
        1. Compute the user's pLTV
        2. Pick an action via the decision engine
        3. If it's IAP, build an LLM prompt with segment + context
        4. Call Gemini → structured JSON (title + body + CTA)
        5. Render for the user

        **Fallback templates**: Without `GOOGLE_API_KEY`, one of 12 hand-written templates
        is served. System always works.

        **copy_source badge**:
        - 🤖 `llm` — Real Gemini call
        - 📝 `fallback` — Template used
        - 📋 `cache` — Same (segment, action, context) seen this session
        """,
    ))


engine = require_db()
try:
    users_df = load_sample_users(engine)
except Exception as ex:
    st.error(L("Veritabanı sorgusu başarısız", "Database query failed") + f": {ex}")
    st.stop()


with st.sidebar:
    picked = st.selectbox(
        L("Kullanıcı", "User"),
        users_df["user_id"].tolist(),
        format_func=lambda uid: (
            f"{uid[:8]}… "
            f"({users_df.loc[users_df.user_id==uid, 'segment'].iloc[0]})"
        ),
    )
    context = st.selectbox(
        L("Bağlam", "Context"),
        ["level_complete", "after_loss", "app_open"],
    )
    gen_btn = st.button(
        L("✨ Teklif Üret", "✨ Generate offer"),
        type="primary", use_container_width=True,
    )


if not gen_btn and "last_offer" not in st.session_state:
    st.info(L(
        "Kullanıcı + bağlam seç, sonra **Teklif Üret** butonuna bas.",
        "Pick a user + context, then click **Generate offer**.",
    ))
    st.stop()


if gen_btn:
    try:
        r = requests.post(
            f"{API_URL}/personalized/offer",
            json={"user_id": picked, "context": context},
            timeout=30,
        )
        r.raise_for_status()
        st.session_state["last_offer"] = {
            "inputs": {"user_id": picked, "context": context},
            "response": r.json(),
        }
    except Exception as ex:
        st.error(L("API çağrısı başarısız", "API call failed") + f": {ex}")
        st.stop()


stored = st.session_state["last_offer"]
o = stored["response"]
if stored["inputs"] != {"user_id": picked, "context": context}:
    st.warning(L(
        "Gösterilen sonuç farklı bir seçim için üretildi — güncellemek için butona bas.",
        "Shown result was generated for a different selection — click the button to refresh.",
    ))

c1, c2, c3, c4 = st.columns(4)
c1.metric(L("Değer Segmenti", "Value segment"), o["value_segment"])
c2.metric(L("Seçilen Aksiyon", "Chosen action"),   o["action"])
c3.metric(L("Teklif Adı", "Offer name"), o.get("offer_name") or "—")
c4.metric(L("Fiyat", "Price"),
          f"${o['offer_price_usd']:.2f}" if o.get("offer_price_usd") else "—")


st.divider()


copy = o["copy"]
source_badge = {
    "llm":      L("🤖 Gemini LLM",       "🤖 Gemini LLM"),
    "cache":    L("📋 Önbellek",          "📋 Cached"),
    "fallback": L("📝 Fallback template", "📝 Fallback template"),
}.get(o["copy_source"], o["copy_source"])

st.subheader(L("Üretilen Metin", "Generated Copy") + f"   ·   {source_badge}")

with st.container(border=True):
    st.markdown(f"### {copy['title']}")
    st.markdown(f"*{copy['body']}*")
    if copy.get("cta"):
        st.button(copy["cta"], key="_preview_cta", disabled=True, type="primary")


st.divider()
st.subheader(L("Tanılar", "Diagnostics"))
diag = o["diagnostics"]
d1, d2, d3 = st.columns(3)
d1.metric("p_payer", f"{diag['p_payer']:.3f}")
d2.metric("pLTV", f"${diag['pLTV']:.2f}")
d3.metric(L("Beklenen Gelir", "Expected revenue"), f"${o['expected_revenue']:.5f}")

st.caption(f"{L('Model versiyonu', 'Model version')}: `{o['model_version']}`")


with st.expander(L("🔎 Ham JSON cevabı", "🔎 Full response JSON")):
    st.json(o)
