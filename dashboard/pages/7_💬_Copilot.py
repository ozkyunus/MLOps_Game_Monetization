"""Monetization Copilot — RAG + tools chat (calls POST /copilot/chat)."""
from __future__ import annotations

import os
import uuid

import requests
import streamlit as st
from _i18n import L, sidebar_lang_toggle
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Copilot", page_icon="💬", layout="wide")
sidebar_lang_toggle()

st.title(L("💬 Monetization Copilot", "💬 Monetization Copilot"))
st.caption(L(
    "RAG + tool'larla çalışan analist asistanı: endüstri benchmark dokümanlarından "
    "kaynak göstererek cevaplar, kendi verini platform API'lerinden çeker.",
    "Analyst copilot with RAG + tools: cites industry benchmark documents and "
    "pulls your own data from the platform APIs.",
))

with st.expander(L("ℹ️ Ne sorabilirim?", "ℹ️ What can I ask?")):
    st.markdown(L(
        """
        - **Benchmark soruları**: "Hybrid-casual için iyi D7 retention nedir?"
        - **Kendi verin**: "TikTok kampanyamın ROAS'ı benchmark'a göre nasıl?"
        - **Tahmin**: "Şu user_id'nin pLTV'si ne?" (user_id yapıştır)
        - **Teklif**: "Whale bir kullanıcıya after_loss bağlamında teklif metni üret"
        - **Platform**: "Real cohort tahminlerine neden temkinli yaklaşmalıyım?"

        Cevaplardaki 📄 rozetleri kaynak dokümanları, 🔧 rozetleri çağrılan
        tool'ları gösterir.
        """,
        """
        - **Benchmarks**: "What is a good D7 retention for hybrid-casual?"
        - **Your data**: "How does my TikTok ROAS compare to industry?"
        - **Predictions**: "What's the pLTV for user_id X?"
        - **Offers**: "Draft an offer for a whale user after a loss"
        - **Platform**: "Why should I be cautious about real-cohort predictions?"

        📄 badges show cited documents, 🔧 badges show tools called.
        """,
    ))

if "copilot_msgs" not in st.session_state:
    st.session_state["copilot_msgs"] = []
if "copilot_conv_id" not in st.session_state:
    st.session_state["copilot_conv_id"] = str(uuid.uuid4())[:8]

# ── History ──────────────────────────────────────────────────────────────────
for msg in st.session_state["copilot_msgs"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption("📄 " + " · ".join(
                f"{s['title']} → {s['section']}" for s in msg["sources"]))
        if msg.get("tools"):
            st.caption("🔧 " + " · ".join(
                f"{t['tool']} ({t['ms']:.0f}ms)" for t in msg["tools"]))

# ── Input ────────────────────────────────────────────────────────────────────
question = st.chat_input(L("Sorunu yaz...", "Ask a question..."))
if question:
    st.session_state["copilot_msgs"].append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(L("Düşünüyor (kaynaklar + tool'lar)...",
                          "Thinking (sources + tools)...")):
            try:
                r = requests.post(
                    f"{API_URL}/copilot/chat",
                    json={"question": question,
                          "conversation_id": st.session_state["copilot_conv_id"]},
                    timeout=90,
                )
                r.raise_for_status()
                data = r.json()
            except Exception as ex:
                st.error(L("API çağrısı başarısız", "API call failed") + f": {ex}")
                st.stop()

        st.markdown(data["answer"])
        if data.get("sources"):
            st.caption("📄 " + " · ".join(
                f"{s['title']} → {s['section']}" for s in data["sources"]))
        if data.get("tool_trace"):
            st.caption("🔧 " + " · ".join(
                f"{t['tool']} ({t['ms']:.0f}ms)" for t in data["tool_trace"]))
        st.caption(
            f"⏱ {data['latency_ms']:.0f}ms · "
            f"🎟 {data['usage']['input_tokens']}→{data['usage']['output_tokens']} tokens · "
            f"prompt {data['prompt_version']}"
        )

    st.session_state["copilot_msgs"].append({
        "role": "assistant",
        "content": data["answer"],
        "sources": data.get("sources", []),
        "tools": data.get("tool_trace", []),
    })
