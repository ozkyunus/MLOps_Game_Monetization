"""Channel ROI page — calls GET /channel/roi."""
from __future__ import annotations

import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st
from _i18n import L, sidebar_lang_toggle
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Channel ROI", page_icon="💰", layout="wide")
sidebar_lang_toggle()

st.title(L("💰 Kanal ROI Analitiği", "💰 Channel ROI Analytics"))
st.caption(L(
    "Edinim kanalına göre CPI, conversion, gözlemlenen ROAS ve geri ödeme süresi tahmini. "
    "Metrik periyodu 'observed' (real için D7, synth için D30).",
    "Per-acquisition-channel CPI, conversion, observed ROAS, and naive payback estimate. "
    "Metric period is 'observed' (D7 for real backbone, D30 for synth).",
))

with st.expander(L("ℹ️ Bu sayfa ne yapıyor?", "ℹ️ What does this page do?")):
    st.markdown(L(
        """
        Kullanıcıların hangi kanaldan geldiğini (Organic / Facebook Ads / Google Ads /
        TikTok Ads) ve her kanalın **kar getirip getirmediğini** gösterir.

        **Anahtar metrikler:**
        - **CPI** (Cost Per Install) — Bir install'un maliyeti = harcama / install
        - **Conversion** — Ödeyici olma oranı
        - **ROAS** (Return on Ad Spend) — Getiri / harcama. 1.0 = başabaş, >1.0 = karda
        - **Payback days** — Yatırımın geri gelmesi kaç günde (naive linear tahmin)

        **Bu proje ne demonstre ediyor**: `CHANNEL_LTV_MULTIPLIER` sabiti (bkz.
        `benchmarks.py`) — TikTok users %60 LTV baseline'da, Organic %100'de.
        Endüstri gözleminden hareketle sentetik veriye kanal-bazlı LTV farkı zerkedildi.
        """,
        """
        Shows where users come from and whether each channel is profitable.

        **Key metrics:**
        - **CPI** — Cost per install
        - **Conversion** — Payer conversion rate
        - **ROAS** — Return on ad spend (1.0 = breakeven, >1.0 = profit)
        - **Payback days** — Time to recoup investment (naive linear estimate)

        **What this demonstrates**: `CHANNEL_LTV_MULTIPLIER` in `benchmarks.py` seeds
        channel-specific LTV variance into synthetic data — TikTok users get 0.6× baseline
        LTV, Organic gets 1.0×. Reflects real-world channel-quality observations.
        """,
    ))


@st.cache_data(ttl=60)
def fetch_roi() -> dict:
    r = requests.get(f"{API_URL}/channel/roi", timeout=10)
    r.raise_for_status()
    return r.json()


try:
    data = fetch_roi()
except Exception as ex:
    st.error(f"API call failed: {ex}")
    st.stop()


overall = data["overall"]
by_channel = pd.DataFrame(data["by_channel"])


# ── Overall ──────────────────────────────────────────────────────────────────
st.subheader(L("Genel", "Overall"))
o1, o2, o3, o4, o5 = st.columns(5)
o1.metric("Installs",                                f"{overall['installs']:,}")
o2.metric(L("Ödeyici", "Payers"),                    f"{overall['payers']:,}")
o3.metric(L("Toplam Harcama", "Total Spend"),        f"${overall['spend_usd']:,.0f}")
o4.metric(L("Gözlemlenen Gelir", "Observed Revenue"),f"${overall['revenue_observed_usd']:,.0f}")
o5.metric("Blended ROAS", f"{overall['blended_roas_observed']:.2f}",
          f"CPI ${overall['blended_cpi']:.2f}")


st.divider()

# ── Table ───────────────────────────────────────────────────────────────────
st.subheader(L("Kanal Başına", "Per Channel"))
st.dataframe(
    by_channel.assign(
        spend_usd=lambda d: d["spend_usd"].apply(lambda v: f"${v:,.0f}"),
        revenue_observed_usd=lambda d: d["revenue_observed_usd"].apply(lambda v: f"${v:,.0f}"),
        cpi_usd=lambda d: d["cpi_usd"].apply(lambda v: f"${v:.2f}" if v is not None else "—"),
        avg_ltv_payers=lambda d: d["avg_ltv_payers"].apply(lambda v: f"${v:.2f}" if v is not None else "—"),
        conversion_rate=lambda d: d["conversion_rate"].apply(lambda v: f"{v*100:.2f}%"),
        arpu_usd=lambda d: d["arpu_usd"].apply(lambda v: f"${v:.3f}" if v is not None else "—"),
        roas_observed=lambda d: d["roas_observed"].apply(lambda v: f"{v:.2f}" if v is not None else "—"),
        payback_days_estimate=lambda d: d["payback_days_estimate"].apply(
            lambda v: f"{v:.1f}" if v is not None else "—"),
    ),
    use_container_width=True, hide_index=True,
)


# ── Charts ──────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader(L("Installs vs Gelir", "Installs vs Revenue"))
    df_sorted = by_channel.sort_values("revenue_observed_usd", ascending=True)
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=df_sorted["channel"], x=df_sorted["installs"],
        name="Installs", marker_color="#BFBFBF", orientation="h",
    ))
    fig.add_trace(go.Bar(
        y=df_sorted["channel"], x=df_sorted["revenue_observed_usd"],
        name=L("Gelir ($)", "Revenue ($)"), marker_color="#1E395E", orientation="h",
    ))
    fig.update_layout(barmode="group", height=340, margin=dict(l=0, r=0, t=10, b=0))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader(L("ROAS (Kanal Başına)", "ROAS by Channel"))
    df_r = by_channel.copy()
    df_r["roas_display"] = df_r["roas_observed"].fillna(0)
    df_r = df_r.sort_values("roas_display", ascending=True)
    fig2 = px.bar(
        df_r, x="roas_display", y="channel", orientation="h",
        text=df_r["roas_display"].apply(lambda v: f"{v:.2f}"),
        color="roas_display", color_continuous_scale="RdYlGn",
        labels={"roas_display": L("ROAS gözlemlenen", "ROAS observed"), "channel": ""},
    )
    fig2.add_vline(x=1.0, line_dash="dash", line_color="black",
                   annotation_text=L("ROAS=1 (başabaş)", "ROAS=1 breakeven"))
    fig2.update_traces(textposition="outside")
    fig2.update_layout(height=340, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig2, use_container_width=True)


col3, col4 = st.columns(2)

with col3:
    st.subheader(L("Geri Ödeme Süresi", "Payback Days (Estimate)"))
    df_p = by_channel.copy()
    df_p["payback"] = df_p["payback_days_estimate"].fillna(0)
    df_p = df_p[df_p["payback"] > 0].sort_values("payback")
    if df_p.empty:
        st.info(L("Payback hesaplanabilir paid kanal yok.", "No paid channels with computable payback."))
    else:
        fig3 = px.bar(
            df_p, x="channel", y="payback",
            text=df_p["payback"].apply(lambda v: f"{v:.0f}d"),
            color="payback", color_continuous_scale="RdYlGn_r",
            labels={"payback": L("Payback günü", "Days to payback"), "channel": ""},
        )
        fig3.update_traces(textposition="outside")
        fig3.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
        st.plotly_chart(fig3, use_container_width=True)

with col4:
    st.subheader(L("Conversion Oranı", "Conversion Rate"))
    df_c = by_channel.copy().sort_values("conversion_rate", ascending=True)
    fig4 = px.bar(
        df_c, x="conversion_rate", y="channel", orientation="h",
        text=df_c["conversion_rate"].apply(lambda v: f"{v*100:.1f}%"),
        color="conversion_rate", color_continuous_scale="Blues",
        labels={"conversion_rate": L("Conversion oranı", "Conversion rate"), "channel": ""},
    )
    fig4.update_traces(textposition="outside")
    fig4.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0), coloraxis_showscale=False)
    st.plotly_chart(fig4, use_container_width=True)


with st.expander(L("⚠️ Disclaimer (dürüstlük bölümü)", "⚠️ Disclaimer (honesty section)")):
    d = data.get("_disclaimer", {})
    if isinstance(d, dict):
        st.write(f"**{L('Metrik periyodu', 'Metric period')}:**", d.get("metric_period", "—"))
        st.write(f"**{L('Payback metodu', 'Payback method')}:**", d.get("payback_method", "—"))
    else:
        st.write(d or "—")
