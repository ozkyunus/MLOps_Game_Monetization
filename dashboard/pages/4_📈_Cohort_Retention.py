"""Cohort Retention page — calls GET /cohort/retention?dim=..."""
from __future__ import annotations

import os

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from _i18n import L, sidebar_lang_toggle
from dotenv import load_dotenv

load_dotenv()
API_URL = os.getenv("API_URL", "http://localhost:8000")

st.set_page_config(page_title="Cohort Retention", page_icon="📈", layout="wide")
sidebar_lang_toggle()

st.title(L("📈 Cohort Retention (Kalıcılık)", "📈 Cohort Retention"))
st.caption(L(
    "Cohort boyutuna göre D1/D7/D30 proxy retention, endüstri kriterleriyle karşılaştırmalı.",
    "D1/D7/D30 proxy retention by cohort dimension, compared with industry benchmarks.",
))

with st.expander(L("ℹ️ Bu sayfa ne yapıyor?", "ℹ️ What does this page do?")):
    st.markdown(L(
        """
        **Retention (kalıcılık)** = install olan kullanıcıların ne kadarının belirli günlerde
        hâlâ oyuna geri döndüğü. Mobil oyun endüstrisinde:

        - **D1 retention** — Install'un ertesi günü geri gelen % (hedef ~34%)
        - **D7 retention** — 7. gün geri gelen % (hedef ~17%)
        - **D30 retention** — 30. gün hâlâ oynayan % (hedef ~10%)

        **⚠️ Bizim durum**: Gerçek daily-activity log'umuz yok. Bu yüzden **proxy** kullanıyoruz:
        - D1 ≈ `sessions_d7 >= 2`
        - D7 ≈ `sessions_d7 >= 5`
        - D30 ≈ `target_is_payer == 1`

        Bu proxy'ler gerçek D1/D7/D30'un **yaklaşımı** — data limitasyonu yüzünden böyle.
        README'de bu bir "known limitation" olarak açıkça belgeli.

        Soldan boyut seç (channel/country/platform) → o boyutta hangi kohortun daha
        "sağlam" olduğunu gör. Benchmark line'lar endüstri ortalamasıyla karşılaştırır.
        """,
        """
        **Retention** = the fraction of installed users still coming back on specific days.

        - **D1 retention** — % returning next day (target ~34%)
        - **D7 retention** — % on day 7 (target ~17%)
        - **D30 retention** — % on day 30 (target ~10%)

        **⚠️ Our data**: No daily-activity logs, so we use **proxies**:
        - D1 ≈ `sessions_d7 >= 2`
        - D7 ≈ `sessions_d7 >= 5`
        - D30 ≈ `target_is_payer == 1`

        Documented explicitly as a known limitation in the README.
        Pick a dimension on the left to compare cohorts.
        """,
    ))


# ── Fetch ────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=60)
def fetch_cohort(dim: str) -> dict:
    r = requests.get(f"{API_URL}/cohort/retention", params={"dim": dim}, timeout=10)
    r.raise_for_status()
    return r.json()


# ── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    dim = st.selectbox(
        L("Boyut", "Dimension"),
        ["channel", "country", "platform"],
    )
    st.caption(L(
        "Kullanıcıları kanala / ülkeye / platforma göre grupla",
        "Group users by acquisition channel / country / platform",
    ))


try:
    data = fetch_cohort(dim)
except Exception as ex:
    st.error(L("API çağrısı başarısız", "API call failed") + f": {ex}")
    st.stop()


overall = data.get("overall") or {}
cohort_rows = data.get("cohorts") or []
if not cohort_rows or any(
    overall.get(k) is None
    for k in ("installs", "d1_retention", "d7_retention", "d30_retention")
):
    st.info(L(
        "Henüz veri yok — veritabanı boş görünüyor. Önce veri pipeline'ını çalıştır.",
        "No data yet — the database appears to be empty. Run the data pipeline first.",
    ))
    st.stop()

cohorts = pd.DataFrame(cohort_rows)
benchmarks = overall.get("benchmarks", {})


# ── Overall ──────────────────────────────────────────────────────────────────
st.subheader(L("Genel (Karma)", "Overall (blended)"))
o1, o2, o3, o4 = st.columns(4)
o1.metric("Installs", f"{overall['installs']:,}")
for label_key, key in [("D1", "d1_retention"), ("D7", "d7_retention"), ("D30", "d30_retention")]:
    col = {"D1": o2, "D7": o3, "D30": o4}[label_key]
    actual = overall[key]
    bench = benchmarks.get(label_key)
    delta_text = (f"{(actual - bench)*100:+.1f}pp " +
                  L(f"vs {bench*100:.0f}% hedef", f"vs {bench*100:.0f}% target")) if bench else None
    col.metric(f"{label_key} retention", f"{actual*100:.1f}%", delta_text)


st.divider()


# ── Per-cohort chart ────────────────────────────────────────────────────────
st.subheader(f"{dim.capitalize()} {L('Bazında', 'Breakdown')}")
st.caption(L(
    "Her grup için D1/D7/D30. Kesikli çizgiler endüstri kriteri.",
    "D1/D7/D30 per group. Dashed lines are industry benchmarks.",
))

fig = go.Figure()
for label_key, key, color in [
    ("D1",  "d1_retention",  "#5B9BD5"),
    ("D7",  "d7_retention",  "#2D74B5"),
    ("D30", "d30_retention", "#1E395E"),
]:
    fig.add_trace(go.Bar(
        name=label_key,
        x=cohorts["cohort"],
        y=cohorts[key] * 100,
        marker_color=color,
        text=cohorts[key].apply(lambda v: f"{v*100:.1f}%"),
        textposition="outside",
    ))
for label_key in ["D1", "D7", "D30"]:
    if label_key in benchmarks:
        fig.add_hline(
            y=benchmarks[label_key] * 100, line_dash="dash", line_color="gray",
            annotation_text=(
                L(f"{label_key} kriter {benchmarks[label_key]*100:.0f}%",
                  f"{label_key} benchmark {benchmarks[label_key]*100:.0f}%")
            ),
            annotation_position="top left",
        )
fig.update_layout(
    barmode="group", height=420,
    yaxis_title=L("Retention %", "Retention %"),
    margin=dict(l=0, r=0, t=10, b=0),
)
st.plotly_chart(fig, use_container_width=True)


# ── Full table ──────────────────────────────────────────────────────────────
st.subheader(L("Detaylar", "Details"))
display_df = cohorts.assign(
    installs=lambda d: d["installs"].apply(lambda v: f"{v:,}"),
    payers=lambda d: d["payers"].apply(lambda v: f"{v:,}"),
    d1_retention=lambda d: d["d1_retention"].apply(lambda v: f"{v*100:.1f}%"),
    d7_retention=lambda d: d["d7_retention"].apply(lambda v: f"{v*100:.1f}%"),
    d30_retention=lambda d: d["d30_retention"].apply(lambda v: f"{v*100:.1f}%"),
    d1_vs_benchmark=lambda d: d["d1_vs_benchmark"].apply(lambda v: f"{v*100:+.1f}pp"),
    d7_vs_benchmark=lambda d: d["d7_vs_benchmark"].apply(lambda v: f"{v*100:+.1f}pp"),
    d30_vs_benchmark=lambda d: d["d30_vs_benchmark"].apply(lambda v: f"{v*100:+.1f}pp"),
    avg_ltv=lambda d: d["avg_ltv"].apply(lambda v: f"${v:.2f}" if pd.notna(v) else "—"),
)
st.dataframe(display_df, use_container_width=True, hide_index=True)


# ── Disclaimer ──────────────────────────────────────────────────────────────
proxy_defs = overall.get("proxy_definitions", {})
with st.expander(L("⚠️ Proxy metrik anlamları (dürüstlük)",
                   "⚠️ Proxy metric semantics (honesty section)")):
    d = data.get("_disclaimer", {})
    if isinstance(d, dict):
        st.write(f"**{L('Metrik tipi', 'Metric type')}:**", d.get("metric_type", "PROXY"))
        st.write(f"**{L('Anlamı', 'Implication')}:**", d.get("implication", "—"))
        st.write(f"**{L('Benchmark üstündeyse', 'If above benchmark')}:**",
                 d.get("if_above_benchmark", "—"))
    if proxy_defs:
        st.subheader(L("Kullanılan proxy tanımları", "Proxy definitions used"))
        for k, v in proxy_defs.items():
            st.write(f"- **{k}** → `{v}`")
