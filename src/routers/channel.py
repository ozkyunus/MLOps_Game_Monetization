"""
/channel router — Channel ROI analytics (Service 3).

GET /channel/roi
    Aggregates per acquisition channel:
      - installs
      - spend (USD)
      - CPI                     = spend / installs
      - conversion_rate         = payers / installs
      - avg_ltv (payers)
      - arpu                    = total_observed_revenue / installs
      - roas_observed           = total_observed_revenue / spend
      - payback_days_estimate   = CPI / (arpu / 30)  — naive linearization

Source tables:
  - user_features_d7   : per-user channel + target_ltv + target_is_payer
  - raw_ad_spend       : campaign-level ad_platform + ad_spend

DISCLAIMER (honest naming, v2):
  Earlier versions used "ROAS-D30" but the underlying LTV is a D7 snapshot
  for real users. We now use "observed" + a disclaimer in the response.
  Real D30 ROAS in production would typically be 1.2-2x the observed value
  due to the long tail of late payers.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter

from src.database import get_engine

router = APIRouter(prefix="/channel", tags=["channel"])

# Lazy shared engine — module import must not require a live database
# (v2 created a private pool here, breaking `import src.main` without env).


def _safe_div(num, denom):
    if denom is None or denom == 0:
        return None
    return float(num / denom)


def _round_or_none(v, digits):
    return None if v is None else round(v, digits)


@router.get("/roi")
def channel_roi():
    users = pd.read_sql(
        """
        SELECT channel,
               COUNT(*)              AS installs,
               SUM(target_is_payer)  AS payers,
               SUM(target_ltv)       AS revenue_observed,
               AVG(CASE WHEN target_is_payer = 1 THEN target_ltv END) AS avg_ltv_payers
        FROM user_features_d7
        WHERE channel IS NOT NULL
        GROUP BY channel
        """,
        get_engine(),
    )
    spend = pd.read_sql(
        "SELECT ad_platform AS channel, SUM(ad_spend) AS spend FROM raw_ad_spend GROUP BY ad_platform",
        get_engine(),
    )

    df = users.merge(spend, on="channel", how="left")
    df["spend"] = df["spend"].fillna(0.0)
    df["payers"] = df["payers"].fillna(0).astype(int)
    df["revenue_observed"] = df["revenue_observed"].fillna(0.0)

    rows = []
    for _, r in df.sort_values("revenue_observed", ascending=False).iterrows():
        installs = int(r["installs"])
        spend_v  = float(r["spend"])
        payers   = int(r["payers"])
        revenue  = float(r["revenue_observed"])

        cpi      = _safe_div(spend_v, installs)
        conv     = _safe_div(payers, installs)
        arpu     = _safe_div(revenue, installs)
        roas     = _safe_div(revenue, spend_v)
        payback  = None
        if cpi is not None and arpu is not None and arpu > 0:
            payback = cpi / (arpu / 30.0)

        rows.append({
            "channel":               r["channel"],
            "installs":              installs,
            "payers":                payers,
            "spend_usd":             round(spend_v, 2),
            "revenue_observed_usd":  round(revenue, 2),
            "cpi_usd":               _round_or_none(cpi, 2),
            "conversion_rate":       _round_or_none(conv, 4),
            "avg_ltv_payers":        _round_or_none(
                float(r["avg_ltv_payers"]) if pd.notna(r["avg_ltv_payers"]) else None, 2
            ),
            "arpu_usd":              _round_or_none(arpu, 3),
            "roas_observed":         _round_or_none(roas, 2),
            "payback_days_estimate": _round_or_none(payback, 1),
        })

    total_installs = int(df["installs"].sum())
    total_payers   = int(df["payers"].sum())
    total_revenue  = float(df["revenue_observed"].sum())
    total_spend    = float(df["spend"].sum())
    overall = {
        "installs":              total_installs,
        "payers":                total_payers,
        "spend_usd":             round(total_spend, 2),
        "revenue_observed_usd":  round(total_revenue, 2),
        "conversion_rate":       _round_or_none(_safe_div(total_payers, total_installs), 4),
        "blended_roas_observed": _round_or_none(_safe_div(total_revenue, total_spend), 2),
        "blended_cpi":           _round_or_none(_safe_div(total_spend, total_installs), 2),
    }

    return {
        "overall":    overall,
        "by_channel": rows,
        "_disclaimer": {
            "metric_period": (
                "Revenue/ROAS reflect OBSERVED LTV at the data snapshot "
                "(roughly D7 for real backbone, D30 for synthetic users). "
                "This is NOT a true D30 measurement. Production D30 ROAS "
                "would typically be 1.2-2× higher due to the late-payer tail."
            ),
            "payback_method": (
                "Naive: CPI / (ARPU / 30). Assumes uniform revenue arrival, "
                "which understates payback for whale-heavy cohorts."
            ),
        },
    }
