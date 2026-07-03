"""
/cohort router — Cohort Retention analytics (Service 4).

GET /cohort/retention?dim=channel
    dim ∈ { channel | country | platform }
    Returns per cohort:
      - installs, payers
      - D1 retention proxy : sessions_d7 >= 2
      - D7 retention proxy : sessions_d7 >= 5
      - D30 retention proxy: target_is_payer == 1
      - Delta vs industry benchmark (D1=34%, D7=17%, D30=10%)

Used by marketing to spot under-performing cohorts vs benchmarks.

Note on proxies: we have D7 behavior snapshots, not daily activity logs.
The proxies are conservative thresholds calibrated against the population
distribution + the benchmark targets in benchmarks.py.
"""
from __future__ import annotations

import pandas as pd
from fastapi import APIRouter, HTTPException, Query

from src.database import get_engine
from src.synthetic import benchmarks as B

router = APIRouter(prefix="/cohort", tags=["cohort"])

# Lazy shared engine — module import must not require a live database
# (v2 created a private pool here, breaking `import src.main` without env).

ALLOWED_DIMS = {"channel", "country", "platform"}

# Proxy thresholds (see module docstring)
D1_THRESHOLD = 2   # sessions_d7 >= 2
D7_THRESHOLD = 5   # sessions_d7 >= 5


def _round_or_none(v, digits=4):
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return None
    return round(float(v), digits)


@router.get("/retention")
def cohort_retention(
    dim: str = Query("channel", description="channel | country | platform"),
    min_installs: int = Query(50, ge=1, description="filter out tiny cohorts"),
):
    if dim not in ALLOWED_DIMS:
        raise HTTPException(
            status_code=422,
            detail=f"dim must be one of {sorted(ALLOWED_DIMS)}",
        )

    q = f"""
        SELECT
          {dim} AS cohort,
          COUNT(*)                                       AS installs,
          SUM(target_is_payer)                           AS payers,
          AVG(CASE WHEN _sessions_d7 >= {D1_THRESHOLD} THEN 1.0 ELSE 0 END) AS d1_retention,
          AVG(CASE WHEN _sessions_d7 >= {D7_THRESHOLD} THEN 1.0 ELSE 0 END) AS d7_retention,
          AVG(target_is_payer::float)                    AS d30_retention,
          AVG(_sessions_d7::float)                       AS avg_sessions_d7,
          AVG(target_ltv::float)                         AS avg_ltv
        FROM user_features_d7
        WHERE {dim} IS NOT NULL
        GROUP BY {dim}
        HAVING COUNT(*) >= {min_installs}
        ORDER BY installs DESC
    """
    df = pd.read_sql(q, get_engine())

    bench = B.RETENTION_TARGETS  # {'D1': 0.34, 'D7': 0.17, 'D30': 0.10}

    rows = []
    for _, r in df.iterrows():
        d1, d7, d30 = float(r["d1_retention"]), float(r["d7_retention"]), float(r["d30_retention"])
        rows.append({
            "cohort":           r["cohort"],
            "installs":         int(r["installs"]),
            "payers":           int(r["payers"]),
            "d1_retention":     _round_or_none(d1),
            "d7_retention":     _round_or_none(d7),
            "d30_retention":    _round_or_none(d30),
            "d1_vs_benchmark":  _round_or_none(d1 - bench["D1"]),
            "d7_vs_benchmark":  _round_or_none(d7 - bench["D7"]),
            "d30_vs_benchmark": _round_or_none(d30 - bench["D30"]),
            "avg_sessions_d7":  _round_or_none(r["avg_sessions_d7"], 2),
            "avg_ltv":          _round_or_none(r["avg_ltv"], 2),
            "verdict":          _verdict(d1, d7, d30, bench),
        })

    # Overall blended
    overall_q = f"""
        SELECT
          COUNT(*)                                                          AS installs,
          AVG(CASE WHEN _sessions_d7 >= {D1_THRESHOLD} THEN 1.0 ELSE 0 END) AS d1,
          AVG(CASE WHEN _sessions_d7 >= {D7_THRESHOLD} THEN 1.0 ELSE 0 END) AS d7,
          AVG(target_is_payer::float)                                       AS d30
        FROM user_features_d7
    """
    o = pd.read_sql(overall_q, get_engine()).iloc[0]
    overall = {
        "installs":           int(o["installs"]),
        "d1_retention":       _round_or_none(o["d1"]),
        "d7_retention":       _round_or_none(o["d7"]),
        "d30_retention":      _round_or_none(o["d30"]),
        "benchmarks":         {k: float(v) for k, v in bench.items()},
        "proxy_definitions": {
            "D1":  f"sessions_d7 >= {D1_THRESHOLD}",
            "D7":  f"sessions_d7 >= {D7_THRESHOLD}",
            "D30": "target_is_payer == 1",
        },
    }

    return {
        "dim":      dim,
        "overall":  overall,
        "cohorts":  rows,
        "_disclaimer": {
            "metric_type": "PROXY — derived from D7 session aggregates, NOT daily event logs",
            "implication": (
                "Real production retention requires event-level data "
                "(user_active_day or similar). Our proxies upper-bound "
                "early retention because 'D7 has 2+ sessions' captures "
                "broader engagement than 'D1 returning users'. D30 proxy "
                "uses payer status, which is a different concept altogether."
            ),
            "if_above_benchmark": (
                "Likely an artifact of (a) survivorship in the real backbone "
                "(only engaged-enough users got logged) and (b) the synthetic "
                "augmentation being calibrated to industry benchmarks at the "
                "engagement level, not retention level."
            ),
        },
    }


def _verdict(d1: float, d7: float, d30: float, bench: dict) -> str:
    """Quick narrative summary of cohort health vs benchmarks."""
    above = sum([d1 >= bench["D1"], d7 >= bench["D7"], d30 >= bench["D30"]])
    if above == 3:
        return "healthy — all metrics ≥ benchmark"
    if above == 2:
        return "mixed — 2/3 metrics on target"
    if above == 1:
        return "at risk — only 1/3 metrics on target"
    return "underperforming — all metrics below benchmark"
