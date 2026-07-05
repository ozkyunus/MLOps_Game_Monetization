"""Drift detection — compares live prediction traffic against the training
reference and writes results to the `driftlog` table.

    uv run python -m src.drift.check [--days 7]

Runs as a K8s CronJob in production (k8s/drift-cronjob.yaml) — the missing
piece the course's week-9 module built standalone; here it's wired into the
platform's own tables and surfaced in Grafana via the Postgres datasource.

What is measured (input drift + prediction drift; concept drift needs
delayed ground truth we don't have — documented limitation):

  numeric features  → two-sample Kolmogorov-Smirnov (scipy ks_2samp)
  categoricals      → Population Stability Index (PSI; >0.25 = drift,
                      the banking-industry convention)
  p_payer served    → PSI of the confidence column vs the reference
                      predicted distribution (prediction drift)

Reference = the training table `user_features_d7` (what the model learned
from). Live = `predictionlog.features_snapshot` JSON of the last N days.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from scipy.stats import ks_2samp
from sqlmodel import Session

load_dotenv()

KS_ALPHA = 0.05          # reject "same distribution" below this p-value
PSI_THRESHOLD = 0.25     # industry convention: >0.25 = significant shift
MIN_LIVE_ROWS = 20       # below this, tests are statistically meaningless

NUMERIC_FEATURES = ["_sessions_d7"]              # present in features_snapshot
CATEGORICAL_FEATURES = ["country", "channel"]    # present in features_snapshot


def psi(reference: pd.Series, live: pd.Series, bins: int = 10) -> float:
    """Population Stability Index between two samples (numeric or categorical)."""
    if reference.dtype == object or live.dtype == object:
        categories = sorted(set(reference.unique()) | set(live.unique()))
        ref_pct = reference.value_counts(normalize=True).reindex(categories).fillna(0)
        live_pct = live.value_counts(normalize=True).reindex(categories).fillna(0)
    else:
        edges = np.histogram_bin_edges(reference, bins=bins)
        ref_pct = pd.Series(np.histogram(reference, bins=edges)[0] / max(len(reference), 1))
        live_pct = pd.Series(np.histogram(live, bins=edges)[0] / max(len(live), 1))
    # Avoid log(0) — standard epsilon substitution
    ref_pct = ref_pct.clip(lower=1e-6)
    live_pct = live_pct.clip(lower=1e-6)
    return float(((live_pct - ref_pct) * np.log(live_pct / ref_pct)).sum())


def run_checks(days: int = 7) -> list[dict]:
    from src.database import get_engine
    engine = get_engine()

    reference = pd.read_sql(
        "SELECT _sessions_d7, country, channel FROM user_features_d7", engine)
    live_rows = pd.read_sql(
        f"SELECT confidence, features_snapshot FROM predictionlog "
        f"WHERE created_at > NOW() - INTERVAL '{int(days)} days'", engine)

    results: list[dict] = []
    if len(live_rows) < MIN_LIVE_ROWS:
        print(f"⚠ Only {len(live_rows)} live predictions in the last {days}d "
              f"(need ≥{MIN_LIVE_ROWS}) — skipping tests, logging a no-data marker.")
        results.append({"feature_name": "_insufficient_traffic",
                        "ks_statistic": 0.0, "p_value": 1.0,
                        "drift_detected": False})
        return results

    snapshots = pd.DataFrame([json.loads(s) for s in live_rows["features_snapshot"]])

    for feat in NUMERIC_FEATURES:
        if feat not in snapshots.columns:
            continue
        live = pd.to_numeric(snapshots[feat], errors="coerce").dropna()
        stat, p = ks_2samp(reference[feat], live)
        results.append({"feature_name": feat, "ks_statistic": float(stat),
                        "p_value": float(p), "drift_detected": bool(p < KS_ALPHA)})

    for feat in CATEGORICAL_FEATURES:
        if feat not in snapshots.columns:
            continue
        value = psi(reference[feat].astype(str), snapshots[feat].astype(str))
        results.append({"feature_name": f"{feat} (PSI)", "ks_statistic": value,
                        "p_value": float("nan"),
                        "drift_detected": bool(value > PSI_THRESHOLD)})

    # Prediction drift: has the SERVED p_payer distribution shifted between
    # the first and second half of the window? (no ground truth needed)
    conf = live_rows["confidence"].dropna()
    if len(conf) >= MIN_LIVE_ROWS * 2:
        half = len(conf) // 2
        value = psi(conf.iloc[:half], conf.iloc[half:])
        results.append({"feature_name": "p_payer_served (PSI)",
                        "ks_statistic": value, "p_value": float("nan"),
                        "drift_detected": bool(value > PSI_THRESHOLD)})
    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--days", type=int, default=7)
    args = parser.parse_args()

    results = run_checks(days=args.days)

    from src.database import engine as db_engine
    from src.models import DriftLog
    week = pd.Timestamp.utcnow().isocalendar().week
    with Session(db_engine) as session:
        for r in results:
            session.add(DriftLog(service="propensity_ltv", week_number=int(week),
                                 **{k: (0.0 if isinstance(v, float) and np.isnan(v) else v)
                                    for k, v in r.items()}))
        session.commit()

    print(f"\n── Drift raporu (son {args.days} gün) ──")
    any_drift = False
    for r in results:
        flag = "🚨 DRIFT" if r["drift_detected"] else "✓ stabil"
        any_drift |= r["drift_detected"]
        print(f"  {r['feature_name']:<24} stat={r['ks_statistic']:.4f}  {flag}")
    print(f"\n{'🚨 En az bir metrikte drift var — modele/veriye bak!' if any_drift else '✓ Drift yok.'}")
    print(f"✓ {len(results)} satır driftlog tablosuna yazıldı (week {week}).")


if __name__ == "__main__":
    main()
