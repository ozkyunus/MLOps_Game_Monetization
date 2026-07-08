"""Continuous Training — the loop that acts on what drift detection finds.

    uv run python -m src.retrain.run [--force]

Runs weekly as a K8s CronJob (k8s/retrain-cronjob.yaml). Logic:

  1. Read the LATEST drift-check batch from `driftlog`.
  2. No drift (and no --force)  → exit 0, nothing to do.
  3. Drift detected             → retrain both models against the MLflow
     tracking server. Each training script ends with the PROMOTION GATE
     (src/ml/promotion.py): the new model takes the `champion` alias only
     if it beats the incumbent on the held-out metric. A degraded retrain
     stays in the registry as a recorded experiment — serving is safe.
  4. On promotion-eligible completion, best-effort POST to the API's
     /admin/reload-models so pods pick up the new champion without a
     restart.

This closes the MLOps loop: monitor → detect → retrain → gate → serve.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import urllib.request

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

RELOAD_URL = "http://api:8000/admin/reload-models"


def latest_drift_batch() -> pd.DataFrame:
    from src.database import get_engine
    df = pd.read_sql(
        "SELECT feature_name, ks_statistic, drift_detected, created_at "
        "FROM driftlog ORDER BY created_at DESC LIMIT 20", get_engine())
    if df.empty:
        return df
    newest = df["created_at"].max()
    # one batch = rows written by the same check run (sub-second apart)
    return df[(newest - df["created_at"]).dt.total_seconds() < 60]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true",
                        help="retrain regardless of drift verdict")
    args = parser.parse_args()

    batch = latest_drift_batch()
    if batch.empty and not args.force:
        print("driftlog is empty — run the drift check first. Nothing to do.")
        return

    drifted = batch[batch["drift_detected"]]["feature_name"].tolist() if not batch.empty else []
    if not drifted and not args.force:
        print(f"✓ Latest drift batch ({len(batch)} metrics) is clean — no retraining needed.")
        return

    reason = "--force" if args.force and not drifted else f"drift in: {', '.join(drifted)}"
    print(f"🔁 Retraining triggered ({reason})\n")

    for module in ("src.ml.train_propensity", "src.ml.train_ltv"):
        print(f"── {module} ──")
        result = subprocess.run([sys.executable, "-m", module], check=False)
        if result.returncode != 0:
            print(f"❌ {module} failed (rc={result.returncode}) — aborting; "
                  f"champion aliases untouched, serving unaffected.")
            sys.exit(1)

    # Ask serving to refresh its model cache (safe: pods pull the champion
    # alias, which only moved if the promotion gate approved).
    try:
        req = urllib.request.Request(RELOAD_URL, method="POST")
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"✓ API model cache reloaded ({resp.status}).")
    except Exception as exc:
        print(f"⚠ API reload skipped ({exc}) — pods refresh on next restart.")

    print("\n✓ Continuous-training cycle complete. Promotion verdicts above "
          "decide what serving actually loads.")


if __name__ == "__main__":
    main()
