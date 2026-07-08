"""Model promotion gate — the governance layer between training and serving.

Serving NEVER loads "the newest" model; it loads the version carrying the
MLflow alias **champion**. A freshly trained candidate earns that alias only
by beating (or matching within tolerance) the current champion on the
held-out test metric. A bad retrain therefore lands in the registry as a
recorded experiment — and never reaches production.

    promote_if_better("propensity_model", run_id, "test_auc",
                      higher_is_better=True, tolerance=0.005)
"""
from __future__ import annotations

import mlflow

CHAMPION = "champion"


def promote_if_better(model_name: str, candidate_run_id: str, metric_key: str,
                      higher_is_better: bool = True,
                      tolerance: float = 0.0) -> bool:
    """Move the `champion` alias to the candidate iff it is not worse.

    tolerance = how much the candidate may UNDERPERFORM the champion and
    still be promoted (guards against refusing equivalent models over noise).
    Returns True if the candidate is champion after the call.
    """
    client = mlflow.MlflowClient()

    versions = client.search_model_versions(f"name='{model_name}'")
    candidate = next((v for v in versions if v.run_id == candidate_run_id), None)
    if candidate is None:
        print(f"⚠ promotion: no registered version for run {candidate_run_id[:8]} "
              f"— registration failed upstream, champion unchanged.")
        return False

    cand_metric = client.get_run(candidate_run_id).data.metrics.get(metric_key)
    if cand_metric is None:
        print(f"⚠ promotion: candidate run lacks metric '{metric_key}' — refusing.")
        return False

    try:
        champ = client.get_model_version_by_alias(model_name, CHAMPION)
    except Exception:
        champ = None

    if champ is None:
        client.set_registered_model_alias(model_name, CHAMPION, candidate.version)
        print(f"👑 {model_name} v{candidate.version} is the FIRST champion "
              f"({metric_key}={cand_metric:.4f}).")
        return True

    champ_metric = client.get_run(champ.run_id).data.metrics.get(metric_key)
    if champ_metric is None:
        # Champion has no comparable metric (legacy run) — candidate wins.
        client.set_registered_model_alias(model_name, CHAMPION, candidate.version)
        print(f"👑 {model_name} v{candidate.version} promoted (champion v{champ.version} "
              f"had no '{metric_key}' to compare).")
        return True

    if higher_is_better:
        better = cand_metric >= champ_metric - tolerance
        verdict = f"{cand_metric:.4f} vs champion {champ_metric:.4f} (higher wins, tol {tolerance})"
    else:
        better = cand_metric <= champ_metric + tolerance
        verdict = f"{cand_metric:.4f} vs champion {champ_metric:.4f} (lower wins, tol {tolerance})"

    if better:
        client.set_registered_model_alias(model_name, CHAMPION, candidate.version)
        print(f"👑 {model_name} v{candidate.version} PROMOTED — {metric_key}: {verdict}")
        return True

    print(f"🛑 {model_name} v{candidate.version} NOT promoted — {metric_key}: {verdict}. "
          f"Champion stays v{champ.version}; candidate remains in the registry "
          f"as a recorded experiment.")
    return False
