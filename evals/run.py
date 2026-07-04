"""Copilot eval harness — the LLM's equivalent of the model test set.

    uv run python -m evals.run                # retrieval + behavior evals
    uv run python -m evals.run --judge        # + LLM-as-judge faithfulness
    uv run python -m evals.run --mlflow       # log scores to MLflow

Three eval categories (see design spec):
  1. Retrieval — did the expected documents get cited?
  2. Behavior  — right tools called? out-of-scope refused? content present?
  3. Faithfulness (--judge) — second-LLM 1-5 score: is the answer derivable
     from its cited sources? (costly → not run per-PR; nightly/manual)

Needs a live stack: Qdrant (ingested), Postgres, GOOGLE_API_KEY.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

GOLDEN_PATH = Path(__file__).resolve().parent / "golden_set.yaml"
RESULTS_DIR = Path(__file__).resolve().parent / "results"

# Launch thresholds (spec §Success Criteria — tune after first baseline)
THRESHOLDS = {"retrieval_hit_rate": 0.8, "behavior_pass_rate": 0.9,
              "faithfulness_mean": 4.0}


def eval_case(case: dict) -> dict:
    from src.copilot.agent import ask
    result = ask(case["question"])

    cited = {s["doc_id"] for s in result["sources"]}
    tools_used = {t["tool"] for t in result["tool_trace"]}
    answer_low = result["answer"].lower()

    checks = {}
    expected_sources = set(case.get("expected_sources") or [])
    if expected_sources:
        checks["retrieval"] = expected_sources.issubset(cited)

    expected_tools = set(case.get("expected_tools") or [])
    if expected_tools:
        checks["tools"] = expected_tools.issubset(tools_used)

    # must_contain items can be a single phrase OR a list of alternatives
    # (any-of) — e.g. [["4.22", "4,22"]] tolerates locale number formats.
    for phrase in case.get("must_contain") or []:
        alternatives = phrase if isinstance(phrase, list) else [phrase]
        label = "|".join(str(a) for a in alternatives)
        checks[f"contains:{label}"] = any(
            str(a).lower() in answer_low for a in alternatives)
    for phrase in case.get("must_not_contain") or []:
        checks[f"not-contains:{phrase}"] = str(phrase).lower() not in answer_low

    if case.get("expected_behavior") == "refuse":
        checks["refused"] = result["refused"] and not tools_used
    else:
        checks["answered"] = not result["refused"]

    return {
        "id": case["id"],
        "checks": checks,
        "passed": all(checks.values()),
        "retrieval_ok": checks.get("retrieval"),
        "answer": result["answer"],
        "cited": sorted(cited),
        "tools": sorted(tools_used),
        "usage": result["usage"],
        "latency_ms": result["latency_ms"],
    }


def judge_faithfulness(case_result: dict) -> int | None:
    """LLM-as-judge: 1-5, is the answer derivable from its cited sources?"""
    if not case_result["cited"]:
        return None
    from langchain_google_genai import ChatGoogleGenerativeAI

    from src.copilot import vectorstore
    # Re-fetch the cited chunks as judge context
    chunks = vectorstore.search(case_result["answer"], top_k=6)
    context = "\n---\n".join(c["text"] for c in chunks
                             if c["doc_id"] in case_result["cited"])
    judge = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.0)
    verdict = judge.invoke(
        "You are a strict grader. Score 1-5 how fully the ANSWER is supported "
        "by the SOURCES (5 = every claim supported, 1 = fabricated). "
        "Reply with ONLY the integer.\n\n"
        f"SOURCES:\n{context}\n\nANSWER:\n{case_result['answer']}"
    )
    try:
        return max(1, min(5, int(str(verdict.content).strip()[:1])))
    except ValueError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--mlflow", action="store_true")
    parser.add_argument("--only", help="run a single case id")
    args = parser.parse_args()

    cases = yaml.safe_load(GOLDEN_PATH.read_text())
    if args.only:
        cases = [c for c in cases if c["id"] == args.only]
    print(f"Golden set: {len(cases)} cases\n")

    results, t0 = [], time.time()
    for i, case in enumerate(cases):
        if i:
            time.sleep(4)   # stay under the free-tier requests-per-minute cap
        try:
            r = eval_case(case)
        except Exception as exc:
            # One flaky/overloaded API call must not kill the whole run —
            # record the case as errored and keep scoring the rest.
            r = {"id": case["id"], "checks": {"error": False}, "passed": False,
                 "retrieval_ok": None, "answer": f"ERROR: {exc}", "cited": [],
                 "tools": [], "usage": {"input_tokens": 0, "output_tokens": 0},
                 "latency_ms": 0.0, "error": f"{type(exc).__name__}"}
            print(f"  ⚠ {case['id']:<28} ERROR: {type(exc).__name__} — retrying once in 20s")
            time.sleep(20)
            try:
                r = eval_case(case)
            except Exception:
                pass
        if args.judge and not r.get("error"):
            r["faithfulness"] = judge_faithfulness(r)
        results.append(r)
        status = "✓" if r["passed"] else "✗"
        failed = [k for k, v in r["checks"].items() if not v]
        print(f"  {status} {r['id']:<28} "
              f"{'' if r['passed'] else 'FAILED: ' + ', '.join(failed)}")

    # ── Aggregate ────────────────────────────────────────────────────────────
    retrieval_cases = [r for r in results if r["retrieval_ok"] is not None]
    metrics = {
        "behavior_pass_rate": sum(r["passed"] for r in results) / len(results),
        "retrieval_hit_rate": (
            sum(r["retrieval_ok"] for r in retrieval_cases) / len(retrieval_cases)
            if retrieval_cases else None
        ),
        "total_tokens": sum(r["usage"]["input_tokens"] + r["usage"]["output_tokens"]
                            for r in results),
        "mean_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results)),
    }
    scores = [r["faithfulness"] for r in results if r.get("faithfulness")]
    if scores:
        metrics["faithfulness_mean"] = round(sum(scores) / len(scores), 2)

    print("\n── Scores ──")
    for k, v in metrics.items():
        print(f"  {k:<22} = {v}")

    RESULTS_DIR.mkdir(exist_ok=True)
    out = RESULTS_DIR / f"eval_{int(t0)}.json"
    out.write_text(json.dumps({"metrics": metrics, "results": results},
                              ensure_ascii=False, indent=2, default=str))
    print(f"\n✓ Detay: {out}")

    if args.mlflow:
        import mlflow
        mlflow.set_experiment("copilot_evals")
        with mlflow.start_run(run_name="golden_set"):
            mlflow.log_metrics({k: v for k, v in metrics.items()
                                if isinstance(v, (int, float))})
            mlflow.log_artifact(str(out))
        print("✓ MLflow'a loglandı (experiment: copilot_evals)")

    # ── Gate (CI uses the exit code) ─────────────────────────────────────────
    failures = []
    for key, threshold in THRESHOLDS.items():
        value = metrics.get(key)
        if value is not None and value < threshold:
            failures.append(f"{key}={value} < {threshold}")
    if failures:
        print("\n❌ EVAL GATE FAILED: " + "; ".join(failures))
        sys.exit(1)
    print("\n✓ Eval gate passed.")


if __name__ == "__main__":
    main()
