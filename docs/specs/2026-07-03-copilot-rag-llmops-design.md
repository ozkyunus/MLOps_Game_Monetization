# Monetization Copilot — RAG + LLMOps Design

**Date**: 2026-07-03 · **Status**: Approved design, implementation deferred
(user will trigger implementation explicitly; nothing in this spec is built yet)

## Goal

Extend the platform's LLM capability from a single generation endpoint
(`/personalized/offer`) into a **RAG-grounded, tool-calling analyst copilot**
("Monetization Copilot") — and use its rollout as the vehicle for the
remaining infrastructure work (Kubernetes, GitOps, monitoring), with the
project owner hands-on in the infra and evaluation work.

## Decisions Made (with rationale)

1. **Product**: Company-facing analyst copilot (option A) — not a
   player-facing wiki assistant. It answers monetization questions by
   combining a real document corpus with live platform data. Chosen because
   it is the natural continuation of the existing system and exercises RAG +
   agentic tool use + LLMOps in one feature.
2. **Coexistence, not replacement**: `/personalized/offer` stays. The copilot
   is Service 6 (`/copilot/chat`) and *uses the offer service as a tool*
   (`draft_offer_copy`), making the existing endpoint more valuable.
3. **Vector store: Qdrant** (dedicated pod). pgvector would technically
   suffice at this corpus size; Qdrant was chosen deliberately for
   (a) job-market signal — dedicated vector-DB experience is listed
   explicitly in GenAI job ads, (b) one more real service to deploy/monitor
   on K8s, matching the owner's hands-on goals. The README will document the
   "when would pgvector have been enough" tradeoff.
4. **Embeddings: Gemini embedding API** (`gemini-embedding-001`) —
   consistent with the existing Gemini stack, free-tier sufficient, avoids
   baking a ~500MB local model into the image.
5. **Agent style: agentic RAG** — `search_knowledge_base` is one tool among
   several (platform APIs are the others); the agent decides per-question
   what to call. Fixed retrieve-then-answer pipeline rejected: cannot answer
   data questions ("what's MY ROAS?").
6. **No fine-tuning** — RAG only. Model weights are never updated; knowledge
   arrives via prompt-context injection at question time. Quality is managed
   through corpus curation, chunking, retrieval tuning, and prompt
   instructions — measured by the eval harness.

## Architecture

```
Dashboard "💬 Copilot" page (new, 8th page)
        │ POST /copilot/chat
        ▼
FastAPI (existing app, new router)                 knowledge/  (real corpus,
  LangChain agent (Gemini 2.5 Flash)                markdown, versioned in git)
  TOOLS:                                                  │
    • search_knowledge_base ───► Qdrant  ◄── ingest job ──┘
    • get_channel_roi   ─┐                    (chunk → embed → upsert)
    • get_retention     ─┼─► existing service functions
    • predict_pltv      ─┤
    • draft_offer_copy  ─┘
  Response = answer + source citations + tool trace
        │
        ▼
Postgres: copilot_log (question, answer, sources, tool calls,
          input/output tokens, latency, prompt_version, conversation id)
```

### Components

- **Corpus (`knowledge/`)**: 10-20 curated markdown documents with source
  attribution — summaries/excerpts of the industry reports already cited in
  `src/synthetic/benchmarks.py`, App Store / Google Play monetization policy
  summaries, and the project's own README/Limitations (so the copilot can
  explain the platform's own constraints). Quality over quantity; a few
  hundred chunks total (~300-500 tokens each).
- **Ingest (`scripts/ingest_knowledge.py`)**: heading-based chunking →
  Gemini embeddings → Qdrant upsert with per-chunk metadata (file, section,
  source URL). Idempotent; records a corpus hash for versioning. Runs as a
  K8s Job in phase 2.
- **Agent router (`src/routers/copilot.py`)**: `create_agent` pattern
  (familiar from course week 8/11 material); structured response including
  citations and tool trace; guardrails — max tool calls, timeout,
  out-of-scope refusal instruction in the system prompt.
- **Retrieval defaults**: top-k 4-5 chunks (~1.5-2.5k tokens into the
  prompt); thresholds tuned via the eval harness, not guessed.
- **Chat UI**: dashboard page with conversation view, source badges, and
  tool-call chips (consistent with the existing bilingual L() convention).

## LLMOps Layer

- **Golden set (`evals/golden_set.yaml`)**: ~25-30 hand-written questions,
  each declaring expected sources, expected tool calls, must/must-not
  content, or `refuse` behavior. ~10 questions authored by the project owner
  (domain involvement).
- **Three eval categories**:
  1. *Retrieval eval* — expected_sources ⊆ retrieved sources → hit rate
  2. *Behavior eval* — correct tool selection; out-of-scope refusal
     (deterministic asserts)
  3. *Faithfulness eval* — LLM-as-judge (second Gemini call) scores 1-5
     whether the answer is derivable from the cited sources
- **`uv run python -m evals.run`** prints scores, writes JSON results, and
  logs metrics to MLflow (retrieval_hit_rate, faithfulness_mean,
  refuse_accuracy) — LLM quality trends live next to classic model metrics.
- **CI eval gate**: new job runs the deterministic evals (mocked/live-smoke
  split) on PRs touching prompts/agent/ingest; fails below threshold.
  LLM-judge runs nightly/manual, not per-PR (cost).
- **Prompt versioning**: system prompt lives at `prompts/copilot_v1.md`;
  responses and `copilot_log` rows carry `prompt_version` (the LLM analogue
  of PredictionLog's `model_version`).
- **Runtime telemetry**: tokens in/out, end-to-end and retrieval-only
  latency, tool-call counts — logged to `copilot_log` and exposed as custom
  Prometheus metrics.

## Kubernetes + GitOps + Monitoring (follows course week-11 template)

Target: Docker Desktop Kubernetes, namespace `monetization`.

- postgres: Deployment + Secret + PVC + `pg_isready` probes
- **qdrant: StatefulSet + PVC + Service** (native `/metrics`)
- mlflow: Deployment + PVC + Service
- api: Deployment ×2 replicas; liveness `/healthz`, readiness `/readyz`
  (the existing readyz finally gates traffic for real); copilot is a router
  in this app, not a separate pod (shares DB/tool plumbing)
- dashboard: Deployment + Service + Ingress
- ingest: on-demand K8s Job
- ArgoCD Application watching `k8s/` (automated + prune + selfHeal)
- kube-prometheus-stack (Helm); ServiceMonitors for api (via
  prometheus-fastapi-instrumentator, to be added) and qdrant
- CI extension: build+push SHA-tagged image to GHCR on main; manifest-bump
  job commits back with `[skip ci]`; ArgoCD reconciles (course GitOps loop)
- Grafana: **Ops dashboard** (request rate, p95, 5xx, in-flight) +
  **LLM dashboard** (tokens, copilot latency split, tool distribution)
- Registry access: GHCR package public OR imagePullSecret — decided at
  implementation time.

### Phasing (each phase ends in a working system)

1. **Phase 1 — RAG on compose**: Qdrant in docker-compose; corpus + ingest +
   agent + chat UI + eval harness working locally.
2. **Phase 2 — K8s migration**: all manifests, stack live on Docker Desktop
   K8s, kube-prometheus-stack, both Grafana dashboards.
3. **Phase 3 — GitOps**: ArgoCD + CI image-bump + eval gate; record the
   push→sync→rolling-update demo for the README.

## Work Split (owner hands-on)

Owner (assistant guides/reviews): enable Docker Desktop K8s; kubectl
exploration; **write the Qdrant StatefulSet+Service from a skeleton**;
run ArgoCD + kube-prometheus-stack installs; **build the Grafana LLM panel**
(PromQL provided); **author ~10 golden-set questions**; curate 3-5 knowledge
documents; break-and-fix drills (delete pod → self-heal; bad tag → ArgoCD
OutOfSync).

Assistant: scaffolding, agent/router/eval/metrics code, remaining manifests,
ArgoCD Application, CI jobs, debugging support.

## Evidence Plan (how K8s skill is demonstrated in the repo)

README gains a section with: manifest tour, GitOps loop capture (commit →
CI → ArgoCD sync → `kubectl get pods -w` rolling update), self-healing
capture, zero-downtime curl-loop proof, Grafana/ArgoCD screenshots, and an
honest framing paragraph: single-node K8s is for operational-pattern
fidelity, with an explicit list of what changes on EKS/GKE.

## Out of Scope

- Fine-tuning of any model
- Multi-turn memory beyond a single conversation id (phase-2+ nice-to-have)
- Cloud deployment / Terraform (documented as roadmap only)
- Replacing `/personalized/offer` (explicitly rejected)

## Success Criteria

1. Copilot answers benchmark questions with correct citations and answers
   data questions via tools (golden set: retrieval hit rate ≥ 0.8,
   behavior eval ≥ 0.9, faithfulness mean ≥ 4.0 at launch thresholds —
   tune after first baseline run)
2. Full stack runs on Docker Desktop K8s; dashboard usable end-to-end
3. GitOps loop demonstrably works (recorded evidence in README)
4. Both Grafana dashboards live with real data
5. CI eval gate blocks a deliberately broken prompt change (tested once)

## Risks / Mitigations

- **Gemini free-tier rate limits** (embeddings + judge): small corpus,
  batched ingest, judge runs nightly only; fallback templates already exist
  for the offer path.
- **Corpus licensing**: store summaries/excerpts with attribution, not full
  copyrighted texts; repo currently private.
- **Scope creep**: three phases, each independently shippable; implementation
  starts only on explicit go-ahead.
