---
title: This Platform — What It Can Answer and Where Its Predictions Are Weak
source: Player Monetization Intelligence Platform README (Limitations & Lessons Learned)
url: internal://README.md
retrieved: 2026-07-05
---

# Platform Capabilities and Honest Limits

## What the platform provides

Five services over a 17,955-user dataset (15,155 real users from a public
freemium-game dataset + 2,800 synthetic users): calibrated payer-propensity
and pLTV prediction, an expected-revenue decision engine (IAP vs rewarded ad
vs interstitial vs skip), channel ROI reporting, cohort retention proxies,
and LLM-generated offer copy. Every prediction and decision is logged for
audit.

## Trust the synthetic-cohort predictions more than the real-cohort ones

The real backbone has **no behavioral telemetry** (no session/ad events), so
real users' behavior features are statistically uninformative by design —
propensity AUC on the real cohort is ≈ 0.57, barely above demographics-only.
Synthetic cohorts carry causally generated behavior and reach **AUC
0.75-0.82**, which is the realistic expectation for production data with
real telemetry. When asked about real-cohort predictions, the honest answer
is: directional at best, driven by demographics.

## Metric semantics to keep in mind

`target_ltv` is the observed value over the available window (a **D7
snapshot** for real users, not true D30 — production D30 would typically run
1.2-2× higher). Retention figures from the cohort service are **proxies**
computed from session thresholds, biased upward versus true return-rate
retention. Channel ROI uses observed spend/revenue joins and shows a
disclaimer for the window mismatch.

## Guardrails in the decision engine

pLTV is gated to $0 when payer probability is below the cohort-aware gate
(real 0.10 / synthetic 0.20) because the LTV model is trained on payers
only. IAP offers additionally require payer probability ≥ 0.25; below that
the engine serves ads or skips. Heavily ad-fatigued users (60+ recent ad
views) are skipped entirely outside of after-loss rewarded moments.
