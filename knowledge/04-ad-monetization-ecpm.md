---
title: Ad Monetization — Formats, eCPM and Ad Fatigue
source: Game Growth Advisor Mobile Game KPIs 2026; industry eCPM ranges
url: https://gamegrowthadvisor.com/blog/2026-03-17-mobile-game-kpis-benchmarks-2026/
retrieved: 2026-07-05
---

# Ad Monetization: Formats, eCPM, Fatigue

## eCPM by format

eCPM (revenue per 1000 impressions) differs sharply by format. US/UK-tier
medians for mobile games: **rewarded video ≈ $18**, **interstitial ≈ $12**,
**banner ≈ $2**. Per single impression that is roughly $0.018 / $0.012 /
$0.002 respectively.

## Why rewarded video earns the most

Rewarded video is **opt-in**: the player chooses to watch in exchange for an
in-game benefit (extra life, doubled reward). Opt-in attention converts far
better for advertisers, hence the premium eCPM — and completion rates around
**85%** are typical. It is also the format with the least retention damage,
precisely because the player asked for it.

## Format mix

A typical hybrid-casual impression mix is roughly 60% rewarded, 30%
interstitial, 10% banner. Interstitials are forced full-screen breaks —
effective revenue but the primary driver of ad fatigue; banners are
low-value background inventory.

## Ad fatigue

Ad fatigue is the decay of both ad value and player patience as impression
counts climb. Symptoms: falling click-through, rising session abandonment
after interstitials, D1/D7 retention erosion. Practical mitigations are
frequency caps per session/day, suppressing interstitials for likely payers,
and preferring rewarded placements. A monetization engine should treat very
high recent ad-view counts as a signal to reduce or stop serving ads to that
player — protecting retention beats squeezing a fatigued user for cents.

## The IAP-vs-ads tension

Showing an ad and showing an IAP offer compete for the same moment. Expected
value per impression favors IAP only when payment propensity is meaningful:
even a $0.99 offer at 2% purchase chance (~$0.02) barely beats one rewarded
impression (~$0.015) — and a mistargeted IAP prompt annoys non-payers. This
is why production decision engines gate IAP offers behind a minimum payer
probability instead of always chasing the higher nominal price.
