# CTDE — Experiment Design & Permutation Plan

The CTDE agent (`ctde_v0/`) is a **configurable composition of swappable modules** (per
`../agent_architecture.md`). This is the canonical list of dials and the **staged, gated sweep** that finds
the agent reaching 90/90 — *not* a full Cartesian (that's ~250k configs). Every run saves its full config.

**Why a v0 baseline exists already:** the estimator study settled the backbone/training dials, so roughly
half the table below is **live as CLI knobs in the v0**; what remains to build is the **cognition layer**
(roles / tools / compass) + a few message/reward knobs + recurrence.

> ## ⛔ SETTLED LEDGER — DO NOT RE-RUN (2026-06-27)
> **Connectivity-mechanism dials (#11–#13, + `conn_signal`) are TRIED & SETTLED: they FAIL at scale.**
> **I1 @ 16²/4 (8 cfg, 1500 iters):** roles WIN — `role_expl_relay` ≈ **90.9 % cov / 100 % conn** (split
> ≈84 % expl/16 % relay); **every `role_off` HUDDLES @ 1.6–5.6 % cov**. Mechanism (action_mask vs soft_lambda)
> & anti_overlap are **2nd-order** (4 role-on cells span **89.6–90.9 %**). **Scale-transfer of that winner**
> (comm_r 5→6→7 for degree ≈1.9): **16²/4 90.9 % → 24²/6 ~53 % → 32²/10 ~16 %**, all **conn 100 %** — held by
> **HUDDLING** (mean λ₂ ~2.7 @32²; relays abandoned, expl-frac → **99.5 %**; λ̂₂ aux ~81 %, controller 100 %
> valid). **`local_edge_margin` sweep** (degree_target 1.0, collision_mask on × {soft_lambda, lagrangian,
> pid_lagrangian} × {24²/6,32²/10}): **NO improvement** — cov 24² = **33/18/55 %**, 32² = **15/13/16 %**, mean
> λ₂ even higher (~4 @32²), **dual λ FLAT (0.30→0.30, violation ≈0.001)**. **DIAGNOSTIC:** at the huddle
> degree ≫ target ⇒ penalty ≈ 0 ⇒ the huddle *satisfies* the connectivity signal ⇒ the mechanism is
> **structurally inert**. ⇒ **Do NOT re-run #11–#13 / conn_signal at scale.** New direction = **L4 phase +
> delivered-coverage + barrier** (rows added below). Full arc: `../JOURNEY.md` 2026-06-27.

## The dials

| # | Axis | Values | stage | status |
|---|---|---|---|---|
| **Backbone** |
| 1 | aggregator | mean · max · multihead | F | live |
| 2 | mp_rounds | 2 · 4 · 8 | F | live |
| 3 | recurrence | feed-forward · recurrent-GNN | F | build |
| 4 | width / depth / norm | 64·128·256 / 2·3 / layer·none | F | live |
| **Messages** |
| 5 | message content | learned · +edge-distance · +geometry | F | build |
| 6 | index in msgs | none · index | I1 | build |
| **Cognition** |
| 7 | role-picker | off · {explorer, relay} | **I1** | live · **✅ KEEP (decisive @16²: 90.9 % vs ≤5.6 %)** |
| 8 | explorer tool | goal-head · frontier-attention | I2 | build |
| 9 | relay tool | λ̂₂-anchor · hold-heuristic | I2 | build |
| 10 | compass | off · on | I2 | build |
| **L4 strategy / phase (NEW — 2026-06-27)** |
| 7′ | **phase layer (L4)** | off · scripted-timing · learned `{disperse,gather}` | **2′** | build |
| 7″ | **phase commit k** | 5 · 8 · 10 steps | **2′** | build |
| **Connectivity** ⛔ **(11–13 + conn_signal SETTLED — fail at scale, do not re-run)** |
| 11 | ~~mechanism~~ ⛔ | action_mask · soft-λ · lagrangian · pid_lagrangian | ~~I1~~ | live · **DONE (huddles @ scale)** |
| 11b | ~~conn_signal~~ ⛔ | global_lambda2 · local_edge_margin | ~~I1c~~ | live · **DONE (local doesn't rescue)** |
| 11c | collision_mask | off · on | I1b | live |
| 12 | aux target | soft-λ₂ · binary | F | live (soft) |
| 13 | ~~trade-off λ (soft only)~~ ⛔ | 0.1 · 0.3 · 1 · 3 | ~~I1~~ | live · **DONE (dual λ flat @ huddle)** |
| **Reward / loss** |
| 14 | **anti-overlap** | off · on | **I1** | live · **2nd-order @16² (89.6–90.9 %)** |
| 14′ | **objective (NEW)** | coverage · delivered-coverage (`PersistantNetwork`) | **2′** | build |
| 14″ | **barrier weight (NEW)** | 0=off · >0 (+ `a/M/p/cap`) | **2′** | live · composes UNDER L4 (never alone) |
| 15 | reward normalize | raw · fractional | I1 | build |
| 16 | edge-margin loss | 0 · >0 | I1 | build |
| 17 | aux loss | mse · huber | F | live |
| **Mission-safety as INPUT (NEW — open gap)** |
| 16′ | **safety→brain input** | enforce-only (today) · λ̂₂/barrier as L4/L3 input | **2′** | build (open) |
| **Scale** |
| 18 | scale strategy | single · multi-joint · ladder | S | build |
| 19 | degree-reg | off · on | F | live |

## The staged plan (the permutation)

Full Cartesian ≈ 2¹⁰·3⁵ ≈ **250k** → impossible. Stage one cluster at a time from a moving baseline, OFAT
within, carry winners forward, gate between stages.

| Stage | Question | Permuted | cfg ×3 seeds | Gate / RESULT |
|---|---|---|---|---|
| **I1 — break the huddle** ✅ **DONE** | does role labour-division + reward fix lift coverage off ~7%? | role-picker × mechanism × anti-overlap (2×2×2); index/edge-content/compass baked ON; backbone fixed (max/2/64) | **8 → 24** | **PASSED @16²/4: roles → 90.9 % cov / 100 % conn** (role_off huddles 1.6–5.6 %); mechanism/anti-overlap 2nd-order |
| **I1-scale + edge-margin** ⛔ **SETTLED-FAIL** | does the I1 winner / per-step guardrails hold at scale? | scale-transfer (24²/6, 32²/10) + `conn_signal=local_edge_margin` × {soft_lambda, lagrangian, pid_lagrangian} | — | **FAILED: cov 90.9 % → 53 % → 16 %**; huddles (mean λ₂ ↑, dual λ flat). **Do not re-run (ledger above).** |
| **2′ — L4 phase + delivered-coverage** 🔜 **NEW** | does resolving cov↔conn in TIME break the scale huddle? | phase-layer {scripted-timing→learned} × commit-k × objective {coverage·delivered} × barrier {0·>0}; mission-safety→brain input | **~10 → 30** | coverage RISES at 24²/6 & 32²/10 (vs ≤16 %), conn held intermittently, **no per-step conn penalty** |
| **I2 — cognition tools** | which explorer/relay tools win? | explorer-tool × relay-tool × compass (on the I1/2′ winner) | **8 → 24** | best cov↔conn frontier |
| **F — foundation (OFAT)** | best backbone + 16→32 transfer | agg, mp_rounds, recurrence, width, content, aux-target — one at a time | **~14 → 42** | transfers to 32²/10 |
| **S — scale** | which warm-start regime? | scale-strategy {single·joint·ladder} | **3 → 9** | best 90/90 @ 32²/10 |

≈ **33 configs ≈ 100 runs** (+ Phase 2′), matching `../EXPERIMENT_PLAN.md`'s envelope.

**Reading it:** I1 was make-or-break — **roles beat the huddle @16²/4 but the win didn't scale**, and
**per-step connectivity guardrails are now settled-closed at scale** (ledger above). **Phase 2′ (L4 phase +
delivered-coverage + barrier) is the new active stage** — resolve cov↔conn *in time*, not per step. I2 tunes
the tools; F is the estimator-derived backbone (rounds/content/recurrence = where the 0.66 ceiling lived); S
pushes to 32²/10.

## Run

Sharded, resumable, parallel on balthar (mirrors `FiedlerValueEstimation/run_grid.py`):
```
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.:~/ZymeraExp/FiedlerValueEstimation \
  python -u -m ctde_v0.run_ctde_sweep <shard> <nshards>      # one worker per shard
```
Each config logs coverage% · connectivity% (steps λ₂>τ) · mean λ₂ · |λ̂₂−λ₂| · controller-valid% · role split
· **dual λ (Lagrangian/PID — the flat-dual diagnostic of the §0 ledger)**, and saves its §5 config.
Leaderboard ranks on the coverage↔connectivity frontier. *(For Phase 2′ also log delivered-coverage% and
phase-occupancy {disperse,gather}.)*
