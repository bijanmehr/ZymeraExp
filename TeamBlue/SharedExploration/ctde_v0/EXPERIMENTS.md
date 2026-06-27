# CTDE — Experiment Design & Permutation Plan

The CTDE agent (`ctde_v0/`) is a **configurable composition of swappable modules** (per
`../agent_architecture.md`). This is the canonical list of dials and the **staged, gated sweep** that finds
the agent reaching 90/90 — *not* a full Cartesian (that's ~250k configs). Every run saves its full config.

**Why a v0 baseline exists already:** the estimator study settled the backbone/training dials, so roughly
half the table below is **live as CLI knobs in the v0**; what remains to build is the **cognition layer**
(roles / tools / compass) + a few message/reward knobs + recurrence.

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
| 7 | role-picker | off · {explorer, relay} | **I1** | build |
| 8 | explorer tool | goal-head · frontier-attention | I2 | build |
| 9 | relay tool | λ̂₂-anchor · hold-heuristic | I2 | build |
| 10 | compass | off · on | I2 | build |
| **Connectivity** |
| 11 | mechanism | action_mask · soft-λ | **I1** | live |
| 12 | aux target | soft-λ₂ · binary | F | live (soft) |
| 13 | trade-off λ (soft only) | 0.1 · 0.3 · 1 · 3 | I1 | live |
| **Reward / loss** |
| 14 | **anti-overlap** | off · on | **I1** | build |
| 15 | reward normalize | raw · fractional | I1 | build |
| 16 | edge-margin loss | 0 · >0 | I1 | build |
| 17 | aux loss | mse · huber | F | live |
| **Scale** |
| 18 | scale strategy | single · multi-joint · ladder | S | build |
| 19 | degree-reg | off · on | F | live |

## The staged plan (the permutation)

Full Cartesian ≈ 2¹⁰·3⁵ ≈ **250k** → impossible. Stage one cluster at a time from a moving baseline, OFAT
within, carry winners forward, gate between stages.

| Stage | Question | Permuted | cfg ×3 seeds | Gate |
|---|---|---|---|---|
| **I1 — break the huddle** | does role labour-division + reward fix lift coverage off ~7%? | role-picker × mechanism × anti-overlap (2×2×2); index/edge-content/compass baked ON; backbone fixed (max/2/64) | **8 → 24** | coverage climbs AND conn ≥ 90% |
| **I2 — cognition tools** | which explorer/relay tools win? | explorer-tool × relay-tool × compass (on the I1 winner) | **8 → 24** | best cov↔conn frontier |
| **F — foundation (OFAT)** | best backbone + 16→32 transfer | agg, mp_rounds, recurrence, width, content, aux-target — one at a time | **~14 → 42** | transfers to 32²/10 |
| **S — scale** | which warm-start regime? | scale-strategy {single·joint·ladder} | **3 → 9** | best 90/90 @ 32²/10 |

≈ **33 configs ≈ 100 runs**, matching `../EXPERIMENT_PLAN.md`'s envelope.

**Reading it:** I1 is make-or-break (does the architecture beat the huddle); I2 tunes the tools; F is the
estimator-derived backbone (rounds/content/recurrence = where the 0.66 ceiling lived); S pushes to 32²/10.

## Run

Sharded, resumable, parallel on balthar (mirrors `FiedlerValueEstimation/run_grid.py`):
```
XLA_PYTHON_CLIENT_PREALLOCATE=false PYTHONPATH=.:~/ZymeraExp/FiedlerValueEstimation \
  python -u -m ctde_v0.run_ctde_sweep <shard> <nshards>      # one worker per shard
```
Each config logs coverage% · connectivity% (steps λ₂>τ) · mean λ₂ · |λ̂₂−λ₂| · controller-valid% · role split,
and saves its §5 config. Leaderboard ranks on the coverage↔connectivity frontier.
