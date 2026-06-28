# SuperBlue Shared-Exploration — Campaign Review & Journey

**Written 2026-06-27.** The honest, consolidated record of the connectivity-constrained-coverage
campaign: the decisions (why & how we made them), the results we got, and a skeptical review of what
the results actually support. This supersedes the rosier in-flight summaries — read this one.

---

## 0. The goal & the agent

**Goal:** a decentralized swarm doing connectivity-constrained coverage reaches **≥90% coverage AND ≥90%
connectivity simultaneously**, pushed up a scale ladder to the hard rung **32×32 grid / 10 agents**, within a
**fixed 100-step budget**.

**Agent (CTDE v0):** LPAC-style backbone (per-agent CNN → GAP → GNN message-passing over the comm graph) →
multi-level action stack (a learned **goal/role head** picks an intent; a fixed L1 controller turns it into a
valid 1-step move). Centralized critic (training only), decentralized actors. Connectivity = the Fiedler
value **λ₂** of the comm graph.

---

## 1. The narrative arc — decisions and results, in order

### Phase 1 — Roles break the huddle (I1 @ 16²/4)
- **Decision:** sweep `role-picker {off, explorer/relay} × mechanism {action_mask, soft_lambda} × anti-overlap`.
- **Why:** the homogeneous agent collapses into a clump ("the huddle"); hypothesis was that an explicit
  **explorer/relay labor division** would break it.
- **Result:** `role_expl_relay` ≈ **90.9% cov / 100% conn**; every homogeneous config **huddled at 1.6–5.6%**.
  Mechanism and anti-overlap were 2nd-order (role configs spanned 89.6–90.9%). **→ Roles are the win; keep them.**

### Phase 2 — The win does NOT transfer (scale-transfer)
- **Decision:** train the winner fresh at each rung 16²/4 → 24²/6 → 32²/10.
- **⚠ Decision that became the campaign's central flaw:** set **comm_r per rung (5 → 6 → 7)** to hold mean
  comm-graph degree ≈ 1.9. **Why we did it:** a documented "transfer precondition" — keep the GNN's local graph
  structure constant so a model trained small transfers to large. **Why it was wrong:** it changed the *agent
  spec* across rungs (the radio grew with the world), which (a) confounded every cross-rung claim and (b) made
  connectivity progressively *easier* at scale. We did this silently in a default; we should have surfaced it.
- **Result:** cov **90.9% (16²) → 53% (24²) → 16% (32²)**, connectivity 100% throughout. **Coverage collapses
  with scale; the team keeps connectivity by huddling** (meanλ₂ high, relays abandoned at 32²: explorer-frac → 99.5%).

### Phase 3 — Diagnose *why* it collapses (the diagnostic sweep)
- **Decision:** before building any fix, probe each candidate cause at 32²/10.
- **Why:** rule out the cheap explanations so we don't build a complex fix for a simple problem.
- **Results (all @ 32²/10, vs the 16% baseline):**

  | hypothesis | runs | result | verdict |
  |---|---|---|---|
  | connectivity mechanism (mask / soft-λ / Lagrangian / PID × global/local) | many | ~16% | ❌ the huddle *satisfies* any per-step signal — λ stayed flat, structurally inert |
  | reward balance (cov:conn 4:2, 8:1, conn-OFF) | 3 | 15–20% | ❌ they don't disperse even with zero incentive to clump |
  | exploration knobs (entropy↑, stride↑) | 3 | 8.9% / 14.5% | ❌ entropy made it *worse* |
  | backbone (mp-rounds, aggregator) | 6 | 18–19% | ❌ more mp-rounds over-smooth → tighter huddle; `max`-agg best |
  | horizon 100→200→400 | 2 | 16→20→30% | ➖ real but capped by the 100-step mission |

- **→ Diagnosis:** the wall is an **exploration / policy-learning failure** — from a clustered spawn the policy
  *cannot discover* how to spread across the big grid. No reward / mechanism / backbone tuning moves it.

### Phase 4 — The breakthrough: give them a disperse skill
- **Decision:** build a **frontier-attention explorer** — a learned per-sector attention that biases the goal
  toward the most *uncovered* compass sector (frontier-positive by construction, scale-invariant).
- **Why:** the diagnosis said the agents can't *learn* to disperse → hand them the behavior explicitly.
- **Result:** cov **16% → 42%** @ 32²/10 (comm_r=7). meanλ₂ *dropped* (genuinely spread, not clumped).
  **The first thing to move the wall.**

### Phase 5 — Stacking the winners → a fuzzy ~47% ceiling
- **Decision:** stack the wins (width, degree-reg, compass, messages) and test the cognition dials.
- **Results (@32²/10, comm_r=7):** + width128+dreg0 = 46% · + compass = 46.9% (but **seed1 = 31.6%** → high
  variance) · + edge_distance msg = 48.9% · **all stacked = 45.6% (non-additive)** · recurrence = 44% (neutral)
  · hold-relay = 30.6% (**worse** than the active relay).
- **→** Single-rung dial-tuning **plateaus ~45–49%, non-additive and seed-noisy**.

### Phase 6 — Warm-start curriculum (the apparent best)
- **Decision:** test the scale-strategy ladder (train 16²→24²→32², warm-starting each rung).
- **Result:** **52.4%** @32² — the highest. *But* the radio grew mid-curriculum (5→6→7), so this number is
  **comm_r-confounded** (see Phase 7).

### Phase 7 — The catch: the spec wasn't fixed
- **What happened:** flagged that "connectivity held 100%" was doing no work — verified `mechanism = soft_lambda`
  (no hard guardrail) but the penalty was **inert** (meanλ₂ ≈ 2.0 vs a trivial `min_lambda2 = 0.001` floor), and
  the connectivity-% metric is `λ₂ > 0.001` = merely *"not fully disconnected."* Then the deeper issue surfaced:
  **comm_r drifted across rungs**, so connectivity was easy *by construction* and the agent was never the same
  agent across the ladder.
- **Decision:** re-run the load-bearing tests with **comm_r = 5 fixed everywhere** (the base radio spec).
- **Result (the honest 32²/10 numbers):**

  | comm_r=5 fixed | cov | conn (λ₂>1e-3) | meanλ₂ |
  |---|---|---|---|
  | goal_head | 13.5% | 100% | 1.54 (huddle) |
  | **frontier-attn** | **31.6%** | **98.7%** | **0.71** |
  | frontier-attn @24²/6 | 49.5% | 100% | 1.46 |

  **The disperse skill's real coverage is ~32%, not 42%** (the confound was worth ~10 pts), but the *relative*
  win holds (**13.5 → 31.6, +18, ~2.3×**). And for the first time **connectivity has a cost** — the dispersed
  agent dips below 100% (98.7%) and its λ₂ halves (0.71). The trade-off finally bites.

---

## 2. Complete results table (one place, comm_r noted)

| run | rung | comm_r | coverage | connectivity | note |
|---|---|---|---|---|---|
| roles win (I1) | 16²/4 | 5 | **90.9%** | 100% | the small-scale win |
| huddle (role_off) | 16²/4 | 5 | 1.6–5.6% | 100% | |
| scale-transfer | 24²/6 | 6 | 53% | 100% | confounded |
| scale-transfer | 32²/10 | 7 | **16%** | 100% | the collapse (confounded) |
| diagnostics (reward/mech/explore/backbone) | 32²/10 | 7 | 8–20% | 100% | all ❌ |
| **frontier-attn (disperse)** | 32²/10 | 7 | **42%** (seed1 38) | 100% | breakthrough (confounded scale) |
| + dials (width/compass/edge/stack) | 32²/10 | 7 | 45–49% | 100% | plateau, seed-noisy |
| recurrence / hold-relay | 32²/10 | 7 | 44% / 31% | 100% | neutral / worse |
| warm-start curriculum | 32²/10 | 5→6→7 | **52%** | 100% | confounded (radio grew) |
| **frontier-attn (HONEST)** | 32²/10 | **5 fixed** | **31.6%** | **98.7%** | the trustworthy number |
| goal_head (HONEST) | 32²/10 | 5 fixed | 13.5% | 100% | honest baseline |

---

## 3. What we can actually stand behind

- **The disperse skill is a real lever** — at a *matched* comm_r it beats the baseline ~2.3× (13.5→31.6 @cr5;
  16→42 @cr7). Not a comm_r artifact.
- **The diagnosis** — "can't *learn* to disperse, but *executes* a given disperse behavior" — is well-supported.
- **The dead-ends** — reward balance, connectivity mechanism, exploration knobs, backbone-stacking, hold-relay,
  recurrence — each has multiple runs behind the "no." Don't revisit.
- **Infrastructure:** six scale-invariant, config-gated, default-byte-unchanged dial-builds shipped & green
  (frontier-attn explorer · relay-tool · compass · recurrence · message-content · warm-start), CTDE v0.

## 4. Honest caveats (what NOT to claim)

- **Absolute coverage numbers above comm_r=5 are confounded** — only the fixed-spec re-runs are trustworthy.
  Headline so far: **disperse skill ≈ 32% @ 32²/10, honest spec.**
- **The "47% ceiling" is fuzzy** — mostly single-seed, with measured seed-noise of ±5–10 pts (compass 32–47%).
- **The "52% warm-start win" is unproven** — radio grew mid-curriculum; needs a fixed-spec re-run.
- **We studied COVERAGE-at-scale, not 90/90.** Connectivity was made easy (generous comm_r + a trivial
  `λ₂>0.001` bar), so the real coverage↔connectivity trade-off was barely tested. The 90/90 goal is **still open.**
- **The gather/L4 idea is a hypothesis, not a result.**
- Everything is on **one architecture (CTDE v0)**; "can't learn to disperse" may be architecture-specific.

## 5. Where it stands / open questions for the next push
- Re-establish the honest baseline at **fixed comm_r=5** with a **real, binding connectivity bar** (proposed
  λ₂>0.5, pending sign-off) — *that's* the true 90/90 starting line.
- Is the coverage plateau the **agent** or the **100-step budget**? (h-sweep says budget matters, but
  disperse-skill@h100 > baseline@h400, so dispersal efficiency dominates.)
- Does **warm-start curriculum** survive a fixed spec?
- **The gather half (L4)** — disperse↔gather rhythm + delivered-coverage — is the proposed lever for the
  coverage↔connectivity trade-off; untested.

---

*Bottom line: we found a real coverage lever (~2.3×) and a clean diagnosis, built the tooling to test the whole
idea-space, and — critically — discovered that a silent spec choice (comm_r per rung) had made connectivity
free, so the actual 90/90 problem is still open. The qualitative story (disperse=lever, dispersal=wall,
gather=next) holds; the quantitative + connectivity claims await the fixed-spec, real-bar re-runs.*

---

## 6. The A/B arm batch (2026-06-27/28) — which lever cracks coverage-at-scale

Four arms attacking the 32²/10 coverage wall, all on the LOCKED honest spec (comm_r=5 every rung, hard
collision-mask, soft/learned connectivity, frontier-attn explorer, 100-step horizon, 3 seeds, density-pinned
ladder 16→24→32). Launcher `ctde_v0/run_ab_overnight.py`; on balthar (`XLA_PYTHON_CLIENT_PREALLOCATE=false`,
jobs=3, tmux `ab`).

- **base** — the shared frontier-attn explorer (reference + fork bootstrap).
- **armA** — a *training* strategy: curriculum (warm-start ladder) + σ-noise + a **DTE tail** (decentralized
  critic at the top rung).
- **B-fork** — an *architecture*: 2 groups with SEPARATE params (`GroupedActor`); shared CTDE critic.
- **B-dico** — an *architecture*: one shared policy + a mean-zero per-agent identity residual.

### Results (partial — 23/39 runs; coverage %, conn = λ₂>0.5)

| arm | 16²/4 (3 seeds) | 24²/6 | 32²/10 | role-div |
|---|---|---|---|---|
| base | ~59 | ~66 | 41 | 0.05–0.15 |
| armA | ~65 | 70 | 47 | 0.06–0.12 (lowest spread) |
| **B-fork** | **~73** | 64 | 50 | **0.11–0.17 (highest)** |
| **B-dico** | **~73** | ~65 | **61 (seed1)** | 0.07–0.14 |
| armA-DTE | — | — | **8 (collapsed)** | 0.04 |

Connectivity sat at **90–100% on the strict λ₂>0.5 bar everywhere** — every arm, every rung.

### What it supports
1. **Every arm beats base on coverage** (+6–14 pts @16²). The mechanisms help.
2. **The diversity/architecture arms (B-fork, B-dico) lead — most at scale.** At 32²/10 they reach 50–61% vs
   base's 41%; B-dico's seed1 hit **61%**, the best big-rung result so far.
3. **The DTE tail COLLAPSED at 32²/10 (8% coverage, 100% connectivity = the huddle).** Dropping the central
   critic at the hard rung returns it straight to the local optimum → **the CTDE central critic is
   load-bearing at scale.** (Validates the cognition design's "keep CTDE for the executor; only the selector
   goes to ES.")
4. **Connectivity is NOT the bottleneck — coverage-at-scale is.** Soft connectivity + the clustered start
   hold 90–100% everywhere; the wall is coverage (~50–61% @32²), still short of 90.
5. **C0 confirmed — specialization emerges.** B-fork shows the highest persistent role differentiation;
   separated/individuated agents genuinely divide labour, most at scale. → green light for the selector-over-
   skills path.

### Honest caveats
- **32²/10 is single-seed-per-arm (different seeds)** → the cross-arm 32² ranking is PRELIMINARY (B-dico 61%
  is seed1; the others seed0). The 16² numbers (3 seeds, tight) are the solid ones.
- Still running (23/39); big-rung seeds incomplete.
- **More diversity ≠ automatically more coverage** (B-fork is the MOST diverse but only edges armA). *Useful*
  (task-grounded) diversity is what matters, not free-floating behavioural spread.

### What it changes
The premise for the **two-level cognition** is now empirically supported (specialization helps + emerges;
CTDE is load-bearing). Design converged → `COGNITION_DESIGN.md`. Next: the **selector over a small skill
library** {disperse, flock, hold} — the adaptive, grown-up version of B-fork — with **ES on the selector +
CTDE on the executor**, swept over flock {scripted,learned} × congestion {off,on} (+ the fixed-world N-sweep).

---

## §7 — Obstacle reruns (finished) + the crowded-obstacle curriculum batch (2026-06-28 cont.)

### 7a · Obstacle batch — FINISHED (50/54; 4 r32 OOM-casualties)
Full factorial **ARM {role,base,sel} × BARRIER {off,on} × EXPLORE {eoff,ebump,einfo} × WORLD {o16-open,
r24-rooms, r32-rooms}**, fresh (no warm-start), 1500 iters, fixed honest spec. The **r24 (24² rooms)
barrier verdict** (the question the rooms runs existed to answer):

| arm (24² rooms) | barrier OFF (cov / conn) | barrier ON (cov / conn) |
|---|---|---|
| role + bump | **32.7 / 88** | 23.0 / 100 |
| role + plain | 26.3 / 92 | 11.7 / 100 |
| base + bump | 28.9 / 92 | 6.1 / 100 |
| sel + bump | 23.7 / 94 | 10.8 / 100 |

**Verdicts (now confirmed across open AND corridors):**
1. **The connectivity barrier is the WRONG tool — refuted even in corridors.** Barrier-OFF already holds
   88–98% connectivity; barrier-ON pays **half the coverage** for the last ~10 pts it didn't need. (The lit
   review explains *why*: a fixed penalty coefficient is brittle — fix = a learned Lagrangian. See `STRATEGY`/
   `OPEN_THREADS` VALIDATED.)
2. **`einfo` (info-gain bonus) reward-hacks everywhere (~1%)** — rewards proximity-to-uncovered, maximized by
   NOT covering. **`ebump` (coverage ×3) wins. role ≥ base ≥ sel.**
3. Rooms cap coverage low (~33% even trained) — chokepoints are hard in 100 steps.

### 7b · Connectivity-safe crowded terrains (`ctde_v0/terrains.py`)
`ConnectedClutter` · `Pillars` · `MixedCluttRooms` · `RandomCrowded` (per-reset mixture). A fixed-iteration
**BFS flood-fill from a central seed walls off every unreachable cell** → free space is *constructed* to be
one connected component (JAX-traceable, runs in the jitted reset; coverage well-posed, spawn safe). 10 tests,
suite **105 green**. Committed `0a80fed` (deployed to balthar by rsync, NOT pushed to main).

### 7c · Crowded curriculum batch — RAN (10/12)
16→24→32 warm-start on `terrain=crowded_mix`, ARM {role,base} × EXPLORE {eoff,ebump}, density ~15%/rung.
**Trained-vs-zero-shot on identical eval maps (role+bump):**

| map | 24² zero-shot → TRAINED | 32² zero-shot → TRAINED |
|---|---|---|
| clutter light | 35.9 → 37.0 (+1) | 24.8 → 22.0 (−3) |
| clutter heavy | 17.3 → **28.5 (+11)** | 14.9 → 17.1 (+2) |
| pillars | 28.6 → **41.3 (+13)** | 30.2 → 28.5 (−2) |
| mixed | 21.0 → 25.3 (+4) | 10.3 → 12.6 (+2) |

**role vs base @32² crowded (both trained):** pillars 28.5 vs 22.8 (**+5.7**), heavy clutter 17.1 vs 13.3
(**+3.8**), light/mixed ≈ tie.

**What it supports / changes:**
1. **Training-on-crowded HELPS at 24² (+11–13 hard maps) but WASHES at 32² (±3).** The crowded skill the
   curriculum learns **doesn't transfer up to 32²** → weak scale-transfer of the harder regime. *This is the
   lit-review's central open question, answered "weakly" — and the strongest evidence yet that 32² is the wall.*
2. **role > base on crowded, most on the hard maps** — the explorer/relay split keeps paying off under clutter.
3. **Crowded 32²/10 tops out ~12–28% — far from 90/90.**

**Honest caveats:** single map per terrain (before/after is same-map/fair, but one sample); 1 seed; `role_eoff/32`
+ `base_eoff/32` flaked on a simultaneous-compile OOM (re-runnable jobs 1). **Next:** barrier → learned-λ (RCPO);
more iters on the 32² rung (under-training test); multi-map eval. Gallery: `report/index.html` (41 runs, 7 cats).
