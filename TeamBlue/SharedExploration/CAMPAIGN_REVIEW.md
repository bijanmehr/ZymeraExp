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
