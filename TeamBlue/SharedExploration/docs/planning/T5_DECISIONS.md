# T5 — Decision & Parameter Ledger

Status tags: **[DECIDED]** settled · **[TEST]** ablation arm to run · **[OPEN]** needs your call · **[CHECK]** code fact to verify first · **[REJECTED]** ruled out, don't revisit.

Object of study: a **fully-learned autonomous agent** (drone/agent with agency) in a team — connectivity-constrained cooperative coverage that generalizes to unseen maps, on the path to adversarial/covert-misbehavior resilience. No classical planner at inference (a classical planner has no agency to study and nothing covert to attack).

---

## 1. Architecture skeleton (the three clocks)

| Piece | Status | Value / note |
|---|---|---|
| Backbone LPAC-KB (CNN → GNN → 64-d KB) | **[DECIDED]** | Keep unchanged; warm-start from best ctde_v0 checkpoint. Only size-invariant thing we have. |
| **L3** — slowest (~20–50 steps) | **[DECIDED]** role | Attend KB → commit to a frontier **region** (goal head) + connectivity stance; commit-and-hold, re-select on *reached* or *belief-drift*. |
| **L2** — medium (~5–10 steps / event) | **[DECIDED]** role | The learned **planner**: route toward L3's region over the occupancy belief. (Placement at L2 was your correction — L1 is too fast for a planner.) |
| **L1** — fastest (every step) | **[DECIDED]** role | Reactive tracker: step toward L2's waypoint + dodge collisions / locally-seen obstacles. |
| L2 roles / skills layer | **[REJECTED]** | T2: flat ≥ role everywhere; RODE win was action-space, which the L3 goal head supplies. Relay = "hold", emerges without a label. |

---

## 2. Planner (L2 — the generalization piece)

| Item | Status | Value / note |
|---|---|---|
| Planner (primary) | **[DECIDED]** | **MVProp** + Highway/DT-VIN depth-fix. GPPN kept only as a fallback if MVProp's RL training proves unstable. |
| VIN | **[REJECTED]** | Dropped per Bijan — MVProp is the pick. (It was only a lit baseline-to-beat; we forgo that comparability.) |
| MSP arm | **[TEST]** | Multi-agent Spatial-Planning Transformer as a competing L2 arm. Eyes-open: O((H·W)²) attention → window-size ceiling, weak scale transfer, no structural dead-end guarantee. |
| Unknown cells | **[DECIDED]** | Treated as **optimistically free** — plan *into* the unknown toward frontiers. |
| K (propagation depth) | **[OPEN]** | Size to **window diameter** (~40–60 @ 32²), NOT the global map. Finalize per map size. |
| Replan trigger (L2 cadence) | **[OPEN]** | Default: event-triggered (next-cell-blocked / stuck-counter / goal-reached) **+ periodic floor** (~5–10 steps). **Bijan's option — a *learned* L3 termination head that decides when to replan** (Option-Critic-style β). Run as an ablation ARM, not the default: learned termination collapses (β→every step, or never) without a deliberation cost. Three arms to compare: **(a)** heuristic scheduled cadence [baseline]; **(b)** learned termination + **energy cost** (each replan/action debits a per-agent energy budget — physically meaningful *and* instantiates RQ5 energy-constraints, better than an arbitrary penalty); **(c)** learned termination + explicit deliberation reward. Keep energy MINIMAL here (a per-replan debit); the full energy-constrained-mission scope is a separate research track, don't sign up for it just to regularize termination. |
| Planner teacher | **[OPEN]** | (a) **distill** from BFS/Neural-A\* teacher → fast, stable, deploy-pure, "reproduces planning" claim; (b) **RL-only** → "discovers planning", finicky. Both keep deployment fully learned. |
| Trajectory Diffusion / Decision Transformer | **[REJECTED]** | Wrong generalization (unseen goals, not unseen walls) + blows per-step budget. |
| Neural-A\* / ANS-FMM at inference | **[REJECTED]** | Classical at inference — out of scope. Allowed only as a distillation *teacher*. |

---

## 3. Goal head (L3)

| Item | Status | Value / note |
|---|---|---|
| Goal-region selection head | **[DECIDED]** | Replaces the 1-step move head (the action-representation bottleneck fix). |
| Encoder capacity | **[DECIDED]** | Keep CNN+GNN; do **not** grow for generalization (gap is planning, not perception). Ensure fully-conv + count/perm-invariant (invariance, not capacity). |
| Rich intention-sharing L3 | **[OPEN / deferred]** | Reads null on coverage; earns its keep only on an **obligate** mission (cornering), where difference rewards collapse. Defer to the 2nd mission. |

---

## 4. Connectivity mechanism

Existing four terms carry into T5 **unchanged** (they're reward/loss → architecture-agnostic):

| Term | Where | Role |
|---|---|---|
| `2·reach_fraction` | reward | The mechanism that actually maintains connectivity (reachability). |
| `−λ·relu(τ − λ₂)` | reward | Lagrangian/PID guardrail — dormant "just in case" (τ below natural λ₂). Keep. |
| `0.1·MSE(λ̂₂, λ₂)` | loss | Critic predicts λ₂ (representation). |
| `0.001·Var(mean_degree)` | loss | Minor degree-spread regularizer. |

| Decision | Status | Note |
|---|---|---|
| Connectivity input to goal head | **[TEST]** | Ablate **critic-only** vs **distilled per-agent Fiedler estimator**. |
| Degree / articulation as the signal | **[REJECTED]** | Fails the K₄-fully-connected-but-dispersing case; only global Fiedler catches sub-team detachment. Also = a covert attack surface. |
| Hard connectivity mask | **[REJECTED]** | Standing rule — soft/learned + emergent relay only. |
| **Occlusion-aware comms** (walls = extra effective distance) | **[DECIDED — model]** | **Effective-distance model:** `d_eff(i,j) = d_ij + c·k_ij` — `k_ij` = walls on the i–j segment, `c` = comm-range cells lost per wall. Substitute `d_eff` for `d` **everywhere**: adjacency `d_eff ≤ comm_r`, λ₂ soft weight `sigmoid(sharp·(comm_r − d_eff))`, multi-hop reach over the `d_eff` graph. **One knob `c` spans free-space (c=0) → hard-LoS (c=comm_r).** **Chosen: `c = comm_r/3`** (a wall costs ~⅓ range → tolerate ~2 close walls, ~1 at moderate range; scale-invariant as a fraction of comm_r), sweepable. **No unmeasured params** (reuses comm_r; d, k computed). Use TRUE walls (physics, not belief) → link-loss = a sensing signal + a predicted-vs-actual connectivity gap. Occlude all three graphs with the same `k`. **Impl = ONE substitution** `d → d + c·k` at `comms.py:68`, `_lambda2` (`missions_terms.py:190`), `kb_adjacency` (`env_utils.py:265`); only new primitive = fixed-K supercover wall-count per pair. No architecture change. Linear approx (drops nonlinear path-loss) — the right level for a connectivity-resilience study. Comm becomes a cliff → fast connectivity guard matters more. |
| Make guardrail bind (raise τ, cut reach reward) | **[OPEN]** | Only if we want the guardrail load-bearing; risk = connectivity collapse if the dual doesn't pick up slack. Default: leave as-is. |

---

## 5. Critic (train-only, CTDE)

| Item | Status | Value / note |
|---|---|---|
| Architecture | **[TEST]** | Ablate **setpool** (count-invariant) vs **setattn**; optional plain conv baseline. Prior (T4): tie. |
| Heads | **[DECIDED]** | Predict V + auxiliary λ̂₂. |

---

## 6. Trainer

| Item | Status | Value / note |
|---|---|---|
| MAPPO-CTDE | **[DECIDED]** | Keep. **Details need a dedicated brainstorm** (critic inputs, credit assignment, rollout/minibatch, PPO knobs). |
| GRPO | **[REJECTED]** | Removes the critic where global connectivity info must live; group-relative credit is a cruder difference reward. |

---

## 7. Reward & loss (exact — current, carried into T5)

**Reward (per agent):**
`r_i = 3·new_coverage + 2·reach_fraction − 4·collision − 1·[λ·relu(τ − λ₂)]`

**Loss:**
`L = PG(goal) + 0.5·MSE(V, ret) + 0.1·MSE(λ̂₂, λ₂) + 0.001·Var_batch(mean_degree) − 0.01·entropy`
— **PG(role) term DROPPED** (roles cut).

**Optimizer / rollout:** GAE γ=0.99, λ=0.95 · clipped-PG clip=0.2 · AdamW lr=3e-4 · grad-clip 0.5 · 6 PPO epochs × 4 minibatches × **16 rollouts** × 100 steps. **[HARD: never reduce 16 rollouts.]**

**Dual update:** lagrangian `λ ← relu(λ + 0.05·v)`; PID `λ = relu(1.0·v + 0.01·Σv + 0.1·(v − v_prev))`; `v = relu(τ − mean λ₂)`.

| Parameter | Current | Status |
|---|---|---|
| w_coverage | 3 | [DECIDED] |
| w_reach (connectivity) | 2 | **[OPEN]** — reduce only if guardrail should bind |
| w_collision | 4 | [DECIDED] |
| τ (constraint threshold) | 0.5 / 0.7 | **[OPEN]** — raise to ~2 only to make guardrail bind |

---

## 8. Mission & training protocol

| Item | Status | Value / note |
|---|---|---|
| T5 baseline mission | **[DECIDED]** | Connectivity-aware cooperative coverage, unseen SAR 32² maps (6 families). |
| Warm-start | **[DECIDED]** | Backbone-init now; scale-ladder transfer later (the ~26× is same-arch scale, not the arch change). |
| Zero-shot eval | **[DECIDED]** | Held-out SAR maps; metrics = % coverage + connectivity (λ₂ / connected-fraction). |
| 2nd mission (obligate collaboration) | **[OPEN / later]** | **Cornering** flagship + **Byzantine agreement** theory anchor. Design session pending. |
| Emergence measurement (collab claims) | **[DECIDED]** | Order parameter + role-MI + ablation vs reward-only. |

---

## 9. Prerequisite code checks (do before finalizing §4)

- **[ANSWERED ✓]** `reach_fraction` is **multi-hop** — transitive closure of the comm graph (`missions_terms.py:57-63` → `metrics.reach`). Matches the recommendation; already catches sub-team dispersal. No change.
- **[ANSWERED ✓]** `λ̂₂` head is **actor-side** (decentralized per-agent; `nets.py:840/904/987`, consumed at execution; critic has none). → The distilled connectivity input is **plumbing, not a new build** — route the existing per-agent λ̂₂ into the goal head. Per-agent *criticality* (∂λ₂/∂me) = a small extension: same head, new target.

---

## 10. Open decisions to finalize tomorrow (the short list)

1. **Planner class** to build first: MVProp+Highway vs GPPN+Highway vs MSP (or run all as arms).
2. **Planner teacher**: distill vs RL-only.
3. **Connectivity input**: none (reward-only) vs critic-only vs distilled Fiedler (pending the two code checks).
4. **Guardrail**: leave dormant (default) vs make it bind (raise τ + cut reach reward).
5. **L2 replan cadence** floor + **K** depth (per map size).
6. **Critic arch** default (or keep as a live ablation axis).
7. Whether to open **MAPPO details** and **cornering mission design** now or after the coverage baseline. → **PARKED per Bijan: discuss new missions later.**
8. **Training budget for 32²**: increase total updates/episodes (more environment interaction) + lean on warm-start; keep PPO epochs ~6 (raising epochs risks off-policy instability — not the lever). Belongs in the MAPPO-details session.
