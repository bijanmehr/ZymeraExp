# SuperBlue · Shared-Exploration — Experiment Plan

A **staged, gated parameter sweep** to find the agent + training recipe that reaches **90/90**
(coverage / connectivity) on shared exploration, and to map the coverage↔connectivity frontier.
Pairs with `agent_architecture.md` (the agent under test) and `JOURNEY.md` (the running log).
**Every run carries a saved config file** — nothing runs unlogged.

**Methodology.** One axis swept at a time from a **moving baseline**, with a **go/no-go gate** between
phases so a dead end never propagates downstream. We do **not** characterize scripted heuristics, do **not**
include an MLP backbone arm, and do **not** expand any phase into a full within-phase grid (decisions of
2026-06-25). **Foundations first:** the network *and* the training algorithm are settled up front (Phase 1),
then the problem-specific phases build on the locked result. Net cost: **~45 configs × 3 seeds ≈ ~135 core
training runs** (+ cheap transfer-evals), roughly **6–7 overnight batches** on the GPU harness.

---

## 0. ⛔ TRIED & SETTLED — DO NOT RE-RUN  *(read this first; 2026-06-27)*

> ### Per-step connectivity guardrails do **NOT** break the scale huddle. This chapter is CLOSED.
> The **entire family** is settled: hard **action-mask**, soft **global-λ₂** penalty, and **local
> degree / edge-margin** — under **fixed / Lagrangian / PID** dual weighting. At the base scale (16²/4)
> the connectivity signal is irrelevant (roles do the work); **at scale every one of them "succeeds"
> only by HUDDLING** — a dense clump satisfies connectivity for free while coverage collapses.
>
> **KILLER DIAGNOSTIC (why it's structural, not a tuning miss):** at the huddle each agent's degree is
> **≫ the target**, so the connectivity penalty is **≈ 0** and the dual **λ never moves** (measured:
> `0.30 → 0.30`, violation `≈ 0.001`) — **the huddle SATISFIES the connectivity signal.** A per-step
> connectivity mechanism is therefore **structurally inert**: it reads clumping as a *solution*, not a
> problem, so it **cannot push the team apart**. Stronger weighting, smarter dual control (Lagrangian /
> PID), and a more local signal (`local_edge_margin`) were **all tried** — none change the sign.
>
> ⇒ **DO NOT re-run per-step connectivity-mechanism sweeps at scale hoping they fix coverage.** The fix
> must resolve coverage↔connectivity **in TIME** (disperse to cover, gather to share, repeat) — see the
> new **Phase 2′ (L4 phase + delivered-coverage)** below, *not* a stronger per-step penalty.

**Results ledger (exact, reproducible — `ctde_v0/` sweeps on balthar; full arc in `JOURNEY.md` 2026-06-27).**

| # | Sweep | Setup | Result | Verdict |
|---|---|---|---|---|
| **1** | **I1 @ 16²/4** | `role_picker{off,expl_relay}` × `mechanism{action_mask,soft_lambda}` × `anti_overlap{off,on}` = **8 cfg, 1500 iters** | `role_expl_relay` ≈ **90.9 % cov / 100 % conn** (best `soft_lambda·ao_on`; split ≈84 % expl / 16 % relay). **Every** `role_off` cfg **HUDDLES @ 1.6–5.6 % cov** / 100 % conn. Mechanism & anti_overlap **2nd-order** (4 role-on cells span **89.6–90.9 %**). | **Roles are DECISIVE → KEEP.** Don't re-run the 16²/4 mechanism/anti-overlap sweep expecting differentiation. |
| **2** | **Scale-transfer of the I1 winner** | `role_expl_relay` + global-λ₂ `soft_lambda`; `comm_r` per rung for mean degree ≈1.9 (**5→6→7**) | **16²/4 cov 90.9 % → 24²/6 ~53 % → 32²/10 ~16 %**, **ALL conn 100 %.** Held by HUDDLING: **mean λ₂ ~2.7 @32²** (dense clump, not a backbone); role-picker **abandons relays** (expl-frac → **99.5 % @32²**). Machinery healthy: **λ̂₂ aux ~81 %**, **controller 100 % valid**. | **The win does NOT transfer.** Coverage collapses with scale. |
| **3** | **Local-edge-margin sweep** | `conn_signal=local_edge_margin`, `degree_target=1.0`, `collision_mask=on` × `mechanism{soft_lambda, lagrangian, pid_lagrangian}` × `{24²/6, 32²/10}` | **NO improvement.** Cov **24²/6 = 33 % / 18 % / 55 %**, **32²/10 = 15 % / 13 % / 16 %** (ties/worse vs 53 %/16 % baseline). All **conn 100 %**, **expl ~99 %**, **mean λ₂ even HIGHER (~4 @32²)**. **Lagrangian dual λ FLAT** (`0.30→0.30`, violation `≈0.001`). | **Local signal doesn't rescue it.** Confirms the diagnostic. |

**What's KEPT vs CLOSED.** ✅ KEEP: **explorer/relay roles** (the proven huddle-fix at 16²/4). ⛔ CLOSED at
scale: the **Phase-2 connectivity-mechanism × λ frontier** as a *coverage* fix (action-mask / soft-λ₂ /
degree-floor / Lagrangian / PID). 🔜 NEW bet: **Phase 2′** — an **L4 disperse↔gather phase layer** +
**delivered-coverage** objective + a **silent connectivity-floor barrier** (below).

---

## 0′. ⮕ NEXT STEP — the FLAT-BASELINE + FALSIFICATION TEST runs FIRST  *(2026-06-27; see `STRATEGY.md`)*

> **Before building the L4 phase tower of Phase 2′, run the cheap test that decides whether the tower is even
> needed.** The 7-search literature review (consolidated in **`STRATEGY.md`**) says the elaborate
> strategy→role→skill→action brain is over-reach: hierarchy's benefit is *exploration, not structure*
> (Nachum et al. 2019, arXiv:1909.10618), and roles/phases are the kind of thing that **emerges** rather than
> something to hand-author. So the corrected first move is **not** "add L4" — it is **train a near-flat system
> with the right objective and check whether labour-division + the disperse↔gather rhythm emerge on their own.**

**The falsification run (do this before any added structure).** Train the **flat-ish system**:
- **goal/region action head** (the keystone fix for the 1-step-move ceiling — the *one* temporal abstraction the
  HRL evidence backs; Nachum et al. 2019, arXiv:1909.10618) over
- a **size-invariant GNN backbone** (LPAC-style; Agarwal et al. 2025, arXiv:2401.04855), with
- a **learned role latent** (ROMA-style readable latent, NOT a hand-built role layer; Wang et al. 2020,
  arXiv:2003.08039), inside
- the **hard connectivity shell** (action-mask safety envelope, *not* a brain level), under
- the **delivered-coverage** objective (so a reason to gather *exists* without a per-step connectivity penalty),
- at the **fixed honest spec** (density-pinned, connectivity binding at every rung, 100-step budget binding).

**Then measure emergence as the outcome** (don't legislate it): redundancy curve bending toward 1,
**MI(role;behavior)** rising (the role-diversity diagnostic of Hu et al. 2022, arXiv:2207.05683), a visible
breathe-out/breathe-in.

**The gate decides the whole "tower vs. emergence" question:**
- **If roles + the disperse↔gather rhythm EMERGE → the 4-level brain is refuted as unnecessary** (and you have
  a clean emergence result for the resiliency program). Proceed *without* the L4 phase head.
- **If they DON'T emerge even with delivered-coverage → that is the earned evidence a thicker scaffold is
  warranted** — and you'll know *which* level to add and *why*, instead of importing the human hierarchy on
  faith.

**Scale precedent for "flat + curriculum, not tower":** EPC (Long et al. 2020, arXiv:2003.10423) scales flat
multi-agent policies by *evolutionary population curriculum*; LPAC (arXiv:2401.04855) gets zero-shot size
transfer from a flat GNN — both without a strategy/skill tower. **This §0′ run supersedes "go straight to
Phase 2′"; Phase 2′'s L4 layer is now *conditional* on this test failing.** (`STRATEGY.md` holds the full
reference list and the reasoning.)

---

## 1. Locked decisions (campaign invariants)

- **Target:** **≥ 90 % coverage AND ≥ 90 % connectivity simultaneously** at **32×32 / 10 agents**, within a
  **100-step** horizon. Secondary deliverable: the full coverage↔connectivity Pareto frontier.
- **Connectivity is the Fiedler value λ₂**, estimated **decentrally from local observation**
  (decentralized power-iteration, Yang 2010 — replaces the exact-eigendecomposition oracle):
  - **Agent-facing signal:** each agent computes its own local estimate **λ̂₂ᵢ** from local observation; this
    feeds the KB / role-picker / mission-safety / reward. This *is* idea **#10 ≡ #4** (one spectral story).
  - **Grading (non-gameable):** the simulator computes the **true λ₂** of the comm-graph each step.
    **connectivity-% = fraction of the 100 steps with λ₂ > τ** (τ = 1e-3, clears numerical noise; λ₂ > 0 ⇔
    connected). We also log **mean λ₂** (magnitude diagnostic) and the **estimator error |λ̂₂ − λ₂|** (so we
    trust the local estimate and know the agent isn't gaming it).
  - **coverage-% = covered free cells / free cells, at t = 100.**
- **Backbone committed:** an **LPAC-style graph-net** (CNN local-perception → permutation-equivariant GNN
  message-passing → head). No MLP comparison arm.
- **Backbone training — RESOLVED (Round-2):** train the equivariant GNN **from scratch with RL** (IPPO /
  MAPPO) — **imitation is *not* required** (SS-MARL n=3→96, LEGO-MAPPO, SHPPO, EPC all show from-scratch
  permutation-equivariant policies that transfer across team size; LPAC's clairvoyant-imitation is *one*
  convenience path, not a necessity). De-risk with **(i)** degree / local-structure **regularization**
  (SizeShiftReg-style) — the guardrail against the GNN "bad-minima" overfit that breaks size-transfer when
  local degree drifts; **(ii)** an *optional* cheap CVT/Lloyd or greedy-frontier **behaviour-cloning
  warm-start → RL-finetune** as insurance, *not* a prerequisite; **(iii)** multi-scale training (Phase 4).
  **Do not** add E(n)-geometric canonicalization (refuted 0-3) or treat recurrence as mandatory (it's an
  *optional* Phase-1a variant — though our in-house GCRN belief already transferred 16²/4→32²/10, so it's a
  promising one).
- **Transfer precondition (Round-2):** GNN size-transfer is provably defensible **only** when the comm-graph
  **local degree stays ~invariant** across the ladder. Ours doesn't obviously hold it: density **drifts ~2×**
  (2/100 → 10/1024) and the **10²/2 rung is a degenerate 2-agent graph** — the known weak corner. **Phase 0
  measures per-rung comm-degree;** if it drifts, tune **comm_radius per rung** to hold degree ~constant (or
  accept the drift and lean on multi-scale-joint + degree-reg). *A precondition to verify, not an assumption.*
- **Foundations first (Phase 1):** the **network (type / size / hyperparameters) and the training algorithm
  are selected together, up front**, before any problem-specific phase. Phases 2–4 then run under the
  **locked (architecture, trainer, core-HP)** tuple — there is **no arbitrary "carrier" optimizer**.
- **Foundation base-task (user steer):** the real **coverage↔connectivity trade-off scored with the *local*
  metric** (local λ̂₂, not a global oracle) — *not* pure coverage. Credit **non-redundant / cooperative**
  coverage so the foundation is selected on a task that elicits coordination & division of labour (the
  emergence we care about), not just summed cells. Kept **fixed and simple** (a placeholder for net+trainer
  selection); the full reward design is swept in Phase 3. *Exact coordination term pending confirmation.*
- **Trainer re-confirm:** because the trainer is locked before the reward/mechanism are final, re-run a cheap
  trainer check after Phase 3 and re-open only if the champion is clearly dominated. **QD/MORL** (MAP-Elites
  / MORL) are judged on the **frontier**, not the scalar base-task, so they get a fair read.
- **Mission fixed:** 100-step horizon; scale ladder **10²/2 · 16²/4 · 24²/6 · 32²/10** (*approximately*
  fixed density — it actually drifts ~2×, see the transfer precondition above). Comm radius, sensing radius,
  and obstacle density **inherit the zymera `comm-coverage` recipe defaults** in **absolute cells**; per-rung
  we override grid size and `n_agents` — and **possibly comm_radius**, if the Phase-0 degree check says we
  must, to hold local degree ~invariant.
- **Carrier scale (Phases 1–3):** develop at **16²/4** (cheap), gate on **zero-shot transfer to 32²/10**.
  Phase 4 runs the full scale-strategy study.
- **Seeds:** **≥ 3** per config; report **mean ± std**. A gate "passes" only if the mean clears it with the
  band not straddling the prior phase's incumbent.

---

## 2. The agent under test

The swept knobs map onto the modules in `agent_architecture.md`:

| Module | What's fixed | What's swept (and where) |
|---|---|---|
| Perception (camera, local) | own-cell + sensing-radius mask | — |
| KB (memory: own + neighbors + priors) | size-invariant belief | recurrence is part of backbone-type (Phase 1a) |
| Comms (radio) | range-limited gossip | message type / gating (Phase 5, #2) |
| **L4 phase layer (NEW, central)** 🔜 | `{disperse↔gather}` option, committed `k≈5–10` steps | **phase layer + commit-k (Phase 2′)** |
| **Role-picker (central, learned)** | the hub, *within* the L4 phase | — (its inputs change with KB/safety) |
| Mission-safety (local) | consumes local λ̂₂; **enforcement only today** | mechanism (**Phase 2 ⛔ settled**); **wiring as a brain INPUT = open build item (Phase 2′)** |
| Goal (per role) | explorer / relay | exploration / **delivered-coverage** objective (**Phase 2′ / 3**) |
| Role tools | λ₂-estimator (always on) | frontier-attention tool (Phase 5, #3); toolkits (Phase 5, #8) |
| Reward (connectivity floor) | — | **Hyper-Singularity `barrier_*` — composes under L4 (Phase 2′)** 🔜 |
| Backbone (shared encoder) | LPAC-style graph-net | type/size/MP-rounds/norm (**Phase 1a**) |
| Trainer (the optimizer) | — | algorithm + core HPs (**Phase 1b**) |

---

## 3. Sweep axes (the full parameter space, organized)

| Axis | Values / range | Phase | Mode |
|---|---|---|---|
| Backbone type | {GNN (default), GAT-attention, recurrent-GNN} | 1a | OFAT |
| Backbone depth | {2, 3} | 1a | OFAT |
| Backbone width | {64, 128, 256} | 1a | OFAT |
| Message-passing rounds | {1, 2, 3} | 1a | OFAT |
| Normalization | {layer-norm (default), none} | 1a | OFAT |
| **Action representation** | {1-step move head (default), goal/frontier-pointer head (IR2)} | 1a′ | OFAT |
| **Training algorithm** | {IPPO, MAPPO-CTDE, ES, MAP-Elites, MORL} | 1b | OFAT |
| Trainer core HPs | per-trainer (lr / entropy / GAE-λ / clip · σ/pop · archive · scalarization) | 1b | OFAT / coarse |
| ~~**Connectivity mechanism**~~ ⛔ **SETTLED** | {action-mask, Lagrangian-PPO, degree-floor·λ, λ₂-soft·λ, **PID-Lagrangian**} | 2 | **DONE — fails at scale, do not re-run (§0)** |
| ~~Trade-off λ (soft mechs only)~~ ⛔ | {0.1, 0.3, 1, 3} | 2 | **DONE — dual λ stays flat at the huddle (§0)** |
| ~~Connectivity signal source~~ ⛔ | {global-λ₂, **local_edge_margin**} | 2 | **DONE — local signal doesn't rescue it (§0)** |
| **L4 phase layer** 🔜 | {off, scripted-timing, learned} | **2′** | OFAT (timing first) |
| **Phase commit k** 🔜 | {5, 8, 10} steps | **2′** | coarse |
| **Objective** 🔜 | {coverage, **delivered-coverage** (`PersistantNetwork`)} | **2′** | OFAT |
| **Barrier (conn-floor)** 🔜 | `barrier_weight` {0=off, >0} (+ `barrier_a/M/p/cap`) | **2′** | composes UNDER L4 (never alone) |
| Exploration reward | {extrinsic, PBRS, coordinated-intrinsic, max-state-entropy} | 3 | OFAT |
| Reward normalization | {raw, fractional/normalized} | 3 | OFAT |
| Scale strategy | {single-point, multi-scale-joint, warm-start-ladder} | 4 | OFAT |
| Idea ablations | {energy, ~~multi-phase~~ (→ promoted to 2′), comm-gating, frontier-attn, toolkits, diffusion} | 5 | +1 vs best |

The **coverage-only control point (~98 % cov / ~32 % conn)** is **not re-run** — it's the known incumbent
from prior results. **The Phase-2 connectivity-mechanism rows above are STRUCK THROUGH: tried & settled,
fail at scale, do not re-run (§0).** The active frontier is now **Phase 2′** (L4 phase + delivered-coverage +
barrier floor).

---

## 4. The phase ladder

### Phase 0 · Substrate + decentralized Fiedler estimator  *(build, not a sweep)*
- **Build:** the trainer-agnostic agent skeleton (8 modules as swappable callables), the **config-file
  schema** (§5), and the eval harness (logs coverage-%, connectivity-%, mean λ₂, |λ̂₂−λ₂|, redundancy).
- **Implement + validate** the decentralized local-Fiedler λ₂ estimator (power-iteration over the comm
  graph from local observation) against the sim's true λ₂ across all four rungs.
- **Measure the transfer precondition:** the **comm-graph degree distribution per rung**. If degree drifts
  across the ladder, choose a **comm_radius per rung** that holds it ~invariant — this is the precondition
  that makes the whole scale-invariance bet defensible (Round-2).
- **Gate G0:** substrate runs a 100-step episode at 32²/10; estimator error |λ̂₂−λ₂| small enough to use as
  a feature/reward (target: median relative error < ~15 % once power-iteration has mixed); **per-rung comm
  degree approximately matched** (or a comm_radius schedule chosen that makes it so). If the estimator can't
  track true λ₂ at the sparse 10²/2 end, or degree can't be held ~invariant, that's a finding — surface it
  before building on it.

### Phase 1 · Foundations: network + training algorithm  *(front-loaded; train 16²/4 → transfer 32²/10)*
Establishes the **(architecture, action-head, trainer, core-HP)** tuple everything downstream is built on.
Run on the **foundation base-task** (§1: local-metric coverage↔connectivity + non-redundant-coverage credit).
- **1a · Network** (OFAT from default: GNN, depth 3, width 128, MP-rounds 2, layer-norm; local λ̂₂ as an
  input feature). Sweep **type** {GNN, GAT-attention, recurrent-GNN}, **width** {64,128,256}, **MP-rounds**
  {1,2,3}, **depth** {2,3}, **norm** {layer,none}. ~8 configs. Lock the architecture that **transfers**.
  *(MARVEL, arXiv:2502.20217, validates a permutation-equivariant **graph-attention** backbone as
  scale-invariant across agent count @2/4/8 with no retraining — the concrete reference for the GAT variant.)*
- **1a′ · Action representation (NEW — IR2-style, the keystone axis).** Sweep **{1-step move head (flat,
  default) vs goal/frontier-pointer head}**. The pointer head = a **graph-attention encoder + pointer decoder
  that selects a candidate frontier/node to head for** (IR2, arXiv:2409.04730), with a **heuristic A\*/greedy
  low-level controller** emitting the actual 1-step moves — the experiment owns this L3-goal→L1-move
  translation, so the sim still sees movement-only actions and the **100-step budget is unchanged** (the
  controller's moves count). Directly tests our own flagged keystone — the **1-step move head is the suspected
  coverage ceiling** ([[marl-action-representation-bottleneck]]) — now corroborated by IR2/MARVEL. ~+2 configs.
  *Build cost: candidate-frontier extraction on the grid + pointer head + A\* controller; horizon semantics
  (re-plan every k moves vs on goal-reach) locked at Phase 0/1.*
- **1b · Algorithm** (under the locked architecture). Sweep **trainer** {IPPO, MAPPO-CTDE, ES, MAP-Elites,
  MORL} × **per-trainer core HPs** (PPO: lr / entropy / GAE-λ / clip · ES: σ / population · QD: archive /
  behavior-descriptor · MORL: scalarization). OFAT / coarse, ~10 configs. QD/MORL judged on the **frontier**,
  not the scalar base-task.
- **Re-check:** if 1b's winning trainer would change the 1a architecture pick, do one quick loop.
- **Gate G1:** a locked (architecture, trainer, core-HP) foundation that **learns the base task and
  transfers across scale** (32²/10 zero-shot within a small margin of 16²/4). *If no architecture transfers,
  the LPAC / size-invariance bet is wrong — stop before Phases 2–5.*

### Phase 2 · Connectivity mechanism + trade-off frontier  ⛔ **TRIED & SETTLED — FAILS AT SCALE, DO NOT RE-RUN**
> **STATUS (2026-06-27): CLOSED as a coverage fix.** This whole phase was run (see §0 ledger, rows 2–3) and
> **the connectivity mechanism is structurally inert at scale** — at the huddle degree ≫ target ⇒ penalty
> ≈ 0 ⇒ dual λ flat (`0.30→0.30`) ⇒ **the huddle SATISFIES the connectivity signal and the mechanism cannot
> push the team apart.** action-mask / soft-λ₂ / degree-floor / Lagrangian / PID **all** "hold" connectivity
> only by **huddling** (mean λ₂ *rises* with scale), while coverage collapses **90.9 % → 53 % → 16 %**.
> **Do NOT re-run mechanism × λ at scale hoping it fixes coverage.** Connectivity is now handled by the L4
> phase rhythm + a silent barrier floor (**Phase 2′**), not a per-step penalty. *Retained below for the
> record + the original hypothesis it falsified.*
- **Original hypothesis (FALSIFIED at scale):** a connectivity mechanism driven by **local λ̂₂** lifts the
  ~32 % connectivity of pure coverage toward 90 % without crashing coverage. **Round-3 corroborated the
  *connectivity* half** — Li et al. (ICRA'22, arXiv:2109.08536) hold **71–77 % connectivity** with an
  explicit **CMDP constraint on the same λ₂ signal** while an unconstrained baseline collapses to **7–30 %**.
  **What our runs add:** holding connectivity was *never* the problem on the grid — **huddling holds it for
  free**; the unsolved half is **coverage**, and a per-step connectivity constraint *cannot* recover it
  (the huddle is feasible). Li et al. is multi-robot *navigation*, where clumping isn't a free connectivity
  solution; on dense-grid coverage it is.
- **Swept (DONE, do not repeat):** {action-mask (no λ), Lagrangian-PPO (auto-λ), **PID-Lagrangian**
  (arXiv:2007.03964), degree-floor / **`local_edge_margin`** × λ} × {24²/6, 32²/10}. PID-Lagrangian *did*
  pre-empt the in-place-oscillation failure mode — but with the dual λ flat there was nothing to control.
- **Gate G2 — RESULT: FAILED.** No mechanism reaches 90/90 @ 32²/10; best is **~16 % coverage** (PID at 32²,
  ties baseline). The expected "hard-mask-first + (PID-)Lagrangian backstop" lead **did not materialize at
  scale** — the 16²/4-only advantage of mask is the 2nd-order effect from §0 row 1, which **vanishes** once
  the huddle dominates. ⇒ **proceed to Phase 2′, not deeper into this phase.**

### Phase 2′ · 🔜 NEW DIRECTION — L4 phase layer + delivered-coverage + barrier floor  *(the pivot; 2026-06-27)*
> **⚠️ NOW CONDITIONAL on §0′ (2026-06-27).** Per the consolidated `STRATEGY.md` verdict, the **L4 phase head
> is a hand-built brain level the evidence does not support** (hierarchy's benefit is exploration, not the
> tower — Nachum et al. 2019, arXiv:1909.10618; phases EMERGE — Couzin et al. 2002). **Run the §0′
> flat-baseline falsification test FIRST.** The **delivered-coverage objective and the barrier floor below
> survive** (they make the rhythm *emerge* with no per-step penalty); the **explicit L4 phase head / discrete
> skill library is only built if §0′ shows the rhythm does NOT emerge on its own.** Read this phase as the
> contingency, not the default next step.

**The lesson from Phase 2: stop fighting coverage↔connectivity per step — resolve it in TIME.** The team
should **fan out to cover, regroup to share, repeat** — hold connectivity *periodically*, not every step, so
the huddle stops being the optimum. Three composing pieces:

- **(a) L4 brain layer — a strategy / mission-phase layer ABOVE the L3 role-picker (NEW top level).** L4
  picks the **team PHASE `{disperse ↔ gather}`** (≡ explore ↔ deliver) as a **temporally-extended option —
  committed for `k ≈ 5–10` steps, NOT per step.** L3 picks the **role** within the phase, L2 the **skill**,
  L1 the **move** (the cognition stack is now **L4→L3→L2→L1**; see `agent_architecture.md`). Decentralized
  per-agent, **cohering via the shared belief** — this is the **micro→macro bridge made an explicit module.**
  **BUILD STAGED:** add the L4 phase head **on top of the existing role-picker**, keep gather/disperse
  **SKILLS scripted FIRST (learn only the *timing* — the switch), grow learned-ness later.** **Do NOT train
  4 learned levels at once.** *(This is idea #7 "multi-phase cycle" promoted from a Phase-5 carry-over to the
  spine, and the temporally-extended-option fix for the in-place-oscillation / huddle failure mode.)*
- **(b) Delivered-coverage objective — `PersistantNetwork` (already in the codebase).** Coverage counts
  **only when in contact to share it.** This makes the **disperse→gather rhythm EMERGE from the objective,
  with NO connectivity penalty** — clumping no longer scores (nothing new is delivered), and
  spreading-without-returning no longer scores (nothing gets home). The cleanest huddle-killer: the reward
  *itself* wants the breathing, so there is no per-step constraint for the team to satisfy-by-clumping.
- **(c) Hyper-Singularity barrier reward — a SILENT connectivity FLOOR** (`reward.barrier_*`, config-knobbed,
  being built now). Per-agent on nearest-neighbour distance, `f(x)=k·relu(x−a)²/(M−x)^p` **CAPPED finite
  (RL-safe)**: **exactly 0 in the safe zone (`x < a`), an explosive-but-finite wall as a link nears the comm
  edge `M`**, saturating at `barrier_cap`. Knobs: `barrier_weight=k` (0 ⇒ OFF), `barrier_a` (launch; default
  `comm_r·0.6`), `barrier_M` (wall; default `comm_r`), `barrier_p`, `barrier_cap`. **It COMPOSES under the L4
  breathing** as a floor the team can ride *out* to — **NOT a per-step pull inward — and is NEVER tested in
  isolation** (alone it is itself a per-step signal and would re-huddle, exactly per §0).

- **Mission-safety-as-brain-INPUT — OPEN BUILD ITEM.** Today `MissionSafety` is an **enforcement** mechanism
  (action-mask / reward-penalty); it is **NOT wired as an explicit INPUT** to the role/phase brain — the role
  head in `nets.py` conditions only on the belief `z`. `agent_architecture.md` *intends* mission-safety as an
  input the role-picker ingests. **The L4/L3 brain must READ the connectivity-danger signal (λ̂₂ / barrier
  proximity) to time gather vs disperse.** Close this before L4 can react to danger rather than only enforce.
- **Hypothesis (2′):** an L4 disperse↔gather rhythm + delivered-coverage **breaks the scale huddle that no
  per-step mechanism could** — coverage rises at 24²/6 and 32²/10 while connectivity is held *intermittently*
  (and floored by the barrier), *without* a per-step connectivity penalty.
- **Build/sweep order:** (1) scripted skills + **learned phase TIMING** only; (2) `PersistantNetwork`
  delivered-coverage as the base-task, **no conn penalty**; (3) compose the **barrier underneath**
  (`barrier_weight > 0`); (4) wire **mission-safety as an INPUT** to L4/L3; (5) grow learned-ness down the
  stack one level at a time. New axes: **`phase_layer {off, scripted_timing, learned}`**, **phase commit
  `k`**, **`objective {coverage, delivered_coverage}`**, **`barrier_weight`** (+ `barrier_a/M/p/cap`).
- **Gate G2′:** the L4+delivered configuration whose coverage **rises with scale at 24²/6 and 32²/10** vs the
  Phase-2 ceiling (≤16 % @32²) while connectivity stays high intermittently — *with no per-step connectivity
  penalty.* If even the phase rhythm + delivered-coverage cannot lift scale coverage, that is a deeper finding
  about the action/belief substrate (revisit the keystone goal-head / belief-plug-in), not another mechanism.

### Phase 3 · Exploration reward + scale-invariance
- **Hypothesis:** a **scale-invariant, coordinated intrinsic** coverage reward holds 90/90 *and* survives
  warm-start; tabular/count rewards break it.
- **Sweep:** {extrinsic, PBRS, coordinated-intrinsic, max-state-entropy} × {raw, normalized}. ~8 configs.
- **Gate G3:** the reward that holds the Phase-2 frontier **and** transfers across scale (normalized beats
  raw on transfer, per the lit). **Then run the cheap trainer re-confirm** (§1) before Phase 4.

### Phase 4 · Scale strategy  *(recipe set by Round-2; this phase confirms it)*
- **Resolved recommendation (Round-2):** **multi-scale JOINT sampling across the ladder is the spine**, with
  **curriculum finetune at the 32²/10 target** — *not* a small-only warm-start hop (the 10²/2 end is the
  degenerate 2-agent weak corner you'd over-anchor to). Up-weight sampling at **both** the sparse-small and
  hard-large ends.
- **Sweep (confirm, don't re-derive):** {single-point @32²/10 from scratch, **multi-scale-joint (default)**,
  warm-start-ladder 10→16→24→32}. 3 regimes — multi-scale-joint is the expected winner; the other two are
  the controls that show *why*.
- **Gate G4:** the regime with the best 90/90 @ 32²/10 + sample-efficiency (expected: multi-scale-joint +
  target-finetune). *Medium-confidence call — never tested head-to-head on a coverage task, so we verify.*

### Phase 5 · Idea ablations  *(ranked & resolved by Round-3 — targeted +1 vs the best G4 agent)*
- **Slot 1 (keystone) — connectivity-aware explorer tool (idea #3).** IR2-style (arXiv:2409.04730)
  **non-myopic, λ̂₂-biased frontier attention** as the explorer tool vs the base policy; directly closes the
  in-house **98 %/32 %** gap. Cells: {connectivity-aware attention vs plain} (the mask/Lagrangian/PID
  machinery already lives in Phase 2). *Borrow IR2's learned explore-vs-relay trade-off, but keep
  **MAINTAIN ≥90 %**, not intermittent rendezvous-disconnect.* **Presupposes the goal/frontier-pointer action
  head (Phase 1a′)** — slot 1 is the λ̂₂-biased version of that head; if the flat move-head wins Phase 1a′,
  reconsider.
- **Slot 2 — explorer/relay role-picker (idea #8).** {fixed typed roles vs homogeneous} × {role
  **embeddings** (ROMA-style) vs **restricted action-subsets** (RODE-style "role calls its tool")}, layered
  on the G4 backbone; measure **role emergence** (MI(role;behaviour) / labour-division redundancy). **Roles
  as embeddings/action-subsets ON the permutation-equivariant backbone — NOT a QMIX-mixer or pre-fixed-K
  cluster head (both break scale-invariance; GraphMIX arXiv:2010.04740).** Option/skill *discovery* is a
  later test, not v1.
- **Deferred / conditional — energy-as-effort-cost (idea #1).** **DEFER until 90/90 is reached.** No evidence
  it induces role-specialization (all sources were battery/recharge or classical Voronoi — the excluded
  framing); expect it to *lower* 90/90 → a single efficiency-frontier cell {action-cost on/off} with
  **per-AREA (not per-agent) normalization** to protect scale-invariance; a stress test, not an enabler.
- **Dropped — multi-agent diffusion (idea #11).** No slot: per-step inference conflicts with the 100-step
  budget and no permutation-equivariant size-invariant graph-diffusion evidence survived. Revisit only if
  amortized **once-per-episode** graph-diffusion for variable agent count is demonstrated (→ a relay
  trajectory tool, not the per-step policy).
- **Lower-priority carry-overs (Round-1/2):** +multi-phase cycle (#7), +learned comm-gating (#2) — run only
  if slots 1–2 leave headroom.

---

## 5. Config schema (every run carries one)

```yaml
# experiments/<id>.yaml — exact, reproducible, one per run
id: p2_mask_l0                 # unique run id
phase: 2                       # which phase
hypothesis: "mask lifts connectivity toward 90 without crashing coverage"

world:                         # zymera comm-coverage recipe + per-rung overrides
  recipe: comm-coverage
  grid: [16, 16]               # swept across the ladder in Phase 4
  n_agents: 4
  comm_radius: null            # null = recipe default (absolute cells)
  sensing_radius: null
  obstacle_density: null
  horizon: 100

agent:                         # the composition under test (see agent_architecture.md)
  backbone: {type: gnn, depth: 3, width: 128, mp_rounds: 2, norm: layer}  # SWEPT in Phase 1a
  action_head: {kind: move, controller: null}     # SWEPT in Phase 1a′: move | goal_pointer (+ controller: astar)
  comms: {type: gossip, gating: none}
  phase_layer: {kind: off, commit_k: 8}         # SWEPT in Phase 2′: off | scripted_timing | learned (L4 {disperse,gather})
  role_picker: central_learned
  tools: {explorer: frontier_default, relay: fiedler_local}
  mission_safety:                               # mechanism = ENFORCEMENT (Phase 2, ⛔ settled at scale)
    mechanism: action_mask                      #   action_mask | soft_lambda | lagrangian | pid_lagrangian
    conn_signal: global_lambda2                 #   global_lambda2 | local_edge_margin  (⛔ both settled, §0)
    as_brain_input: false                       # OPEN (Phase 2′): feed λ̂₂/barrier-proximity into L4/L3
  reward:                                        # SWEPT in Phase 3 + Phase 2′
    kind: extrinsic
    objective: coverage                         # SWEPT in Phase 2′: coverage | delivered_coverage (PersistantNetwork)
    normalized: false
    barrier_weight: 0.0                          # SWEPT in Phase 2′: Hyper-Singularity conn-floor; 0 = off (composes UNDER L4)
    barrier_a: null                              #   null -> comm_r*0.6   (launch; 0 below it)
    barrier_M: null                              #   null -> comm_r       (wall / break range)
    barrier_p: 2.0                               #   explosion power
    barrier_cap: 50.0                            #   finite "almost-infinity" ceiling (RL-safe)

connectivity:                  # the locked metric + agent signal
  estimator: fiedler_local_poweriter
  grade_on: true_lambda2
  connected_threshold: 1.0e-3
  trade_off_lambda: null       # SWEPT in Phase 2 (soft mechs only)

trainer: {kind: ippo, lr: 3.0e-4, ...}          # SWEPT in Phase 1b
scale_strategy: single_point                    # SWEPT in Phase 4
warm_start_from: null                           # a prior run id, in Phase 4

seeds: [0, 1, 2]
```

## 6. Eval protocol

- **Horizon:** 100 steps. **Seeds:** ≥3; report mean ± std on **held-out eval seeds** (distinct from
  training seeds).
- **Metrics logged every run:** coverage-%, connectivity-% (steps with λ₂ > τ), mean λ₂, |λ̂₂ − λ₂|,
  redundancy, delivered-coverage (if the relay variant is active).
- **Transfer-eval:** Phases 1–3 develop at 16²/4 and report **zero-shot 32²/10** as the headline + gate.
- **Gate pass:** the mean clears the stated bar and the ±std band does not straddle the prior incumbent.

## 7. Compute envelope

~45 configs × 3 seeds ≈ **~135 core training runs** + cheap transfer-evals. Phase 1 spans ~2 overnight
batches (1a network, 1b algorithm); Phases 2–5 ≈ one batch each → **~6–7 nights** on the GPU harness. Gates
between phases mean a failed phase stops spend rather than feeding four more rungs of wasted runs.

## 8. Open-idea status — RESOLVED (Round-3)

Round-3 (narrowed to robotics/MARL) confirmed **24/25 claims** and **finalizes the Phase-5 matrix** (above).
The headline is bigger than the ablation ranking: **the robotics/MARL literature independently validates the
whole spine.**
- **Connectivity-aware learned exploration (idea #3) is the keystone** — Li et al. (arXiv:2109.08536)
  constrain the *same* decentralized λ₂ signal we adopted; MARVEL (arXiv:2502.20217) proves the
  graph-attention backbone is scale-invariant; IR2 (arXiv:2409.04730) proves the trade-off is *learnable*; a
  HAPPO explorer (arXiv:2412.20049) reproduces our 98/32 gap. **Brought forward into the spine:**
  PID-Lagrangian (Phase 2) + MARVEL backbone reference (Phase 1).
- **Explorer/relay roles (idea #8)** improve performance + convergence and can emerge (RODE/ROMA/ACORM) —
  adopt the *idea* (embeddings / action-subsets) but **not** their QMIX-mixer / pre-fixed-K (break
  scale-invariance) → Phase-5 slot 2.
- **Energy (idea #1) deferred · diffusion (idea #11) dropped** — see Phase 5.
- **Multi-phase cycle (idea #7) PROMOTED** (2026-06-27) from a Phase-5 carry-over to the **spine as Phase 2′**
  — the L4 disperse↔gather layer is now the primary lever after per-step guardrails settled (§0).
- **Cross-round note:** Li et al. needed **behaviour-cloning** to make the λ₂-constrained two-objective
  tractable → reinforces keeping our *optional* cheap-expert BC warm-start (Round-2) as insurance for the
  constrained case. Open: can hard-mask + PID-Lagrangian remove that need on a grid (we train from scratch)?
  *(Moot for the connectivity fix — that path is settled-closed; relevant only if Phase 1's constrained
  base-task needs it.)*

**Plan status (2026-06-27): Phases 0–1 FINAL · Phase 2 SETTLED-CLOSED (per-step guardrails fail at scale,
§0) · Phase 2′ NEW & ACTIVE (L4 phase + delivered-coverage + barrier) · Phases 3–5 unchanged.**

## 9. Open / to resolve before the relevant phase

- **Foundation base-task** — local-metric coverage↔connectivity trade-off + a *non-redundant/cooperative*
  coverage credit (user steer); confirm the exact coordination term (anti-overlap vs explicit
  division-of-labour) before Phase 1.
- **comm_radius schedule** — fixed-absolute vs per-rung to hold comm-degree ~invariant (decide at Phase 0
  from the degree measurement; precondition for size-transfer).
- **Mission-safety enforcement** beyond the Phase-2 mechanisms (its own discussion) — consumes local λ̂₂.
  **⛔ As a *coverage-recovery* mechanism this is SETTLED-CLOSED (§0).** **OPEN, NEW (Phase 2′):** wire
  mission-safety as an explicit **INPUT to the L4/L3 brain** (λ̂₂ / barrier-proximity → the phase/role head,
  which today reads only the belief `z`) so the brain *times* gather/disperse on connectivity danger.
- **L4 phase layer (Phase 2′, NEW)** — scripted-skills-first vs learned skills; phase-commit `k` (5/8/10);
  the categorical `{disperse, gather}` head and its commit/timeout logic; whether timing is learned by PPO or
  event-triggered on λ̂₂/barrier proximity.
- **Hyper-Singularity barrier (Phase 2′, NEW)** — `barrier_a/M/p/cap` defaults vs swept; confirm it stays a
  *silent floor under L4* (never run alone — it re-huddles, being a per-step signal).
- **Delivered-coverage objective (Phase 2′, NEW)** — reuse `PersistantNetwork`; lock the "in-contact-to-count"
  delivery rule + whether it fully replaces the connectivity reward term (intended: yes, no conn penalty).
- **Sensing_radius / obstacle_density** — inherit `comm-coverage` recipe; confirm when the substrate is built
  (Phase 0).
- Whether to also **grade on the local estimate** (purist decentralized framing) vs the true-λ₂ grader
  (default; non-gameable).
- **Phase-5 ablation matrix** — RESOLVED by Round-3 (slot 1 connectivity-aware explorer · slot 2 role-picker
  · energy deferred · diffusion dropped).
- **Goal-pointer head (Phase 1a′, NEW)** — heuristic A* vs learned low-level executor (default A*); goal
  re-plan cadence (every k moves vs on goal-reach); candidate-frontier extraction on the grid. Lock at
  Phase 0/1.
