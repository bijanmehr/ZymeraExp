# SuperBlue · Shared-Exploration — Research Journal

A dated, detailed log of the campaign: **decisions, experiments, results, what went well/badly, changes,
and next steps** — one entry per working day. Newest at the bottom. Pairs with `../design/T5_DESIGN.md`
(the current design) and the per-experiment configs (below). *(The June `agent_architecture.md` /
`LOCALITY_DESIGN.md` design docs were folded into the architecture-decision-process entry below and removed.)*

> **Status (2026-08-08).** This journal now spans **June → August 2026**. The arc evolved well past the
> original "SuperBlue coverage" framing: June's coverage campaign → July's honest fixed-spec **T1–T5**
> re-run and the **T5 planner** pivot → August's planner validation, the **KB audit**, missions #2/#3, and
> the workspace consolidation (all code → `zymera_lab/experiments/`, results-only experiment folder, GPU
> machines wiped after a full backup). `CAMPAIGN_REVIEW.md` has been **merged into this file** (the
> consolidated June results table + caveats, below). Precise per-finding detail lives in the memory corpus;
> this is the narrative spine.

**Target.** A "superblue" TeamBlue agent that reaches **≥ 90% coverage AND ≥ 90% connectivity**
simultaneously on shared exploration, and pushes the coverage↔connectivity frontier as far as possible.

**Fixed mission setup.**
- **Horizon:** **100 timesteps** per episode (a hard mission requirement).
- **Scale ladder (fixed agent density):** 10×10 → **2** agents · 16×16 → **4** · 24×24 → **6** ·
  **32×32 → 10** (the real push).
- **Warm-start (must validate):** train small, then **initialize the next rung from the previous** up the
  ladder. Requires the agent to be **scale-invariant by construction** (same params across grid size *and*
  agent count — local/relative perception, size-invariant belief, agent-count-invariant aggregation).
  Study: does it transfer and help reach 90/90 at 32²/10 (vs from-scratch / zero-shot)?
- **Methods (try all):** PPO (IPPO / MAPPO+CTDE), ES, **Quality-Diversity (MAP-Elites)**, and
  **multi-objective / constrained RL** — all driving the *same* trainer-agnostic agent.

**Conventions.**
- **Every experiment is documented by a config file** (text/YAML/JSON) capturing its **exact** parameters
  — scale, n_agents, comm_r, horizon, the agent composition (KB / comm / roles / per-role tools /
  mission-safety), trainer + hyperparams, trade-off λ, seed, and any `warm_start_from`. Nothing runs
  without a saved config so any result is exactly reproducible.
- **Each journal entry references the config(s) used** and the metrics out (coverage %, connectivity %
  — fraction-of-steps-connected / giant-component / λ₂ — redundancy, delivered-coverage if used).
- Metric definition (to lock): **coverage %** = covered free cells / free cells at t=100; **connectivity %**
  = TBD exact estimator (fraction of steps fully connected vs giant-component fraction) — *decide before
  first run.*

---

## Prologue — lead-up (context)

- **2026-06-25 (earlier):** Consolidated the codebase: `kymera` → renamed to the **`zymera`** package in the
  **`zymera_lab/`** repo; `zymera_env` archived as read-only reference (tag `archive/pre-zymera-lab-fold`).
  Lab P1 done (sim + `nets.py`/`train.py`/`sensor.py` seeds, 221 tests green). Experiments live separately
  in `zymera_experiments/` (`TeamBlue/SharedExploration`, `TeamBlue/PersistantNetwork`).

---

## 2026-06-25 — agent architecture + campaign framing (design phase)

**Decisions.**
- **Agent decomposition (8 modules):** perception (own camera, sensing-radius, opt. EKF fusion) · KB
  (memory fusing own-perception + neighbor-comms + priors) · comms (radio, ping-pong; stack = simple /
  composition / learnable / λ₂) · **role picker (central, learned)** · mission-safety (local) · goal
  (per role) · role tools (heuristic **or** learned, e.g. frontier-attention / λ₂-estimator) ·
  operation-action (scan/mine the current cell).
- **Multi-level cognition (brain-like):** L3 deliberative (goal/role-picker/KB) → L2 executive (tool +
  planning) → L1 reactive (movement/collision). Higher levels set intent and **delegate — no
  micromanagement**; each runs at its own timescale.
- **Boundary:** `zymera` runs the **world only** (obs+state+raw signals; reward-agnostic). The experiment
  owns the **entire agent**; **reward engineering is fully separate**, here.
- **Trainer-agnostic + big run plan:** the agent is a configurable composition of swappable strategy
  modules; the *same* agent is driven by different learners and compared in a large sweep.
- **Learning methods — try ALL:** PPO (IPPO / MAPPO+CTDE), ES, **Quality-Diversity (MAP-Elites)**,
  multi-objective / constrained RL. (Reframed from "PPO vs ES": the structural choice is **HRL**; the
  optimizer is a separate, swept axis; QD/MORL added because the goal is a *frontier*, which they produce
  natively.)
- **Campaign target + setup fixed:** **90 / 90** coverage/connectivity; **100-step** horizon; the scale
  ladder above; **warm-start study**; **per-experiment config files**.

**Experiments run.** None yet — design phase.

**Good / bad.** Good: architecture and boundary are clear and modular; a clean trainer-agnostic plan.
Open risks: scope is large (many axes × scales × methods); two key knobs still parked.

**Changes.** Created this journal; `agent_architecture.md` updated with the campaign params + the
scale-invariance constraint. Pitched **10 ideas to verify** → captured in `IDEAS.md` (energy tokens ·
learnable GNN comms · frontier-attention · CBF/Lagrangian-MARL · physics-inspired / LPAC · network
fundamentals · multi-phase mission cycle · per-role tools · reward-eng / intrinsic · graph embeddings),
with a 4-theme lit-review plan. **Launched a deep-research lit-review across all 11 ideas** (critic lens,
in-field + off-field, ADOPT/TEST/SKIP verdicts + cited report; run `wf_2b07f581-6cc`) — results pending.

**Parked / open.** Mission-safety method (own discussion); trade-off scheme (decide after first slice);
exact connectivity-% estimator (decide before first run).

**Next.**
1. Lock the **connectivity-% metric** and the **eval protocol** (so 90/90 is unambiguous).
2. Resolve the **parked items** (mission-safety method; trade-off scheme).
3. Design the **run-plan matrix** (which axes sweep vs fix; the first concrete set, incl. the warm-start
   curriculum and a heuristic-only reference point).
4. Build the **trainer-agnostic agent skeleton** + the **config-file format**, then a first heuristic
   reference run to validate the 100-step / scale-ladder substrate.

---

## 2026-06-25 (cont.) — deep-research lit-review: results (run `wf_2b07f581-6cc`)

**Effort.** 106 agents · ~2.7M tokens · 24 sources fetched · 116 claims → **25 verified (19 confirmed,
6 killed)**. No single paper does the full SuperBlue conjunction → the recommended path is an **assembled
architecture (medium-confidence synthesis)**.

**Verdicts (covered ideas).**
- **ADOPT — LPAC graph-net backbone** (CNN local-perception + permutation-equivariant GNN comms + MLP head;
  arXiv:2401.04855, T-RO 2025) for **size-invariance + coverage + learnable comms**. One policy trained at
  32 robots transfers **zero-shot 8→128 robots / up to 2048²**. Use as the shared encoder / tool substrate
  — **do NOT inherit its pure-coverage objective**.
- **ADOPT — decentralized Fiedler/λ₂** (Yang 2010, Automatica): estimate λ₂ via decentralized
  power-iteration (**O(1) state, cost ∝ local degree, no team-size-n dependence**) + λ₂-gradient control.
  **Replaces our exact-eigendecomposition "Fiedler oracle."** Scale-invariant, fits the 100-step budget.
  *Soft, not hard; degrades when the eigengap is small.*
- **TEST — connectivity constraint on-grid:** hard **action-mask** (forbid disconnecting moves — matches
  our prior guardrail result) OR soft **Lagrangian-PPO**. **SKIP the formal hard-CBF guarantee as-is**
  (λ₂-as-CBF proven for continuous-control / global-Laplacian *centralized* settings).
- **TEST — coordinated, scale-invariant intrinsic coverage reward** (max-state-entropy is reward-free,
  Hazan ICML'19; coordinate novelty across teammates, Iqbal&Sha NeurIPS'19). **SKIP tabular count-based**
  (not parameter-share scale-invariant → breaks warm-start).
- **TEST — multi-phase / recurrent-connectivity** (IR2, IROS'24) as a time-multiplexing scaffold — but it
  optimizes *time-to-coverage*, not a 90/90 target, and *assumes known relative positions* (we can't); don't
  expect 90/90 from it alone.

**Verified unifications.** graph-embedding ↔ Fiedler/λ₂ ↔ connectivity (#10≡#4 — *one spectral story*) ·
intrinsic info-gain ≡ the shared-exploration objective (#9) · the CBF/Lagrangian trade-off is realizable
through a graph net (primal-dual LPAC) · LPAC unifies scale-inv+coverage+comms but **NOT connectivity**.

**Uncomfortable truths (design-changing).**
1. **LPAC's zero-shot scale-invariance comes from IMITATING a clairvoyant centralized expert — not
   from-scratch RL.** Conflicts with our trainer-agnostic goal → open Q: can the GNN backbone be RL-trained
   from scratch (or imitate a CVT heuristic, then RL-finetune) without losing scale transfer?
2. **Scale-invariance is NOT free and is WORST at the small/sparse end** (LPAC −9% at 8 robots) — exactly
   where the 10×10/2 warm-start *starts*; the "symmetric scaling" claim was **REFUTED 0-3** → **train
   multi-scale across the ladder, not single-point-then-transfer.**
3. Two SuperBlue-favored ambitions were **REFUTED**: max-ent optimal policy is a *mixture*, not one
   stationary policy (0-3); a learned modality/role-**selector** matching a hand-picked oracle (1-2) →
   **temper optimism about a learned central role-picker beating scripted**, and about one-action-per-step
   reward-agnostic exploration.

**OPEN — no verified claims, need round 2.** #1 energy tokens · #3 connectivity-respecting
frontier-attention · #8 per-role toolkits / option discovery · #11 diffusion (belief-completion / Diffuser
/ MADiff + inference-cost vs 100-step budget). *Treat as OPEN, not skipped.*

**Changes.** Verdicts table added to `IDEAS.md`. Architecture implication: backbone = LPAC-style GNN;
mission-safety/relay = decentralized Fiedler; trade-off = mask/Lagrangian; explorer reward = scale-invariant
intrinsic. The exact-eigendecomp Fiedler oracle is **replaced**.

**Next.** (1) Round-2 deep-research on #1/#3/#8/#11. (2) Resolve: can the LPAC backbone be **RL-trained from
scratch** (vs imitation)? (3) Lock the connectivity-% metric. (4) Decide **multi-scale-ladder vs
warm-start-transfer** training. (5) Then the run-plan matrix.

**Sources (primary).** LPAC arXiv:2401.04855 + constrained-LPAC arXiv:2409.11311 · Fiedler estimator/control
Yang 2010 (Automatica 46(2)) · λ₂-as-CBF Automatica 156:111209 (2023) · MADDPG-CBF arXiv:2103.12553 ·
max-ent exploration arXiv:1812.02690 · coordinated MARL exploration arXiv:1905.12127 · IR2 arXiv:2409.04730.

---

## 2026-06-25 (cont.) — experiment plan locked → `EXPERIMENT_PLAN.md`

**Decisions.**
- **Methodology = staged + gated OFAT** (one axis at a time from a moving baseline, go/no-go gate per
  phase), *not* a full-factorial sweep. Rejected the literal "sweep all params on all ideas" (~10⁶ runs,
  confounded). User cuts: **no heuristic-policy baselines · no MLP backbone arm · no within-phase grids** →
  sweep shrank from ~230 to **~45 configs × 3 seeds ≈ ~135 core runs**, ~5–7 overnight batches.
- **Connectivity metric LOCKED = the Fiedler value λ₂, estimated decentrally from local observation**
  (Yang-2010 power-iteration; *kills the exact-eigendecomp Fiedler oracle*). Split use: agent consumes its
  **local λ̂₂ᵢ** (feature/reward/safety); we **grade on the sim's true λ₂** with
  **connectivity-% = fraction of 100 steps with λ₂ > 1e-3**, and log **|λ̂₂−λ₂|** so the agent can't Goodhart
  its own estimate. coverage-% = covered/free at t=100. (Resolves the parked "connectivity-% estimator.")
- **Backbone committed = LPAC-style graph-net** (no MLP comparison). **Foundations first** (user reorder,
  fixes the arbitrary-carrier weakness): the **network *and* the training algorithm are selected together up
  front** (Phase 1: 1a network, 1b algorithm), on a base-task = coverage + a fixed light λ₂ term; Phases 2–4
  run under the **locked (net, trainer, HP)** tuple — no carrier optimizer. Guards: trainer **re-confirm**
  after Phase 3; **QD/MORL judged on the frontier**, not the scalar base-task. **Carrier scale = train
  16²/4, gate on zero-shot transfer to 32²/10**; scale-strategy study is Phase 4.
- **Phase ladder (0→5):** 0 substrate + Fiedler-estimator build/validate · **1 Foundations — 1a network
  (type/size/MP-rounds/norm) + 1b training algorithm (IPPO/MAPPO-CTDE/ES/MAP-Elites/MORL) + core HPs** ·
  2 connectivity-mechanism × λ frontier (mask / Lagrangian / degree-floor / λ₂-soft) · 3 exploration reward ×
  normalization · 4 scale strategy (single / multi-scale / warm-start) · 5 idea ablations
  (+energy/+multi-phase/+comm-gating/+frontier-attn/+toolkits/+diffusion). Each phase carries a **hypothesis
  + gate**; the ~98/32 coverage-only point is the known incumbent, *not* re-run.

**Good / bad.** Good: the metric is now one quantity that unifies measure + agent-signal + idea #10≡#4; the
plan is comprehensive (all 11 ideas appear) yet tractable and gated. Risk: the local λ̂₂ estimator must
actually track true λ₂ at the sparse 10²/2 end (Gate G0) — if it can't, it can't drive rewards there.

**Changes.** Wrote `EXPERIMENT_PLAN.md` (locked decisions · sweep-axis table · phase ladder with gates ·
config schema · eval protocol · compute envelope · round-2 dependency). Memory updated (estimator locked).

**Parked / open.** Mission-safety enforcement beyond Phase-2 mechanisms · exact comm/sensing/obstacle
numbers (inherit `comm-coverage`, confirm at Phase 0) · whether to *also* grade on the local estimate ·
**Round-2 deep-research on #1/#3/#8/#11** before their Phase-6 ablations.

**Next.** (1) User reviews `EXPERIMENT_PLAN.md`. (2) Then → writing-plans on the **Phase-0 slice** (agent
skeleton + config schema + decentralized Fiedler estimator + eval harness). (3) Optionally launch Round-2
research in parallel so #1/#3/#8/#11 are specified by the time Phase 5 arrives.

---

## 2026-06-25 (cont.) — Round-2 deep-research: results (run `wf_45751865-0aa`)

**Effort.** 107 agents · ~3.3M tokens · 25 sources · 119 claims → **25 verified (16 confirmed, 9 killed)**.
**Twist:** all 16 confirmed claims bear on the make-or-break methodological gate (#5); the verification round
surfaced **zero confirmed claims on the four Phase-5 ideas** (#1 energy / #3 frontier-attention / #8 toolkits
/ #11 diffusion) — so those stay **OPEN, not SKIP** (absence of evidence; the report is explicit on this).

**#5 — RESOLVED DECISIVELY → from-scratch RL is viable; imitation NOT required.** Four independent primary
sources show permutation-equivariant policies trained **from scratch** (no clairvoyant expert) that transfer
zero-shot across team size: **SS-MARL** (arXiv:2501.13727, n=3→96 >90 %), **LEGO-MAPPO** (arXiv:2509.14431,
4→{2,3,5,6} graceful), **SHPPO** (arXiv:2404.03869), **EPC** (arXiv:2003.10423). LPAC's
imitation-of-clairvoyant-CVT (arXiv:2401.04855) is *one* convenience path (faster convergence), not a
necessity. **Recipe locked into the plan:** equivariant GNN trained from scratch (IPPO/MAPPO); de-risk with
(i) degree/local-structure **regularization** (SizeShiftReg), (ii) *optional* cheap-expert BC
warm-start→RL-finetune as insurance, (iii) multi-scale training.

**Two hard caveats — now plan constraints.** (a) **Size-transfer is conditional** — provably defensible only
when comm-graph **local degree stays ~invariant** (graphon / shared-local-structure; Yehudai ICML'21
arXiv:2010.08853, arXiv:2510.03923); else GNNs converge to small-graph "bad minima." **Our ladder drifts ~2×
in density and 10²/2 is a degenerate 2-agent graph** → **Phase 0 now measures per-rung comm degree**, may set
**comm_radius per rung**. (b) **Domain-mismatch** — all positive evidence is continuous coop-nav over modest
(~0.5–1.5×) ranges, *not* grid coverage+connectivity over our ~3.2× area / ~5× agent jump → **Phase 4
validates transfer directly**, BC warm-start kept as insurance. *In-house corroboration more domain-matched
than the lit:* our GCRN belief already transferred 16²/4→32²/10 ([[gcrn-size-invariant-belief]]).

**Refuted (don't do).** E(n)-geometric canonicalization on top of the GNN is **not** mandatory (0-3);
"feed-forward GNN won't size-transfer without recurrence/BPTT" is **not** established → recurrence stays an
*optional* Phase-1a variant.

**Scale recipe (lighter half of #5, medium-confidence).** **Multi-scale JOINT sampling is the spine** +
curriculum finetune at 32²/10; **don't** rely on small-only warm-start (over-anchors the degenerate 2-agent
corner); up-weight both the sparse-small and hard-large ends. → Phase 4 locked to this; the 3-way sweep now
*confirms* rather than *derives*.

**#1/#3/#8/#11 — still OPEN.** No confirmed evidence this round. Triage (low-confidence): keep **#3
frontier-attention** as the primary Phase-5 ablation (λ₂-biased / cut-vertex-aware attention — unifies with
the Fiedler signal we adopted); **#1 energy / #8 toolkits** optional; **#11 diffusion** defer/drop (per-step
budget + non-size-invariance).

**Good / bad.** Good: the highest-leverage uncertainty resolved cleanly and in our favour; **spine Phases
0–4 now final**. Bad: Round-2's verifier spent its whole confirmed-claim budget on #5 → the four ablation
ideas are no more resolved than before; a focused Round-3 would be needed to actually settle them.

**Changes.** `EXPERIMENT_PLAN.md` updated — §1 from-scratch-RL recipe + transfer precondition; Phase 0
degree-measurement + gate; Phase 4 multi-scale-joint locked; §8 → open-idea triage; §9 open items. Memory
updated.

**Next.** (1) Decide #1/#3/#8/#11: **focused Round-3** vs proceed-on-triage. (2) → **writing-plans on the
Phase-0 slice**. (3) Confirm foundation base-task (coverage + light λ₂ vs pure).

---

## 2026-06-26 — Round-3 deep-research (v2): results → Phase-5 matrix FINAL (run `wf_274a9e29-9d4`)

**Effort.** 106 agents · ~3.1M tokens · 24 sources · 118 claims → **25 verified (24 confirmed, 1 killed)** —
the narrowed robotics/MARL scope worked (the v1 relaunch had died on a fetch outage, 0 claims). **The whole
campaign spine is now externally validated and Phases 0–5 are FINAL.**

**Big picture: the literature is independently building SuperBlue's spine.** **Li et al.** (ICRA'22,
arXiv:2109.08536) constrain the **exact decentralized λ₂/Fiedler signal we adopted** as a hard CMDP cost and
hold **71–77 % connectivity** while an unconstrained baseline collapses to **7–30 %** (worse as the team
grows) — proving an explicit *constraint* (not a soft reward) holds connectivity, and documenting the
**in-place-oscillation** failure mode. **MARVEL** (ICRA'25, arXiv:2502.20217) proves a permutation-equivariant
**graph-attention** backbone is scale-invariant @2/4/8 with no retraining. **IR2** (IROS'24, arXiv:2409.04730)
proves the connectivity↔explore trade-off is **learnable**. **HAPPO explorer** (arXiv:2412.20049) reproduces
our **98 %/32 %** gap — the exact thing to fix.

**Phase-5 matrix — RANKED & FINAL.**
- **Slot 1 (keystone) — connectivity-aware explorer tool (idea #3):** IR2-style non-myopic, **λ̂₂-biased
  frontier attention** vs base policy. Borrow IR2's learned trade-off but keep **MAINTAIN ≥90 %**, not
  rendezvous-disconnect.
- **Slot 2 — explorer/relay role-picker (idea #8):** {fixed roles vs homogeneous} × {ROMA-style **embeddings**
  vs RODE-style **action-subsets**}; measure emergence. **Adopt the idea, reject QMIX-mixer / pre-fixed-K**
  (both break scale-invariance; GraphMIX arXiv:2010.04740). Option *discovery* = later, not v1.
- **Deferred — energy (idea #1):** SKIP for 90/90 (no role-emergence evidence; all sources battery/recharge
  or classical Voronoi = excluded framing). Post-90/90 efficiency stress-test only, **per-AREA** normalization.
- **Dropped — diffusion (idea #11):** no slot (per-step cost vs budget; no size-invariant graph-diffusion
  survived). Revisit only for amortized once-per-episode relay trajectory generation.

**Brought FORWARD into the spine.** (1) **PID-Lagrangian** (arXiv:2007.03964) → Phase 2 connectivity sweep:
near-free (KP=KD=0 = plain Lagrangian), pre-empts the oscillation failure mode; start KP=KD=0, sweep KP.
(2) **MARVEL** graph-attention → Phase 1 backbone reference. (3) **Hard-mask-first + (PID-)Lagrangian
backstop**, not soft-reward-alone — corroborated by Li et al.

**Cross-round synthesis.** Li et al. needed **behaviour-cloning** to make the λ₂-constrained two-objective
tractable → reinforces keeping our **optional** cheap-expert BC warm-start (Round-2) as insurance for the
constrained case. Open: can hard-mask + PID-Lagrangian remove that need on a grid (we train from scratch)?

**Refuted (1).** An over-reaching ACORM t-SNE "roles emerge" claim (1-2) — roles *can* emerge (ROMA, 3-0)
but that specific t-SNE evidence didn't hold.

**Good / bad.** Good: external validation of the entire spine + a clean evidence-ranked Phase-5 matrix + two
free spine upgrades (PID-Lagrangian, MARVEL ref). Caveats: Li et al. is multi-robot *navigation*, not grid
coverage — its 71–77 % are mechanism evidence, not coverage benchmarks; energy is "no-evidence-found", not
impossible; the diffusion SKIP is evidence-of-absence.

**Changes.** `EXPERIMENT_PLAN.md` — Phase 2 (+PID-Lagrangian + Li et al.), Phase 1 (+MARVEL ref), Phase 5
(ranked matrix), §8 (resolved), §9. `IDEAS.md` verdicts updated. Memory updated. **Plan FINAL (Phases 0–5).**

**Next.** → **writing-plans on the Phase-0 slice** (substrate + config schema + decentralized Fiedler
estimator + per-rung degree measurement + eval harness).

---

## 2026-06-26 (cont.) — read IR2's code → added the action-head axis to the sweep

Cloned + read `github.com/marmotlab/IR2`. Implementation: discrete **SAC** (γ=0.995, long-horizon),
**graph-attention encoder (6×8-head) + pointer decoder**, action = **pick a frontier/graph-node to head for**
(K=30 neighbors), reward = shaped `explore_util + rendezvous_util − dist` + team done-bonus, where
**`rendezvous_util` is an A\*-painted info-delivery field** toward disconnected teammates (env.py:462).
Connectivity = **connected-components, metric only, *tolerated* not maintained, NO λ₂**; comms = signal-strength
w/ wall attenuation. **Their env ≈ our PersistantNetwork (delivered-coverage), not SharedExploration.**

**Decision: add IR2's goal/frontier-pointer action head as a Phase-1a′ axis** {1-step move head (default) vs
goal-pointer head + A\* controller}. Rationale: it's our own flagged **keystone** — the 1-step move head is the
suspected coverage ceiling ([[marl-action-representation-bottleneck]]) — now externally corroborated by
IR2/MARVEL, and **Phase-5 slot 1 (λ̂₂-biased attention) presupposes a frontier/goal action space**. The
experiment owns the L3-goal→L1-move translation (A\*/greedy controller); the sim stays **movement-only** and the
**100-step budget is unchanged** (controller moves count). The **rendezvous-utility-as-feature** is noted for
the **relay tool / PersistantNetwork** (delivered-coverage), *not* the SharedExploration spine.

**Changes.** `EXPERIMENT_PLAN.md` — Phase 1a′ axis + §3 table row + Phase-5 slot-1 dependency note + config
schema `action_head` + §9 open items. **Open:** A\* vs learned executor (default A\*); goal re-plan cadence;
grid candidate-frontier extraction. Churn: +1 Phase-1 axis (~+2 configs).

**Next.** → **writing-plans on Phase 0** (now also stubs the action-head interface so 1a′ can slot in).

---

## 2026-06-26/27 — Phase-0 Fiedler estimator settled + first CTDE agent (the 48-hour arc)

Two threads, both feeding the campaign: (A) built and **exhausted the decentralized λ₂ estimator** (the
Phase-0 substrate) as a standalone study, and (B) stood up the **first grounded CTDE agent** and found its
failure mode + the design that fixes it.

### A. The Fiedler / λ₂ estimator — `zymera_experiments/FiedlerValueEstimation/`

Standalone supervised study (JAX/Equinox, run on balthar): estimate global λ₂ from local agent views, on
hard-connectivity-guardrail dispersion data (proxy for coverage comm-graphs). Full write-up in that repo's
`RESULTS.md` / `FINDINGS.md`.

- **Aggregator:** `max` & `multihead` co-best (~0.66); single-head attention worst (0.557); gcn/sum/gated/
  laplacian mid. `max` most reliable (cv-std .0055), multihead best extrapolation.
- **Content:** edge-distance content (`margin`/`geom`/`signal`) and `learned` all lift mean .57→.64
  (the prior that `learned` was weak was WRONG).
- **Identity:** `index` helps (+~.05) and was the **single dominant ingredient** in the combination grid;
  `random` is a dead no-op (λ₂ is permutation-invariant → index is a positional fit aid, not identity).
- **Two structural walls:** in-distribution **~0.66 ceiling** (no message-design choice breaks it) and
  **N=30 zero-shot = 0.00** for all 32 configs (the size-transfer wall).
- **Combination grid (20 cfg):** stacking compounds **modestly** — best `max+learned+index` cv20 **.703** (vs
  .66), best extrap →24 **.64** (vs .50); reliability + transfer improve, the ~0.70 ceiling holds. Dynamics
  features (Δdegree / neighbour approach-rate / speed) help a touch (+.01).
- **The real answer (power-iteration verification):** the learned net caps at 0.66 because it is
  **rounds-bound** — 2 message rounds can't compute a global spectral property. Decentralized power-iteration
  hits **0.99**, but rounds-to-precision **grows with N** (N=4 ~8, N=20 ~128). **Mission-budget problem:** 128
  cold-start rounds doesn't fit a 100-step mission; warm-started **tracking** beats cold-start hugely but still
  needs ~K=8 rounds/step (~.55–.83) — neither vanilla option is cheap-and-precise. The gap is
  **rounds/iteration, not message design.**
- **Ensemble / permutation:** measured — do NOT help (correlated bias, not variance; estimator is invariant
  so permutation gives zero diversity). Clean negative.
- **Anticipatory-estimator lit review (robotics/MARL):** every component (NRI neighbour-prediction, GNS
  world-models, decentralized λ₂, prediction-residual Byzantine detection, online spectral tracking) is
  **mature**, and the dual-use conjunction is already 2025 work → demote the anticipatory estimator to
  **borrowed substrate**; the open novelty is the **stealth-adversary vs predictive-detector** game.
- **Infra:** fixed a **GPU XLA vmap miscompile** (single-head attention at N=20 → eval via `lax.map`); made
  all launchers **resumable** + added **step-level checkpoints**; renamed **Fidler → Fiedler** everywhere (36
  files + memory, 219 tests green); **scan-loop trainer refactor** (per-step Python launches → chunked
  `lax.scan` + on-device eval), **bit-exact, ~10× faster CPU** — the cure for the "GPU 99% util / 33% power"
  inefficiency (small host-launched kernels, not capacity).

### B. First grounded CTDE agent — `TeamBlue/SharedExploration/ctde_v0/`

The first agent that matches `agent_architecture.md`: **LPAC backbone (CNN → GNN message-passing KB,
configurable aggregator) → multi-level goal head (L3 goal → fixed L1 controller, NO direct moves) →
decentralized λ̂₂ aux head → centralized MAPPO critic (CTDE)**, full §5 config (every knob logged) + reg.

- **Validated 16×16/4 on balthar GPU:** end-to-end, **controller 100% valid**, conn 99%, aux loss down.
- **2000-iter run:** **aux-λ₂ accuracy 21%→90%** (the head learns λ₂ *better* than the passive estimator) —
  BUT **coverage COLLAPSED 30%→7%** at 100% connectivity. **Diagnosis: the degenerate "huddle" optimum** —
  clumping gives trivial connectivity *and* makes λ₂ trivially easy (hence the inflated 90%). A
  **reward-balance failure**: connectivity dominates coverage; the GPU finally drew real power (197→490 W),
  confirming MARL (vmapped rollouts) uses the card where the tiny estimator kernels didn't.

**Design convergence (with the user) → the full SuperBlue agent as a configurable sweep:**
- **Roles {explorer, relay}** (the labour-division huddle-fix): **explorer** = frontier-attention pointer
  (IR2-style, the Phase-5 keystone) over the KB; **relay** = **λ̂₂-anchor** (holds the bridge by maximizing
  local connectivity) — *the Fiedler estimator becomes the relay's brain.*
- Plus **Compass** (frontier-heading from the KB — the missing exploration drive), **index in messages** (the
  grid's top ingredient), **edge-distance message content** (top estimator finding), **anti-overlap reward**
  (the proven 90%+ lever, [[marl-coverage-clustered-and-push]]), **recurrence/GRU** (the temporal twin of more
  rounds). All **config axes**; permute → gate → mix winners (not a 250k-cfg Cartesian).
- **Already live in the v0:** aggregator, mp_rounds, mechanism, aux-loss, regularization, norm/width/depth,
  the goal head + aux-λ₂ head + central critic, soft-λ₂ target. So the build that remains is the **cognition
  layer** (roles/tools/compass) + a few message/reward knobs + recurrence.

**Staged experiment plan** (`ctde_v0/EXPERIMENTS.md`): **I1** role-picker × mechanism × anti-overlap (8 cfg —
the make-or-break huddle test) → **I2** explorer/relay tools × compass → **F** backbone OFAT (agg, mp_rounds,
recurrence, content — the estimator-derived levers) → **S** scale. ≈33 cfg × 3 seeds ≈ 100 runs, matching the
EXPERIMENT_PLAN envelope; each stage gates the next.

**Changes.** New `FiedlerValueEstimation/` study (RESULTS/FINDINGS). New `ctde_v0/` agent + sweep harness +
`EXPERIMENTS.md`. This JOURNEY entry. **Next.** I1 modules building; preliminary `mechanism × mp_rounds` CTDE
sweep running on balthar; then the I1 roles+anti-overlap sweep (sharded, parallel). Mix winners → I2 → F → S.

---

## 2026-06-27 — I1 roles WIN, but the win does NOT scale; per-step connectivity guardrails are SETTLED (closed) → pivot to L4 phase + barrier + delivered-coverage

The make-or-break arc ran end to end: **roles break the huddle at 16²/4** → **the win collapses with
scale** → **every per-step connectivity guardrail fails to fix it at scale, for a diagnosable structural
reason** → **pivot to a strategy/phase layer (L4) + a connectivity-floor barrier + a delivered-coverage
objective.** This entry is the ledger; the numbers below are the canonical "tried & settled" record.

> ### 🛑 SETTLED — DO NOT RE-RUN: per-step connectivity guardrails do **not** break the scale huddle
> **The whole family is closed:** hard **action-mask**, soft **global-λ₂** penalty, **local degree /
> edge-margin** — under **fixed / Lagrangian / PID** dual weighting. None lift coverage at scale; all
> "succeed" only by **huddling** (dense clump satisfies connectivity for free).
> **KILLER DIAGNOSTIC:** at the huddle each agent's degree ≫ the target, so the connectivity penalty ≈ 0
> and the dual λ never moves — **the huddle SATISFIES the connectivity signal.** A per-step connectivity
> mechanism is therefore **structurally inert**: it reads clumping as a *solution*, not a problem, so it
> cannot push the team apart. ⇒ **Do NOT re-run per-step connectivity-mechanism sweeps at scale hoping
> they fix coverage. This chapter is CLOSED.** The fix has to resolve coverage↔connectivity **in TIME**
> (breathe out / in), not with a stronger per-step penalty.

**The results ledger (exact, reproducible — `ctde_v0/` sweeps on balthar).**

1. **I1 sweep @ 16²/4** — `role_picker{off, expl_relay} × mechanism{action_mask, soft_lambda} ×
   anti_overlap{off, on}` = **8 cfg, 1500 iters.** **ROLES break the huddle.**
   - `role_expl_relay` ≈ **90.9 % coverage / 100 % connectivity** (best cell: `soft_lambda · ao_on`; role
     split ≈ **84 % explorer / 16 % relay**).
   - **Every** `role_off` config **HUDDLES** at **1.6–5.6 % coverage / 100 % conn.**
   - **Mechanism** (action_mask vs soft_lambda) and **anti_overlap** are **2nd-order at this scale** — the
     four role-on cells span just **89.6–90.9 %**. ⇒ **Roles are decisive; KEEP them.** *Don't re-run the
     16²/4 mechanism/anti-overlap sweep expecting differentiation — there is none to find at this scale.*
2. **Scale-transfer of the I1 winner** — `role_expl_relay` + global-λ₂ `soft_lambda`, `comm_r` set per rung
   to hold mean degree ≈ 1.9 (**5 → 6 → 7**). **Coverage COLLAPSES with scale:**

   | rung | coverage | connectivity | how connectivity is held |
   |---|---|---|---|
   | **16²/4** | **90.9 %** | 100 % | (the win) |
   | **24²/6** | **~53 %** | 100 % | huddling |
   | **32²/10** | **~16 %** | 100 % | huddling |

   The win **does NOT transfer.** Connectivity is held by **HUDDLING**, not by a stretched backbone:
   **mean λ₂ stays HIGH (~2.7 @ 32²/10)** = a dense clumped graph; and the role-picker **ABANDONS relays**
   (**explorer-frac → 99.5 % @ 32²**). The machinery is healthy — **λ̂₂ aux ~81 %**, **controller 100 %
   valid** — so this is not an estimator or controller bug; it is the **degenerate huddle optimum
   re-asserting itself at scale**.
3. **Local-edge-margin sweep** — `conn_signal=local_edge_margin`, `degree_target=1.0`, `collision_mask=on`,
   × `mechanism{soft_lambda, lagrangian, pid_lagrangian}` × `{24²/6, 32²/10}`. **NO improvement** — a clean
   confirmation that the *signal source* (local, per-agent, anticipatory) doesn't rescue it either:

   | rung | soft_lambda | lagrangian | pid_lagrangian | global-λ₂ baseline |
   |---|---|---|---|---|
   | **24²/6** | 33 % | 18 % | **55 %** | ~53 % |
   | **32²/10** | 15 % | 13 % | **16 %** | ~16 % |

   All **conn 100 %**, **expl-frac ~99 %**, and **mean λ₂ even HIGHER (~4 @ 32²/10)** — i.e. *more* clumped.
   Ties or loses vs the 53 % / 16 % global baseline. **The Lagrangian dual λ stayed FLAT** (e.g.
   **0.30 → 0.30, violation ≈ 0.001**) — the smoking gun: there is **nothing for the dual to push against**
   because the huddle already satisfies the constraint. This is the diagnostic in the box above, measured.

**Why it's settled, restated.** Across all three sweeps the failure is the *same* and it is **structural,
not a tuning miss**: a per-step connectivity signal (hard mask / soft global λ₂ / local degree-margin, under
fixed / Lagrangian / PID) is **satisfied by clumping**, so at the huddle its gradient/penalty/dual-pressure
is **≈ 0** and it cannot do the one thing we need — *push the team apart*. Stronger weighting, smarter dual
control, and a more local signal were all tried; none change the sign of the problem. **Per-step
connectivity mechanisms treat the huddle as a solution.** No more per-step-guardrail sweeps at scale.

**The pivot — resolve coverage↔connectivity in TIME, not per step.**
- **L4 brain layer (NEW top level) — a strategy / mission-phase layer ABOVE the L3 role-picker.** L4 picks
  the **team PHASE {disperse ↔ gather}** (≡ explore ↔ deliver) as a **temporally-extended option** —
  **commit ~5–10 steps**, *not* per step. L3 picks the **role** within the phase, L2 the **skill**, L1 the
  **move**. The team **fans out to cover, regroups to share, repeats** — connectivity is held *periodically*,
  not every step, so the huddle is no longer the optimum. Decentralized per-agent, **cohering via the shared
  belief** — this makes the **micro→macro bridge an explicit module**. **Build STAGED:** add the L4 phase
  head on top of the existing role-picker, keep gather/disperse **SKILLS scripted first (learn only the
  *timing*)**, grow learned-ness later. **Do NOT train 4 learned levels at once.**
- **Delivered-coverage objective** — coverage counts **only when in contact to share it** (the
  relay / `PersistantNetwork` mission already exists in the codebase). This makes the **disperse→gather
  rhythm EMERGE from the objective itself, with NO connectivity penalty** — the cleanest way to kill the
  huddle, since clumping no longer scores and spreading-without-returning no longer scores.
- **Hyper-Singularity barrier reward** (being built now as a config-knobbed term, `reward.barrier_*`) — a
  per-agent wall on nearest-neighbour distance, `f(x) = k·relu(x−a)² / (M−x)^p` **CAPPED finite (RL-safe)**:
  **0 in the safe zone, an explosive-but-finite wall as a link nears the comm edge `M`.** It is a **SILENT
  connectivity FLOOR** that **composes under the L4 breathing** (a floor the team can ride out to, not a
  per-step pull inward). **It is NOT tested in isolation** — alone it is itself a per-step signal and would
  re-huddle; it only earns its keep *underneath* the L4 phase rhythm / delivered-coverage objective.
- **Mission-safety-as-brain-INPUT gap (open build item).** Today the `MissionSafety` block is an
  **enforcement mechanism** (action-mask / reward-penalty); it is **NOT wired as an explicit INPUT to the
  role/phase brain** — the role head in `nets.py` conditions only on the belief `z`. `agent_architecture.md`
  *intends* mission-safety as an input the role-picker ingests. **The L4/L3 brain should explicitly READ the
  connectivity-danger signal (λ̂₂ / barrier proximity) to decide gather vs disperse.** Logged as open.

**Good / bad.** Good: the I1 huddle test paid off exactly as designed — **roles are the proven huddle-fix at
the base scale**, and we now have a *diagnosed, closed* dead-end (per-step guardrails) rather than an open
question, which saves the whole Phase-2 mechanism×λ budget at scale. The diagnostic (flat dual λ, degree ≫
target, mean λ₂ rising) is mechanistic, not vibes. Bad: the headline 90.9 % is a **16²/4-only** result — the
real target (32²/10) sits at **16 %**; the scale wall is unbroken and the pivot is **unproven** (L4 timing +
delivered-coverage are the new bet, not a settled win).

**Changes.** `EXPERIMENT_PLAN.md` — Phase-2 mechanism×λ work marked **TRIED & SETTLED (fails at scale, do
not re-run)** with the diagnostic + a prominent "don't redo" box; **L4 gather/disperse + delivered-coverage**
inserted as the new phase; results ledger added. `agent_architecture.md` — **L4 strategy/phase layer** added
above L3 (cognition table now L4→L3→L2→L1); mission-safety-as-INPUT gap documented; barrier added as the
connectivity floor. `ctde_v0/EXPERIMENTS.md` + `IDEAS.md` — results recorded in the dials/ledger,
connectivity-mechanism axes marked settled-at-scale, new axes (`barrier_weight`, L4 phase,
delivered-coverage) added.

**Next.**
1. **Build the L4 phase head** on top of the role-picker — categorical `{disperse, gather}` option committed
   for `k≈5–10` steps; **scripted gather/disperse skills first**, learn only the **switch timing**.
2. **Wire the delivered-coverage objective** (reuse `PersistantNetwork`) as the L4 base-task — **no
   connectivity penalty**; check the disperse→gather rhythm **emerges**.
3. **Compose the barrier UNDER L4** (`barrier_weight > 0`) as the silent floor — never alone.
4. **Close the mission-safety-as-INPUT gap:** feed λ̂₂ / barrier-proximity into the L4/L3 head so the brain
   *reads* connectivity danger to time the phase.
5. Only then grow learned-ness down the stack (scripted → learned skills), one level at a time.

---

## 2026-06-27 (cont.) — 7-search literature review → consolidated strategy pivot (`STRATEGY.md`)

A 7-search literature review of every direction we'd been weighing (ES · QD · evolve-then-finetune ·
curriculum/scale · hierarchical RL · role-based MARL · swarm-flat-vs-cognitive) consolidated into a single
verdict — **`STRATEGY.md`** (with full citations) — and it **reverses the in-flight "build the L4 tower next"
plan.** The headline: the elaborate **strategy→role→skill→action brain** is over-reach the evidence undercuts;
hierarchy's measured benefit is **exploration + temporally-extended action, not the structural tower** (Nachum
et al. 2019, arXiv:1909.10618), and the only depth that scales is **2-level manager/worker** (FeUdal 1703.01161
/ HIRO 1805.08296). The huddle is really a **hard-exploration / deceptive-optimum problem in an over-shared
homogeneous policy** — and **ES is folklore as a deception escape** (plain ES collapses to the same degenerate
optima — Salimans et al. 2017, arXiv:1703.03864; the real lever is directed novelty/diversity, which works on
PPO too — Conti et al. 2018, arXiv:1712.06560). **QD is demoted to a frontier-mapping / deception-escape
*diagnostic*** on a compact controller (well-precedented for cov+connectivity descriptors — Engebråten et al.
2020, arXiv:2007.08656), **not** the trainer for the deep GNN. Connectivity stays a **hard constraint, not a
brain level**; the trade-off is **constrained, not scalarized**; the curriculum is **sound but must be fixed**
(connectivity binding at every rung, density pinned, budget binding — LPAC arXiv:2401.04855 / EPC
arXiv:2003.10423). Roles + phases are treated as **emergent measured outcomes** (boids/Couzin show even the
gather/disperse *phase* emerges from flat rules — Reynolds 1987 / Couzin et al. 2002), so the corrected next
move is the **§0′ flat-baseline falsification test** (goal-head + GNN + learned role latent + hard-connectivity
shell + delivered-coverage) — which settles "tower vs. emergence" with data **before** any L4 layer is built.
The L4 phase head and discrete skill library are now **conditional on that test failing**, not the default
build. See `STRATEGY.md` for the apparatus; `EXPERIMENT_PLAN.md` §0′ for the executable run;
`agent_architecture.md` correction box for the design downgrade.

**Changes.** Wrote `STRATEGY.md` (the consolidated verdict + full reference list). `agent_architecture.md` —
dated correction box (cap at 2 learned levels; connectivity = constraint; roles/phases emerge). `EXPERIMENT_PLAN.md`
— §0′ flat-baseline + falsification test inserted as the next step; Phase 2′ marked conditional. `IDEAS.md` —
ES / QD / hierarchy / curriculum verdicts updated with their key citations. **Next.** Run the §0′ falsification
test; let the emergence result decide whether any added structure is earned.

---

## 2026-06-28 — the A/B arm batch + the two-level cognition turn

**Ran (balthar, overnight).** A 4-arm batch attacking the 32²/10 coverage wall on the locked honest spec
(comm_r=5 fixed, collision-mask on, soft connectivity, frontier-attn, 100 steps, 3 seeds, ladder 16→24→32):
**base · armA** (curriculum + σ-noise + DTE tail) **· B-fork** (2 separate-param groups) **· B-dico**
(per-agent identity residual). Launcher `ctde_v0/run_ab_overnight.py`; `XLA_PYTHON_CLIENT_PREALLOCATE=false`,
jobs=3, tmux `ab`.

**Results (partial, 23/39).** Every arm beats base (+6–14 @16²). Diversity arms (B-fork/B-dico) lead, most
at scale — 32²/10: **B-dico 61% (seed1), B-fork 50, armA 47, base 41**. **The DTE tail COLLAPSED at 32²
(8%, 100% conn = the huddle)** → the CTDE central critic is load-bearing at scale. Connectivity 90–100% on
the strict λ₂>0.5 bar everywhere → coverage-at-scale is the wall, not connectivity. **C0 confirmed:
specialization emerges** (B-fork highest role-div). Full writeup: `CAMPAIGN_REVIEW.md` §6. Honest caveat: the
32² rung is single-seed-per-arm so the cross-arm 32² ranking is preliminary; the 16² (3-seed) numbers are solid.

**Designed + converged → `COGNITION_DESIGN.md`.** The **two-level cognition**: a general selector over a small
skill *menu* {disperse, flock, hold} (capabilities like SLAM / λ₂-estimation / A* are always-on *substrate*,
NOT menu items — that's what keeps the menu small); uniqueness lives in the **selector** (individuated,
ES-evolved); the **free-market congestion price** (local same-skill crowding) is the anti-collapse force;
**ES evolves the selector + CTDE-gradient trains the executor** (MERL/feudal — same centralized-training
principle as CTDE, disjoint params, so they compose). Locked setup: weak = can't-solo, vision 3×3, memory = KB,
connectivity success = **giant-component** (team in one piece), numbers test = ladder + **fixed-world N-sweep**.

**Built (gated; scaffolding committed d0cc9ba).** Per-agent coverage metric (redundancy + top-agent share =
the can't-solo number), config/CLI axes (`--selector/--flock/--congestion`), the free-market congestion price,
the flock skill (scripted + learned), the 2×2 launcher. Then — via **parallel subagents** — the **gradient
selector core** (hierarchical skill+offset policy in nets+ppo, **89 tests green**) and the **ES coexistence
trainer** `es.py` (OpenAI-ES/CEM + the MERL interleave skeleton, 6 tests). Selector-off stays byte-unchanged.

**Next.** Integrate selector + ES (wire ES fitness/sync onto `actor.selector_head`), run the 2×2
(flock × congestion) sweep + the N-sweep, and try task-grounded individuation variants on the selector
(graph-position role, diversity-as-a-loss).

---

## 2026-06-28 (cont.) — obstacle verdicts land · connectivity-safe crowded terrains · the curriculum batch · the interactive gallery · deep-research lit review

**Obstacle batch — FINISHED (50/54; 4 r32 OOM-casualties).** The full-factorial ARM×BARRIER×EXPLORE×WORLD
reruns completed o16 (18/18, open) + r24 (18/18, rooms) + r32 (14/18, rooms). **Verdicts, now confirmed
across open AND corridors:**
- **The connectivity barrier is the wrong tool — REFUTED even in corridors.** r24 rooms: barrier-OFF already
  holds 88–98% connectivity; barrier-ON buys the last ~10 pts by sacrificing **half** the coverage (role+bump
  32.7→23.0, base+bump 28.9→6.1). Same as open. (→ the lit review explains *why*: a *fixed* penalty
  coefficient is brittle; the fix is a *learned* Lagrangian.)
- **`einfo` (naive info-gain bonus) reward-hacks everywhere** (~1% — agents hover by uncovered cells without
  covering). **`ebump` (coverage-bump ×3) wins. role ≥ base ≥ sel** persists at 32² rooms.
- Rooms cap coverage low (~33% even trained) — chokepoints are hard in 100 steps.

**Connectivity-safe crowded terrains — BUILT (`ctde_v0/terrains.py`; 10 tests, full suite 105 green).**
`ConnectedClutter` · `Pillars` · `MixedCluttRooms` · `RandomCrowded` (per-reset mixture). Key trick: a
fixed-iteration **BFS flood-fill from a central seed walls off every unreachable cell**, so the free space is
*constructed* to be one connected component — JAX-traceable (runs in the jitted reset) → coverage stays
well-posed (100% reachable), spawn always has free cells. Wired through `config.World` · `env_utils.build_env`
· the `train_ctde` CLI (`--terrain clutter/pillars/mixed/crowded_mix`, `--pillar-*`). Committed locally
**`0a80fed`**; deployed to balthar by **rsync (NOT pushed to main)**.

**Crowded curriculum batch — RAN (balthar, 10/12).** `run_crowded_overnight.py`: 16→24→32 warm-start on
`terrain=crowded_mix`, ARM {role,base} × EXPLORE {eoff,ebump}, density ~15%/rung. Ran *behind* the obstacle
batch (GPU contention; queued via a memory watcher, auto-resumed when obstacle freed). **Findings:**
- **Training-on-crowded HELPS at 24² (+11–13 on hard maps: heavy clutter 17→28, pillars 29→41) but WASHES at
  32² (±3).** The crowded skill doesn't transfer up to 32² → **weak scale-transfer of the harder regime**
  (under-training at the hardest rung and/or the 100-step floor). *Exactly the lit-review open question,
  answered "weakly."*
- **role > base on crowded, most at the hard maps** (32²: pillars +5.7, heavy clutter +3.8; light/mixed ≈tie).
- **Crowded 32²/10 tops out ~12–28% — far from 90/90.** Cluttered 32² in 100 steps is genuinely hard.
- 2 arms (`role_eoff/32`, `base_eoff/32`) flaked on a simultaneous-compile OOM (jobs 2, two 32² at once);
  re-runnable at jobs 1.

**The interactive gallery — BUILT + maintained (`ctde_v0/make_report.py`, `report/index.html`).** Self-contained
canvas viewer (play/scrub · sense region · comm radius · dashed comm links · covered cells · agents colored by
skill/role), now **manifest-driven**: categories + per-run descriptions + final coverage + a **render-time
world-override** (drop any policy into any terrain zero-shot). **41 runs across 7 categories** (A/B · cognition
· crowding-sweep · obstacle reward/barrier · corridors · **crowded zero-shot** · **crowded TRAINED** — the
before/after on identical maps). A/B GIFs re-rendered from a worktree at their training commit (arch drift).

**Deep-research lit review — RAN (107 agents, 2.3M tokens; harness hard-verified 2/11 families, 6 open on a
schema bug).** Gist: **(1) the LPAC weight-shared GNN spine IS the validated size-invariant coverage backbone —
keep it** (enabler = shared filter taps + permutation-equivariance, not depth); but coverage backbones are all
connectivity-blind → our gap. **(2) the connectivity BARRIER is the wrong tool — swap the fixed penalty for a
learned-Lagrangian constraint (RCPO/CPO)**; fixed coefficients are brittle across scale (Tessler ICLR'19),
which *predicts* both our barrier-hurts-coverage AND its failure across the ladder. Li 2022 (CPO on λ₂)
quantifies the trade is real (conn 0.1→0.75 at the cost of task success) → **90/90 simultaneously is the
defensible gap. Novelty = the conjunction** (connectivity-constrained + scale-invariant + hierarchical-role +
adversarial-resilience); closest threat = the LPAC follow-on *"Constrained Learning for Decentralized
Multi-Objective Coverage Control"* (read it). Caveat: verified refs are off-distribution (LPAC = 900–1500-step
imitation from a clairvoyant expert, not our PPO/ES + 100 steps).

**Also.** ES coexistence run finished (round 79, es_best ~227, grad_cov matured 25%→57%) but `run_es.py` saves
only history, **no `model.eqx`** → not gallery-able as-is (add a final model-save next time).

**Next.** (a) **barrier → learned-λ (RCPO) connectivity constraint** — the lit-review's #1 actionable; (b)
test the 32²-under-training hypothesis (more iters on the hardest rung); (c) re-run the 2 OOM'd `eoff/32` arms
at jobs 1; (d) re-run verification on the 6 open lit-review families; (e) multi-map eval to firm up the
24²-helps / 32²-washes pattern. **Push `0a80fed` to main when ready** (balthar's on rsync'd code).

---

## 2026-06-28 (evening) — ES ladder collapses · connectivity shootout decided · the locality redesign

**ES coexistence ladder — RAN, full 16→24→32 (model-save + `--init-from` added to `run_es.py`).** Fixed the
no-model bug (now serialises the `(actor,critic)` snapshot) + wired warm-start so the ES selector carries up
the ladder. Result is **decisive and negative**: the learned mode-selector **collapses with scale** —
coverage **48.8% → 25.3% → 7.8%** at 16/24/32, gap to hardcoded roles widening **−28 → −32 → −50**. The MERL
*mechanism* composes (elite advantage stays +5–6) but the *policy* is the opposite of scale-invariant. **The
learned selector over a global skill menu is dead at scale.** All three rungs in the gallery.

**Connectivity-mechanism shootout — RAN + DECIDED (`run_conn_shootout.py`, 32²/10, role/base × 4 mechanisms).**
Tested the lit-review's #1 (learned-Lagrangian) vs soft penalty vs hard action-mask. Verdict (real conn = λ₂>0.5):
role+soft 44.7/81.2 · **role+lagrangian 42.4/84.7** · role+pidlag 39.8/83.0 · role+maskhard 39.9/63.1. **Decision:
role split + learned-Lagrangian dual** — highest real-connectivity, scale-adaptive, and the dual *needs* the roles
(`base_lag` collapsed to 62%). **The headline: connectivity is ~solved (~85%); coverage (~45%) is the entire
remaining gap to 90/90.** Locked into `LOCALITY_DESIGN.md` §4. (Operational note: `ollama` on balthar repeatedly
grabbed 88 GB and OOM-killed runs; resolved by the user freeing it — added a polite GPU babysitter.)

**Synthesis of all results → `LOCALITY_DESIGN.md`.** Six consistent findings (connectivity free · imposed>learned ·
diversity only if task-grounded · reward-shaping fragile · weights transfer but skill doesn't · adding agents
floods not divides) → one diagnosis: **the coverage wall is a spatial-partitioning / credit-assignment problem,
and anything global-or-learned-to-select fails at scale.** New design principle: **scale-invariance through
locality** (a big world is just tiled small worlds; each agent solves only its local cell). The design: a parallel
*local* toolbox (Voronoi cell + in-cell frontier + compass), **difference rewards** (own your cell → partitioning
emerges), **event-staleness exploration bonus**, **per-agent explore temperature**, **hardcoded-context selection**
(for now), all paired with the learned-Lagrangian. Written with a **falsification test** (redundancy must invert
3.7→1; coverage must hold across scale). Supersedes the learned-selector path of `COGNITION_DESIGN.md`.

**Strategic reframe (the discussion).** The contribution is the **formalisation** — that MARL captures a *team of
autonomous systems* across a spectrum (multi-robot → swarm → LLM-agent teams), where the common object is *a team
choosing when to use which capability*. So the learning's real job is **generalisable orchestration ("when to use
what")**, not the primitives; **resilience (covert adversary) is one stratum** of that claim, not the headline.
Open A/B/C: (A) build the local-skill substrate + param tests; (B) study *learnable* skills vs heuristic; (C) a
versatile blue agent as ground-zero for many missions.

**Gallery — REBUILT.** `make_report.py` viewer redone: **left sidebar, collapsible study-groups** (open-ladder ·
team-sweep · obstacles · crowded · ES · shootout), **search box + scale filters**, light/academic theme,
Okabe-Ito colourblind-safe agent palette, serif captions. **146 tiles**, every distinct seed0 run that loads
(gaps: ~49 seed-1/2 RNG duplicates + 5 arch-drift `bfork`/`wl`). `make_report` now leaves a hand-customised
`index.html` alone on re-render.

**Next.** Housekeeping (this commit). Then the architecture discussion (user has ideas), then scope the locality
experiment with the orchestration fork (B) as the centerpiece.

---

## 2026-06-29 → 06-30 — the occupancy build, the metric reckoning, and the warm-start win

**Overnight pipeline — RAN, 5 experiments / ~104 runs @32²/10** (bench → frontier → occ → warmbig, + r9090 reruns),
chained serially on the one GPU. All rendered into a new *categorized* gallery (by experiment type). Verdicts below.

**Recipe ablation (`run_bench`, 36 runs, 3 seeds) — RAN + CONFIRMED.** role/base × soft/lag × bump/flat ×
open/rooms/crowded. Coverage (connectivity ~98–100% throughout): **role_lag_bump ≈ role_soft_bump** (tied top:
open 44/42, rooms 20/21, crowded 27/25) · base_soft_bump lower (open 33) · **role_soft_flat the floor** (open 29,
rooms 15, crowded 15). **The bump-explore term (`--w-coverage 3`) is the single biggest lever (+12/+6/+10 vs flat);
role-split helps open+crowded; soft = lagrangian (no winner).** The single-seed story, now 3-seed solid.

**Occupancy / boundary belief (#72) — BUILT, then FALSIFIED (clean NEGATIVE).** Built `sense_free` (full
free/occupied/unknown belief) + `occ_frontier` (Yamauchi — the egocentric `local_frontier` collapses under
occupancy) + `boundary` (field-edge channel), opt-in, 235 lab tests green, committed + deployed. A/B vs SLAM-only
(16 runs, 2 seeds): **occupancy is WORSE on all four terrains by ~3–7 pts** (open 42→35, rooms 21→19, mixed 18→15,
crowded 26→22), both seeds agreeing, connectivity unchanged — *and* in the warm-start arm (occ < base ~12 pts on
open). **A clean failed experiment: the extra channels are net noise at 2000 iters; the occupancy/mission-field
idea does not help coverage. Shelved.** (Stronger than the earlier "SLAM-as-coverage = null".)

**MAAC / COMA attention critic (#68) — BUILT, then SHELVED (wrong tool, informative dead-end).** Built the
attention critic + COMA counterfactual + `contribution.py` (per-agent share/gini), 12 tests, composes with the real
actor. Then the user's instinct — *"our setup is too sparse for this"* — plus a skeptical lit review killed it:
COMA's per-agent counterfactual is high-variance and degrades under sparse/team-summed/delayed reward (SQDDPG
measured COMA credit↔truth at **r≈0.13**; Dr.Reinforce exists because the learned Q is the weak link). **Verdict:
for set-cover coverage the contribution is the EXACT difference reward (unique cells, submodular → closed-form, no
critic) — measurement AND difference-reward training unify on it. The attention critic is shelved for a future
dense-per-agent-reward mission, not this one.** Whole transformer/attention axis DROPPED (mission token parked till
mission #2).

**The "% of optimal" reckoning — the metric was nonsense; killed it.** We'd scored `cov ÷ god-view oracle`
(open=72%). The user pushed: a god agent should clear far more. Falsified empirically — the oracle was a weak greedy
Voronoi: it clears 8×8/1-agent optimally, but at 32²/10/100-steps a planned boustrophedon beats it (greedy **72% →
planned 80%** same spawn → **87%** scatter), and the visit-budget bound is **98.6%**. So "59% of optimal" was really
"cov ÷ our mediocre controller" — a denominator that moves when you write a better controller. The real picture is
the ladder: learned ~42 < dec-scripted ~54 < cent-greedy 72 < cent-planned 87 < budget 98.6. **Dropped "% of
optimal" — report raw cov + conn, compare within-map.** New framing the user liked: bracket everything in **[L, U]**
(L = scripted floor, U = budget ∧ reachability), incl. resilience `G(k)` — PARKED for later.

**Cov↔conn frontier (`run_frontier`, 24 runs, 4 seeds) — RAN.** Built the right dial: exposed `--soft-lambda-penalty`
as the cov:conn knob (connectivity from that penalty only), swept [0,0.25,0.5,1,2,4]. **Clean monotone
interpolation: coverage 60→52%, strict connectivity (λ₂>0.5) 7→30%, giant-comp conn 65→79%.** But **connectivity
SATURATES ~30% at 32²/10 no matter how hard you push — the connectivity wall, now quantified as a frontier.**
(First w-coverage dial was wrong — it pins connectivity at every point; caught + fixed before launch.)

**Warm-start into the bigger world (`run_warmbig`, 16 runs) — RAN — THE WIN.** Train 16²/4 → warm-start 32²/10
(`--init-from`), occ vs base. **The warm-started base hits open ~60% coverage AND ~100% connectivity (strict λ₂
~99%)** — vs from-scratch ~44% (+16 pts), reaching the coverage ceiling *while holding strict connectivity*.
**Warm-start BREAKS the coverage↔connectivity trade-off the frontier shows** — the one intervention that moved both
axes at once. **The thing to build on.**

**Operational failures.** 4 of 12 r9090 runs OOM'd at launch (`ollama` + a lingering proc holding the 98 GB next to
the run's 71 GB) → reran after the pipeline (r9090 back to 12/12). Self-pkill + JAX-preallocation footguns navigated.

**Net of the day:** the architecture chase is over and most of it was *negative* (occupancy hurts · MAAC is the
wrong tool · "% of optimal" was meaningless · connectivity saturates) — and that clarity is the value. The two
positives: **role+bump is the recipe**, and **warm-start is the lever that breaks the trade-off**.

**Next.** Categorized gallery (rendering). Then the exact difference-reward contribution + remove-`k` resilience
curve (the #70 bridge), the [L,U] bounds, and the adversarial/red core — building on warm-start, shelving occupancy
+ the attention critic.

---

## 2026-06-30 — June campaign: consolidated summary *(merged from `CAMPAIGN_REVIEW.md`)*

The single-table view of the June SuperBlue coverage campaign, kept as the checkpoint the later work builds on.

**Complete results table** (comm_r noted — the confound that made connectivity free):

| run | rung | comm_r | coverage | connectivity | note |
|---|---|---|---|---|---|
| roles win (I1) | 16²/4 | 5 | **90.9%** | 100% | the small-scale win |
| huddle (role_off) | 16²/4 | 5 | 1.6–5.6% | 100% | |
| scale-transfer | 24²/6 | 6 | 53% | 100% | confounded |
| scale-transfer | 32²/10 | 7 | **16%** | 100% | the collapse (confounded) |
| diagnostics (reward/mech/explore/backbone) | 32²/10 | 7 | 8–20% | 100% | all ❌ |
| **frontier-attn (disperse)** | 32²/10 | 7 | **42%** | 100% | breakthrough (confounded scale) |
| + dials (width/compass/edge/stack) | 32²/10 | 7 | 45–49% | 100% | plateau, seed-noisy |
| recurrence / hold-relay | 32²/10 | 7 | 44% / 31% | 100% | neutral / worse |
| warm-start curriculum | 32²/10 | 5→6→7 | **52%** | 100% | confounded (radio grew) |
| **frontier-attn (HONEST, cr5 fixed)** | 32²/10 | **5** | **31.6%** | **98.7%** | the trustworthy number |
| goal_head (HONEST) | 32²/10 | 5 | 13.5% | 100% | honest baseline |

**Stand behind:** the disperse skill is a real ~2.3× coverage lever at matched comm_r; the diagnosis "can *execute*
a disperse behavior but can't *learn* to disperse" is well-supported; the dead-ends (reward balance, connectivity
mechanism, exploration knobs, backbone-stacking, hold-relay, recurrence) each have multiple runs behind the "no."

**Caveats (what NOT to claim):** absolute coverage above comm_r=5 is confounded — only fixed-spec re-runs are
trustworthy (headline: **disperse ≈ 32% @ 32²/10, honest spec**); the "47% ceiling" is single-seed-fuzzy (±5–10);
the 52% warm-start win was radio-confounded; **connectivity was made easy** (generous comm_r + a trivial λ₂>0.001
bar) so the real coverage↔connectivity trade-off was barely tested — **90/90 was still open.** Everything was on one
architecture (CTDE v0), so "can't learn to disperse" may be architecture-specific.

*This caveat — connectivity made free, one architecture — is exactly what July's fixed-spec T-campaign set out to fix.*

**Settled negatives — do NOT re-run** *(consolidated from `EXPERIMENT_PLAN.md` §0, 2026-06-27)*

The June sweep's most durable output is a **closed chapter**: per-step connectivity guardrails do **not** break
the scale huddle. The *entire family* — hard **action-mask** · soft **global-λ₂** penalty · **local
degree/edge-margin**, under **fixed / Lagrangian / PID** dual weighting — was run, and **all of it is
structurally inert at scale.**

**Killer diagnostic (why it's structural, not a tuning miss):** at the huddle each agent's degree is **≫ the
connectivity target**, so the penalty is **≈ 0** and the dual never moves (measured **λ: 0.30 → 0.30**,
violation ≈ 0.001). **The huddle SATISFIES the connectivity signal** — a per-step mechanism reads clumping as a
*solution*, not a problem, so it structurally **cannot push the team apart.**

Exact ledger (`ctde_v0` sweeps on balthar):

| Sweep | Setup | Result | Verdict |
|---|---|---|---|
| **I1 @ 16²/4** | `role_picker{off,expl_relay}` × `mechanism{mask,soft-λ}` × `anti_overlap{off,on}` (8 cfg) | `role_expl_relay` ≈ **90.9% cov / 100% conn** (split ~84% expl / 16% relay); every `role_off` **HUDDLES @ 1.6–5.6%**; mechanism & anti-overlap **2nd-order** | **Roles decisive → KEEP.** Don't re-run the mechanism sweep expecting differentiation. |
| **Scale-transfer** | I1 winner + soft-λ₂, `comm_r` per rung | cov **90.9% → 53% → 16%** (16²→24²→32²), all conn 100% **by HUDDLING** (mean λ₂ ~2.7 @32² — a clump, not a backbone; roles abandon relays, expl-frac → 99.5%) | **The win does NOT transfer.** |
| **Local-edge-margin** | `conn_signal=local_edge_margin`, `degree_target=1` × {soft-λ, Lagrangian, PID} × {24²,32²} | **NO improvement** (32² = 15/13/16%); mean λ₂ even **higher** (~4); **dual λ FLAT (0.30→0.30)** | **Local signal doesn't rescue it.** |

⇒ **Do not re-run mechanism × λ at scale hoping it fixes coverage.** The fix is not a stronger per-step penalty —
it is to resolve coverage↔connectivity **in time**, or (as July found) to fix the **1-step move head** itself.
The plan's own next step (§0′) already saw this: it called for a **flat baseline with a goal/region action head**
and to *measure emergence* rather than hand-author a phase tower — the exact road to the **T5 planner** and the
**coordination diagnostic** (no emergence) that July then took. *(This is the last durable content from
`EXPERIMENT_PLAN.md`; the rest — the phase ladder, config schema, Round-3 lit verdicts — is superseded by the
T-campaign and lives in the memory corpus.)*

**The architecture decision process** *(consolidated from `agent_architecture.md` + `LOCALITY_DESIGN.md` before
deleting them)* — how the nominal agent's shape was argued out, in four moves:

1. **The multi-level brain (Jun 25–27).** The first design was a modular, brain-like agent: eight modules
   (perception → KB → comms → mission-safety → goal → **role-picker hub** → role tools → operation) arranged as
   a 4-level cognitive stack — **L4** team phase `{disperse↔gather}` · **L3** role-picker · **L2** tools · **L1**
   control, each running at its own timescale, no level micromanaging the one below. The L4 phase layer was added
   Jun 27 as the intended fix for the huddle: resolve coverage↔connectivity *in time* (fan out, regroup, repeat),
   supported by a delivered-coverage objective and a "Hyper-Singularity barrier" connectivity floor.
2. **The self-correction (Jun 27).** A 7-search lit review immediately **retracted the tower** (`STRATEGY.md`,
   now in `literature/`): cap at **2 learned levels** — a goal-selector over a GNN, the one temporal abstraction
   the HRL evidence backs; **connectivity is a hard constraint, not a brain level**; **roles and phases must
   EMERGE**, graded as measured outcomes, never authored. The program-specific reason is the load-bearing one: a
   hand-designed hierarchy *short-circuits the very phenomenon Zymera studies* — flat-policy + emergence keeps the
   macro structure a measured outcome **and gives the red team a real emergent target instead of an installed
   scaffold**. (The rule: impose structure only where it encodes a true world invariant, never a guess about how a
   mind should think.)
3. **The divide-and-conquer branch (Jun 28).** Post-ES-collapse, a competing design (`LOCALITY_DESIGN.md`):
   *"a big world is many small worlds."* A **parallel local toolbox** — Voronoi partition + in-cell frontier +
   difference rewards + a **hardcoded context rule (no learned selector)** — with the sharp diagnosis that **the
   coverage wall is a spatial-partitioning / credit-assignment problem, not a connectivity one, and adding agents
   FLOODS rather than divides** (redundancy 3.7 → 7.9). Its connectivity shootout (32²/10, 8/8) picked
   **learned-Lagrangian RCPO** (~85% real-conn) over hard-mask / soft-λ / PID.
4. **What survived vs. died.** The **D&C toolbox** (Voronoi partition + hardcoded selector) was **rejected**; the
   **L4 phase tower, the role-picker hub, the discrete skill library, and the barrier floor were never built.**
   Three things survived: **difference rewards** (adopted — the exact submodular contribution), **the
   flooding-not-dividing diagnosis** (→ July's coordination diagnostic + the obligate-collaboration reframe), and
   **the cap-learned-levels + emergence-as-target principle** — which is exactly what **T5** became: a 3-clock
   **L3-goal / L2-distilled-planner / L1-tracker** with a single learned level over a mechanical planner, no L4,
   no authored roles. (The shootout's "connectivity solved ~85%" was later overturned by **T3b** — the dual sits
   non-binding at the honest spec.)

---

## 2026-07 — the honest re-run: the near-term **T1–T5** campaign (fixed comm_r=5, real λ₂ bar)

Re-ran the June idea-space as a numbered ladder at the trustworthy spec. The verdicts were mostly **negative**, and
the negatives are the value — they say where the ceiling actually is.

- **T1 · frontier-attention** — the June "disperse lever" — **collapses by 24²**. The failure is **representational
  (the 1-step reactive move head), not comms.** The attention head reaches ~98% coverage but disperses the swarm
  (connectivity ~32%); at the honest spec and larger scale it doesn't hold.
- **T2 · roles vs flat** — **flat ≥ role.** The learned explorer/relay role-split does not beat a flat shared policy
  at the honest spec.
- **T3 / T3b · connectivity control (Lagrangian / PID)** — **NULL, both scales.** The dual variable sits at ≡0; the
  target τ is below the natural λ₂, so the constraint is **non-binding** — not a bug, the connectivity bar was slack.
- **T4 · count-invariant critic** — a **set-based critic (setpool/setattn) beats the conv critic**, but
  **setattn ≈ setpool** — attention buys nothing over mean/sum pooling here.

**Architecture audit + coordination diagnostic (July).** Audited the two stacks — Stack A (ES, ~95% heuristic, a
145-param gate) vs Stack B (MARL/PPO); found the GCRN graph-belief **unplugged** and the so-called "Fiedler
estimator" was an **eigendecomp oracle**, not learned. The coordination diagnostic was the sharpest finding:
**no controller divides labor** — the per-cell redundancy floor stays ≥ 2.2 (up to 6), and **adding agents floods
the space rather than dividing it.** The emergence we were chasing isn't there under this setup.

**The reframe — obligate collaboration.** Why doesn't labor divide? Because the mission doesn't *force* it:
**obligation = reward-shape × partial-observability**, and the difference-reward signal collapses when every agent
is pivotal. This became the flagship framing (and the Byzantine anchor for the adversarial program).

---

## 2026-07 → 08 — the **T5** pivot: a fully-learned 3-clock, and solving the planner

The T-campaign localized the ceiling at the **1-step reactive move head**. T5 answers it with a fully-learned
**3-clock** agent: **L3** goal selector · **L2** value-iteration planner · **L1** one-step tracker.

- **The planner (L2) · RL-only fails** — the reward field is flat, no gradient to climb — **so distill against a
  4-connected BFS wavefront.** Wired into `ctde_v0` (`controller='mvprop'`, soft connectivity, no hard mask).
- **Planner-arm matrix** — **mvprop ≈ gppn ≈ highway** (all the same value-iteration wavefront, size-invariant) **>>
  MSP** (the spatial-planning transformer — worst, at/below the greedy floor). **The planner is solved and is NOT
  the mission bottleneck.**
- **Stage-0 verdict** — a classical planner hits **78.9%** vs the learned 1-step head's **25%** zero-shot on unseen
  maps (**3.1×**). The gap was **planning, not learning** — which is exactly why L2 is a distilled planner.

---

## 2026-08 — planner validation, the KB audit, and the connectivity tax

- **Planner validation trilogy.** (1) A single-agent **50-seed** benchmark on *external* maps (maze/cave/rooms +
  MovingAI): the value-iteration arms beat Manhattan everywhere; MSP sits at/below the floor. (2) A multi-agent
  **bracket** — greedy (lower) / god-view-BFS (upper) — with the planner swapped as the middle. (3) A **mechanistic
  w-recovery proof**: MVProp's forward pass *is* value iteration over one learned passability field
  `w = σ(gain_conv[blocked, goal])`; measured **w-recovery 97.7–99.6%** on every map type (incl. OOD),
  **goal-invariant** (|Δw|≈0.001), field↔oracle correlation **0.96–0.999** (vs MSP 0.79–0.93). Optimal
  **by construction**; it generalizes because "passable iff not wall" is map-independent local physics.
- **Distillation ablation.** Deployed K32/γ0.9/rects is **suboptimal**; best = **mixed-maps + K128 + γ0.99**
  (maze 10→29, rooms 70→100). The knobs (depth · training diversity · discount) **compound** — γ0.99 backfires
  alone but wins with K + diverse training behind it. Recursive-backtracker mazes stay hardest (diameter > K).
- **The connectivity tax + corrected ceiling.** The old greedy god-view "ceiling" was **broken on walls** (~2×
  understated — greedy freezes at walls); the BFS-correct ceiling is ~0.60 on rooms/mixed. So **low walled-map
  coverage is the cost of staying connected — the connectivity tax — not weak planning.** The headroom is **L3
  relay coordination, not the L2 planner** (which is solved).
- **Comms subsystem, verified from source.** Chebyshev-disk graph (radius comm_r=5), a **lossless** gossip channel
  (`config.dropout` is NN-dropout, not the channel), a 1-bool SLAM-belief grid, an LPAC-KB 2-round size-invariant
  GNN, trust-by-default — and **occlusion OFF** in the runs (through-wall comms). The wall-knowledge an agent has is
  almost entirely gossiped, not sensed (sense_r=1).
- **The KB audit.** A linear-probe of the shared-belief embedding: roll out a frozen checkpoint, record `z` with
  full comms **vs** with the comm-graph emptied, fit probes to god-view ground truth held out **by map**. Asks:
  does the belief actually encode a property, and is it **comms** that puts it there? Collected over 6 configs
  (open/rooms × 16/24/32).

**Missions #2/#3 (Organize family, adversarial).** **Tether-relay** — a relay chain keeps a moving lead
base-connected through a comms-denied maze; k-redundant connectivity (Menger) under a covert cut-vertex red.
**Intermittent-delivery** — relays carry payloads between moving endpoints over an *un-holdable* graph
(store-carry-forward / time-varying-graph journeys); redundancy = out-disjoint journeys.

**Lit positioning (the honest verdict).** The nominal architecture is **not** a standalone contribution
(LPAC / RODE / G²N / CDS all exist). The surviving novelty is the **covert-internal-adversary substrate** on a
connectivity-constrained coverage mission, plus the Pareto/ceiling framing. Mission = realistic, not novel;
the contribution is resilience + the reproducibility/transfer story.

---

## 2026-08-08 — consolidation, the GPU wipe, and the code/results split

A housekeeping day that reshaped the workspace.

- **Two code-documentation sites.** Auto-generated API docs (mkdocs-material + mkdocstrings, **static** griffe so
  nothing imports/executes): the reusable **env** (`zymera`) → **ZymeraLab** Pages; the **pipeline**
  (`ctde_v0`/`t5lab`/`planner_study`/`audits`) → **ZymeraExp** Pages. The env is documented separately so it stays
  stable while the learning pipeline is rewritten.
- **The GPU wipe + full backup.** `balthar` (heavy 32²) and `gaius` (light 16²/24²) were reset. Before that, **all
  training results (~3.6 GB, 786 histories + 790 checkpoints)** were pulled to `results_archive/`, classified by
  T-stage, and reconciled file-by-file (the BlueAgentV3 credit runs, outside `SharedExploration`, were caught in the
  reconciliation). The **KB-audit** code + results were rescued; the superseded **Fiedler** estimator was let die.
- **The code/results split.** All code moved into **`zymera_lab/experiments/`** (`ctde_v0`, `t5lab`,
  `planner_study`, `audits`); **`zymera_experiments/` is now results-only, zero `.py`.** Pipelines stay importable
  via a venv `.pth` (top-level `import ctde_v0` unchanged). Dead classical `t5/` code deleted, its design `.md` kept.
- **Doc cleanup.** Superseded design docs removed (BLUEAGENT_V3, TRANSFORMER_DESIGN, COGNITION_DESIGN, the stale
  agendas); this journal merged (`CAMPAIGN_REVIEW`) and brought current.

**Where it stands.** The nominal team is a solid substrate: coverage under a soft connectivity constraint, a
**solved** distilled planner, a size-invariant belief, and an honest read that the remaining coverage headroom is
**L3 relay coordination**, not planning. The thesis proper — **covert internal misbehavior → mission failure**, the
minimum m-of-n, and the resilience curve — is the next build, now on the tether-relay / delivery missions where
connectivity is an *obligate* deliverable rather than a slack constraint.
