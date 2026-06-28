# SuperBlue · Shared-Exploration — Research Journal

A dated, detailed log of the campaign: **decisions, experiments, results, what went well/badly, changes,
and next steps** — one entry per working day. Newest at the bottom. Pairs with `agent_architecture.md`
(the design) and the per-experiment configs (below).

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
