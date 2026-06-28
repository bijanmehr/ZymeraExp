# SuperBlue · Shared-Exploration — Ideas to Verify (research backlog)

Ideas pitched for the superblue agent, each to be **verified by literature + our own experiments** before
committing. Per idea: **hypothesis** (why it might help), **initial read** (what I think the lit says +
my assessment — *confidence noted*), **verify** (lit + experiment), **hooks** (where it plugs into
`agent_architecture.md`). Status tags: 🟢 well-grounded · 🟡 plausible, needs grounding · 🔴 uncertain /
define-first. Progress logged in `JOURNEY.md`.

---

### 1. Energy token / battery (agent-level **and** team-level) 🟡
- **Hypothesis.** Each action costs energy; agents collectively optimize the spend → induces efficient,
  coordinated behavior and a realistic budget. Team-level shared budget creates a commons/coordination
  pressure.
- **Initial read.** Grounded in energy-aware multi-robot (persistent monitoring, energy-constrained
  coverage, recharging) and **budget-/cost-constrained (C)MDP** RL. Team-level = shared-resource /
  common-pool → can sharpen emergent coordination but adds credit-assignment difficulty. *Note: the sim's
  `Body.energy (N,) f32` field is already reserved for exactly this.* Combines naturally with the
  connectivity trade-off as another **constraint/objective axis**.
- **Verify.** Lit: energy-aware coverage / persistent monitoring; budget-constrained MARL. Exp: does an
  energy budget change coverage/connectivity behavior or improve coordination? agent-level vs team-level.
- **Hooks.** A cost term (reward-eng) + a state field (Body.energy) + possibly a mission-safety input.

### 2. Learnable comms (GNN); message content & type 🟢(lit) 🟡(fit)
- **Hypothesis.** Letting agents learn *what* to send beats fixed gossip for coverage+connectivity.
- **Initial read.** Heavily studied: **CommNet, DIAL/RIAL, TarMAC** (attention comms), **IC3Net** (gating
  *when* to talk), **DGN / MAGIC** (graph-conv comms), NDQ. Message types: continuous vectors (default),
  discrete/symbolic (emergent language — harder, more interpretable), gated/sparse (bandwidth). Trade-offs:
  expressiveness vs bandwidth, non-stationarity, and — for us — it **breaks the current env-mediated-comms
  contract** (a P3 *comms seam* + a richer adversarial attack surface). GNN message-passing is the natural
  fit for a graph topology.
- **Verify.** Lit: the comms-learning canon (above). Exp: learnable comms vs gossip on cov/conn; bandwidth
  & robustness ablation; discrete vs continuous messages.
- **Hooks.** Comm stack (module 3); ties to graph-embedding KB (#10).

### 3. Frontier attention — how good *really*? 🟢 (we have prior data)
- **Hypothesis.** Attention over frontier cells is a strong explorer mechanism.
- **Initial read.** Classic frontier exploration (Yamauchi) + learned attention. **We already have a
  direct result: FrontierAttnAC hit ~98% coverage but dispersed the swarm (connectivity ~32%).** So it is
  *excellent for coverage, poor for connectivity* — the exact trade-off tension. Useful as the **explorer
  tool**, but must be **connectivity-gated** (mission-safety / λ₂).
- **Verify.** Re-validate on the new sim; key question = can it be made connectivity-respecting and still
  near-98% coverage? (this is half the 90/90 story).
- **Hooks.** Explorer role tool (module 7), learned variant.

### 4. CBF / KKT / Lagrangian for MARL 🟢(lit) ⛔ **SETTLED-CLOSED as a per-step coverage fix (2026-06-27)**
> **Tried & settled (`EXPERIMENT_PLAN.md` §0):** as a **per-step** connectivity constraint this **does not
> fix coverage at scale** — action-mask / soft-λ₂ / degree-floor under fixed/Lagrangian/PID **all huddle**
> (the huddle satisfies the constraint, dual λ stays flat, mechanism is structurally inert). **Do NOT re-run
> per-step connectivity-mechanism sweeps at scale.** The principled-constraint *idea* survives only **in
> time** (the L4 phase floor / the barrier), not as a per-step pull. Connectivity is now handled by Phase 2′.
- **Hypothesis.** Treat connectivity (and energy/safety) as a **constraint**, solved with
  CBF/Lagrangian — principled vs hand-tuned penalties.
- **Initial read.** Strong lineage: **Control Barrier Functions** for connectivity/collision in MRS
  (Zavlanos connectivity control; CBF-QP); **constrained RL** (CPO, **Lagrangian-PPO**, RCPO, IPO); KKT =
  the constrained-optimization formalism. *The lab already ships `cbf_conn` / `cbf_coll` (soft-CBF) reward
  terms.* This is the **principled form of the parked trade-off** (max coverage s.t. connectivity ≥ c) and
  of mission-safety. Hard-mask = discrete CBF; soft = Lagrangian.
- **Verify.** Lit: CBF connectivity control; constrained MARL. Exp: CBF/Lagrangian vs hard-mask vs
  soft-penalty — which reaches 90/90 and how gracefully.
- **Hooks.** Mission-safety (module 5) + trade-off scheme + reward-eng.

### 5. Physics-informed / -enabled / -inspired networks 🔴 (define first)
- **Hypothesis.** Physical priors make exploration/connectivity more sample-efficient or generalizable.
- **Initial read.** Two senses, very different value: (a) **physics-INFORMED** (PINN, PDE-residual loss) —
  unclear fit to a discrete grid task. (b) **physics-INSPIRED** (the strong one) — **potential fields,
  diffusion/heat-equation frontier spreading, gradient-flow coverage (Lloyd/Voronoi = coverage control),
  Laplacian/λ₂ dynamics for connectivity**. The SOTA size-invariant coverage method **LPAC (Learnable
  Perception–Action–Communication)** is a graph/physics-flavored architecture and is already on our
  borrow-list. *Define which sense we mean.*
- **Verify.** Lit: **LPAC**, potential-field/Voronoi coverage, diffusion exploration, Laplacian dynamics;
  PINNs only if a continuous formulation appears. Exp: physics-inspired priors vs plain nets.
- **Hooks.** Perception/KB representations; control tools; the whole architecture (LPAC is a reference).

### 6. Network fundamentals — type / size / hyperparameters 🟢
- **Hypothesis.** These "boring" choices dominate results; we must study them systematically.
- **Initial read.** True and well-documented: in on-policy MARL (**MAPPO benchmark**, "what matters in
  on-policy MARL") hyperparameters/implementation details swing results hugely. For **scale-invariance /
  warm-start**, architecture type matters most: **GNN / attention / recurrent** transfer across N and grid
  size; MLPs don't. Size/depth/normalization = ablation axes.
- **Verify.** Lit: MAPPO/IPPO benchmarks + GNN-MARL. Exp: a deliberate **network ablation axis** in the run
  plan (type × size × key hyperparams), tied to warm-start transfer.
- **Hooks.** Every learned module; a first-class run-plan axis.

### 7. Multi-phase mission cycle: **dispersion → stabilization → contraction**, repeat 🟢 **PROMOTED TO SPINE (2026-06-27)**
> **Now the primary lever (Phase 2′).** After per-step connectivity guardrails settled-closed at scale
> (`EXPERIMENT_PLAN.md` §0), this idea is the chosen fix: an **L4 `{disperse↔gather}` phase as a
> temporally-extended option** (commit ~5–10 steps) **above** the role-picker — resolve cov↔conn **in TIME**.
> Paired with the **delivered-coverage** objective (rhythm emerges, no conn penalty) + the **barrier** floor.
> See `agent_architecture.md` L4 section.
- **Hypothesis.** Structure shared exploration as a repeating macro-cycle — *disperse* to find frontiers,
  *stabilize* (regroup), *contract* to maximally share info — directly trading coverage for connectivity
  in time rather than fighting them simultaneously.
- **Initial read.** Intuitive and well-motivated; resembles **intermittent / periodic-connectivity
  exploration** and **rendezvous-based info sharing** in MRS (agents spread, then reconnect to dump
  maps — the "delivered coverage / relay" family). A learned or scripted **phase controller above the role
  picker** (or a temporal prior on roles). Could be a major lever for 90/90 (don't hold connectivity every
  step — hold it periodically). Somewhat novel as an explicit learned cycle.
- **Verify.** Lit: intermittent/periodic-connectivity exploration; rendezvous; event-triggered comms. Exp:
  phased cycle vs continuous; learned vs scripted phase timing.
- **Hooks.** A new top layer (L3+) above the role picker, or a temporal structure on roles/goals.

### 8. Tool design per role (Claude-Code-inspired toolkits) 🟡 (design)
- **Hypothesis.** Treat each role as an agent with a **toolkit** it can invoke (like Claude Code's tools);
  the right per-role tools make behavior strong + interpretable.
- **Initial read.** Maps directly onto our "role calls a tool" structure (HRL / options / skills). Mostly a
  **design** exercise, not lit: enumerate tools — *explorer:* frontier-finder, A*/BFS path, coverage
  gradient, frontier-attention; *relay:* λ₂/Fiedler estimator, bridge-positioner, hold-position,
  spanning-tree backbone. The Claude-Code framing reinforces *typed, composable tools per role*.
- **Verify.** Lit: HRL/options/skills + tool-use/skill-libraries. Exp: which per-role toolset reaches 90/90;
  heuristic vs learned tools (#2,#3).
- **Hooks.** Role tools (module 7); role set (module 4).

### 9. Reward engineering: size-invariant, max-generalizable, **reward-agnostic?** 🟢
- **Hypothesis.** Rewards that are scale-invariant and general (ideally task-agnostic) transfer and
  generalize best.
- **Initial read.** Size-invariance = **fractional/normalized** signals (coverage *fraction*, connectivity
  *fraction*) — the lab's terms are already fractional. Generalizability = **potential-based reward shaping
  (Ng et al.) — provably policy-invariant.** **"Reward-agnostic"** → **intrinsic motivation / info-theoretic
  exploration** (RND, count-based, empowerment, **maximize info-gain / uncertainty reduction**) — and
  notably *shared exploration's objective IS uncertainty reduction*, so an info-gain intrinsic reward is
  almost **task-free** for us. This is a genuinely promising unifier.
- **Verify.** Lit: potential-based shaping; intrinsic motivation (RND/empowerment/info-gain); unsupervised
  RL. Exp: fractional + PBRS + intrinsic info-gain vs hand-shaped; does intrinsic-only explore well?
- **Hooks.** Reward engineering (separate, experiment-side); ties to KB/belief (uncertainty = belief).

### 10. Graph-embedding ideas — helpful? 🟢
- **Hypothesis.** Embedding the comm/team graph gives the agent structure-awareness (who's a relay /
  cut-vertex) and a size-invariant representation.
- **Initial read.** Highly relevant. The **role-picker deciding relay-vs-explorer by graph position is a
  graph-embedding / centrality problem**; the **Fiedler vector is literally a spectral graph embedding**
  (ties #4). **GNN node embeddings** of the comm graph are a natural **size-invariant KB/belief**
  representation (supports warm-start). So graph embeddings unify KB (#10), role-picker, connectivity
  (#4), and learnable comms (#2).
- **Verify.** Lit: spectral/Laplacian embeddings, node2vec, GNN node embeddings for MARL; graph
  centrality for role assignment. Exp: graph-embedding KB/picker vs flat features.
- **Hooks.** KB/belief (module 2), role picker (module 6/4), comms (#2).

### 11. Diffusion models — can they help? 🟡 (promising in specific roles)
- **Hypothesis.** Generative diffusion helps via expressive multimodal planning/policies, or a learned
  prior over the *unknown* map.
- **Initial read.** Real but **role-specific**. Strongest fits: **(a) generative belief / map-completion** —
  image-style diffusion **inpaints the partial occupancy map** to predict unexplored structure & frontiers
  → a learned KB **prior over the unknown world** (novel-ish, strong fit, ties #9 info-gain); **(b) planner
  tool** — **Diffuser / Decision-Diffuser**-style goal-conditioned trajectory generation for a role
  (multimodal coverage paths) as an L2 tool; **(c) trade-off conditioning** — classifier-free **guidance**
  on the coverage/connectivity λ to generate frontier-respecting plans (ties #4); **(d) MADiff** for
  coordinated joint team trajectories. **Concerns:** diffusion is **iterative → slow** (bad for per-step
  online control at 32²/10 × big sweeps — mitigate with few-step/consistency/DDIM, or use it *offline as a
  planner*); online-MARL diffusion-policy training is **less mature** (DIPO / QSM / diffusion-QL emerging);
  image-diffusion is **not size-invariant** (conflicts with warm-start unless **graph diffusion** / local
  patches). Net: likely **yes as a belief-prior or planner; risky as a per-step online policy.**
- **Verify.** Lit: Diffusion Policy (Chi) · Diffuser / Decision-Diffuser (Janner, Ajay) · **MADiff** ·
  diffusion map/scene completion (inpainting) · graph diffusion · few-step/consistency models · online
  diffusion-RL. Exp: diffusion map-completion prior in the KB vs none; diffusion planner tool vs A*/attention;
  an honest **inference-cost** check at 32²/10.
- **Hooks.** KB/belief **prior** (module 2) · explorer/relay **planner** tool (module 7) · trade-off
  conditioning (#4) · graph-embedding overlap (#10).

---

## Lit-review plan (proposed)

The ten cluster into four review themes; suggest deep-researching them in this priority order
(highest leverage on 90/90 + warm-start first):

1. **Constraints & objectives** — CBF/Lagrangian/constrained-MARL (#4) · reward eng + intrinsic/info-gain +
   PBRS (#9) · energy budgets (#1). *(Directly drives the trade-off + mission-safety + reward.)*
2. **Representation & comms** — graph embeddings (#10) · learnable GNN comms + message types (#2) ·
   frontier attention (#3, mostly re-validate ours) · **diffusion belief-prior / planner (#11)**.
   *(Drives KB, comms, scale-invariance.)*
3. **Structure & control** — multi-phase cycle / intermittent connectivity (#7) · per-role tool design (#8,
   mostly design) · physics-inspired / **LPAC** (#5). *(Drives the macro-structure + control tools.)*
4. **Network fundamentals** — type/size/hyperparams for size-invariant MARL (#6). *(Cross-cutting engineering.)*

Several already have partial grounding in prior reviews (hierarchical role/skill; connectivity role
allocation; MRS-borrow incl. LPAC, λ₂, coverage control) — those get cited, not re-derived.

---

## Round-1 deep-research verdicts (2026-06-25 · run `wf_2b07f581-6cc`) — see `JOURNEY.md` for full writeup + citations

| # | Idea | Verdict | One-line |
|---|---|---|---|
| 5/6/2 | **LPAC graph-net backbone** | **ADOPT (mechanism)** | size-invariant perception+comms+coverage, zero-shot scale transfer; but NO connectivity + trained by **imitation** (open: RL-from-scratch?) |
| 10 + 4 | **Decentralized Fiedler / λ₂** | **ADOPT** | decentralized power-iteration estimator + λ₂-gradient; **replaces the exact-eigendecomp oracle**; scale-invariant; *soft not hard* |
| 4 | CBF / Lagrangian / constrained | **TEST (soft) · SKIP (hard-CBF as-is)** | on-grid → action-mask or Lagrangian-PPO; hard-CBF guarantee is continuous/centralized |
| 9 | Reward-agnostic / intrinsic | **TEST (scale-inv) · SKIP (tabular counts)** | max-state-entropy ≡ our objective; coordinate novelty across teammates; non-tabular only |
| 7 | Multi-phase cycle | **TEST (scaffold)** | recurrent-connectivity (IR2) informs structure; optimizes time-to-coverage not 90/90; assumes known positions |
| 2 | Learnable comms (gating) | **TEST** | when-to-talk gates cut ~80% bandwidth; folds into the GNN comm layer; weak scale-transfer evidence |
| 3 | Frontier-attention (conn-respecting) | **OPEN (round 2)** | no verified claims beyond our prior 98%/32% |
| 1 | Energy tokens | **OPEN (round 2)** | no verified claims |
| 8 | Per-role toolkits / options | **OPEN (round 2)** | no verified claims; learned modality-selector-vs-oracle was **REFUTED** |
| 11 | Diffusion | **OPEN (round 2)** | no verified claims survived; inference-cost vs 100-step budget is the crux |

---

## Round-2 + Round-3 deep-research — FINAL verdicts (2026-06-25/26 · runs `wf_45751865-0aa`, `wf_274a9e29-9d4`)

**Round-2** resolved the make-or-break methodological gate: **from-scratch RL on the equivariant GNN is
viable — imitation NOT required** (SS-MARL / LEGO / SHPPO / EPC); caveats = size-transfer needs
local-degree-invariance + multi-scale-joint training. **Round-3** (narrowed to robotics/MARL) resolved the
four open ideas AND **externally validated the spine** — Li et al. (arXiv:2109.08536) constrain the *same*
λ₂; MARVEL (arXiv:2502.20217) = scale-invariant graph-attention; IR2 (arXiv:2409.04730) = trade-off
learnable; HAPPO (arXiv:2412.20049) reproduces our 98/32 gap.

| # | Idea | FINAL verdict | One-line + Phase |
|---|---|---|---|
| 3 | Connectivity-aware frontier-attention | **ADOPT — Phase-5 slot 1 (keystone)** | IR2-style non-myopic λ̂₂-biased frontier attention; MAINTAIN ≥90 % (not rendezvous); closes our 98/32 gap |
| 8 | Explorer/relay role-picker | **ADOPT/TEST — Phase-5 slot 2** | roles ↑perf+convergence + can emerge (RODE/ROMA/ACORM); use embeddings / action-subsets, **reject QMIX-mixer / pre-fixed-K** (break scale-inv); discovery later |
| 1 | Energy / effort cost | **SKIP for 90/90 — deferred** | no evidence it induces roles; post-90/90 efficiency stress-test only, **per-AREA** norm |
| 11 | Diffusion | **DROP — no slot** | per-step cost vs 100-step budget; no size-inv graph-diffusion survived; revisit only as amortized once-per-episode relay tool |

**Brought forward into the spine (not Phase 5):** **PID-Lagrangian** (arXiv:2007.03964) → Phase 2 connectivity
sweep (near-free, kills the oscillation failure mode); **MARVEL** graph-attention → Phase 1 backbone ref;
**hard-mask-first + Lagrangian backstop** (Li et al.: hard constraint > soft reward). **Plan FINAL (Phases
0–5).** Full writeup + citations in `JOURNEY.md`.

---

## ⛔ EMPIRICAL UPDATE — verdicts REVISED by our own runs (2026-06-27)

The lit-derived spine met the experiment. **What the literature got right:** roles help (idea #8), and a
connectivity *constraint* holds connectivity (idea #4). **What our runs overturned for THIS task:** the
connectivity-constraint family — idea #4's whole apparatus — **does not fix coverage at scale on dense-grid
coverage** (Li et al.'s setting is navigation, where clumping is *not* a free connectivity solution; on our
grid it is). This is the **`EXPERIMENT_PLAN.md` §0 settled ledger**.

> **⛔ SETTLED — DO NOT RE-RUN: per-step connectivity guardrails (idea #4 family) fail at scale.**
> hard action-mask · soft global-λ₂ · local degree/edge-margin, under fixed/Lagrangian/PID — **all huddle.**
> **KILLER DIAGNOSTIC:** at the huddle degree ≫ target ⇒ penalty ≈ 0 ⇒ dual λ flat (`0.30→0.30`) ⇒ **the
> huddle SATISFIES the connectivity signal**, so the mechanism is **structurally inert** and cannot push the
> team apart. **Numbers:** I1 @16²/4 roles → **90.9 % cov** (role_off huddles **1.6–5.6 %**); scale-transfer
> **90.9 → 53 → 16 %** (all conn 100 %, held by huddling, mean λ₂ ↑, relays abandoned); `local_edge_margin` ×
> {soft/lagrangian/pid} → **24² 33/18/55 %, 32² 15/13/16 %**, no gain. ⇒ **Resolve cov↔conn in TIME, not per
> step.**

**Verdict revisions.**

| # | Idea | WAS | NOW (2026-06-27) | One-line |
|---|---|---|---|---|
| 4 | CBF / Lagrangian / constrained (per-step conn) | TEST → spine (Phase 2) | **⛔ SETTLED — does NOT fix coverage at scale; CLOSED as a coverage fix** | huddle satisfies the constraint; dual λ never moves; structurally inert |
| 8 | Explorer/relay role-picker | ADOPT (Phase-5 slot 2) | **✅ CONFIRMED @16²/4 (decisive: 90.9 % vs ≤5.6 %), but win does NOT scale → promote, then top with L4** | roles break the *base-scale* huddle; not the scale huddle |
| 7 | **Multi-phase cycle (disperse→gather)** | TEST (scaffold, Phase-5 carry-over) | **🔜 PROMOTED TO THE SPINE — Phase 2′ (the new primary lever)** | resolve cov↔conn in TIME; L4 temporally-extended `{disperse,gather}` option (commit 5–10 steps) |
| — | **Delivered-coverage objective** (`PersistantNetwork`) | (relay mission, separate) | **🔜 ADOPT — Phase 2′ base-task** | coverage counts only in-contact ⇒ disperse→gather **emerges**, **no conn penalty** |
| — | **Hyper-Singularity barrier** (conn FLOOR) | (new) | **🔜 ADOPT — composes UNDER L4, never alone** | `f(x)=k·relu(x−a)²/(M−x)^p` capped finite; 0 in safe zone, wall at comm edge; silent floor |
| — | **Mission-safety as a brain INPUT** | (intended in architecture) | **🔜 OPEN BUILD ITEM** | today enforce-only; wire λ̂₂/barrier-proximity INTO the L4/L3 head (role head reads only belief `z`) |

**Net:** ideas #4's *mechanism* is settled-closed at scale; **idea #7 (multi-phase) is now the spine**, with
delivered-coverage + the barrier floor + the safety-as-input wiring as its supports. Full arc + exact
numbers: `JOURNEY.md` 2026-06-27; plan: `EXPERIMENT_PLAN.md` §0 + Phase 2′.

---

## ⮕ METHOD-DIRECTION verdicts — 7-search consolidated review (2026-06-27 · `STRATEGY.md`)

A 7-search literature review of the *method* directions we'd been weighing (which cut across the idea numbers
above) lands four verdicts that **revise the build plan**. These concern the **how-to-train / how-to-structure**
axes, not the per-mechanism ideas. Full reasoning + the complete reference list live in **`STRATEGY.md`**; the
load-bearing citations are below.

| Direction | Key citation | Literature verdict (one line) |
|---|---|---|
| **Hierarchy / the L4 strategy→role→skill→action tower** | Nachum et al. 2019, arXiv:1909.10618 (+ 2-level norm: FeUdal arXiv:1703.01161 / HIRO arXiv:1805.08296 / SOL arXiv:2509.00338) | **NOT supported — cap at 2 LEARNED levels** (goal-selector over GNN); hierarchy's benefit is *exploration, not the structural tower*, and only 2-level manager/worker scales. Roles/phases EMERGE (boids/Couzin: Reynolds 1987 / Couzin et al. 2002); the explicit L4 phase head + discrete skill library are downgraded to **conditional** on the §0′ test failing. |
| **ES as the optimizer / deception escape** | Salimans et al. 2017, arXiv:1703.03864; Conti et al. 2018, arXiv:1712.06560 | **FOLKLORE for deception — demote.** Plain ES collapses to the same degenerate optima; the real lever is **directed novelty/diversity exploration, which works on PPO too.** ES survives only inside a MERL-style hybrid (Majumdar et al. 2020, arXiv:1906.07315) *if* the flat path stalls — interleaved, never a one-shot weight handoff. |
| **Quality-Diversity (MAP-Elites) as trainer** | Mouret & Clune 2015, arXiv:1504.04909; Engebråten et al. 2020, arXiv:2007.08656 | **DEMOTED to a frontier-mapping / deception-escape *diagnostic*** on a compact controller (cov+connectivity descriptors are well-precedented). **NOT** the trainer for the deep GNN (QD+GNN-at-scale unexplored). Produces the cov↔conn Pareto frontier as a deliverable. |
| **Curriculum (small→large) / scale-transfer** | Agarwal et al. 2025 (LPAC), arXiv:2401.04855; Long et al. 2020 (EPC), arXiv:2003.10423 | **SOUND but must be FIXED:** connectivity must bind at *every* rung (incl. small), **pin density** across rungs, **100-step budget binds at every rung**, and use **multi-scale fitness** (don't train-once-small-then-transfer — the sparse-small end is the weak corner). |

**Net:** the corrected next move is **not** "build the L4 tower" — it is the **§0′ flat-baseline falsification
test** (goal-head + GNN + learned role latent + hard-connectivity shell + delivered-coverage), measuring whether
labour-division + the disperse↔gather rhythm **emerge** (role-diversity per Hu et al. 2022, arXiv:2207.05683)
before any structure is added. See `STRATEGY.md` (apparatus), `EXPERIMENT_PLAN.md` §0′ (the run),
`agent_architecture.md` (correction box).
