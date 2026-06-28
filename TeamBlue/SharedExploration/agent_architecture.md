# SuperBlue — Shared-Exploration Agent Architecture (working notes)

**Goal.** A "superblue" TeamBlue agent for **shared exploration** that pushes the
**coverage ↔ connectivity** trade-off as far as achievable, by combining a set of mechanisms into one
modular agent (not a single trick — the *combination* is the point).
**Concrete target:** reach **≥ 90% coverage AND ≥ 90% connectivity** at the **100-step** horizon, across
the scale ladder (10²/2 · 16²/4 · 24²/6 · **32²/10**). Day-by-day progress is logged in `JOURNEY.md`.

**Status.** Understanding-capture from the ongoing design discussion (2026-06-25); **updated 2026-06-27**
with the **L4 strategy/phase layer** (above the role-picker), the **delivered-coverage** objective, and the
**Hyper-Singularity barrier** floor — the pivot after per-step connectivity guardrails were settled-closed
(see `JOURNEY.md` 2026-06-27 + `EXPERIMENT_PLAN.md` §0). This reflects the agent **decomposition**; many
internals and the learning-objective framing are deliberately open ("more later") and collected in *Open
points*. Lines marked *(interp.)* are my interpretation/mapping, not yet confirmed.

**The shape of the agent.** A new **L4 phase layer** now sits **above** the hub: it picks the team
**`{disperse ↔ gather}`** phase as a temporally-extended option, and the **role picker (L3) is the central
hub *within* that phase**. Everything upstream exists to inform them (what the agent knows, what it's trying
to do, how safe it is); everything downstream is *called by the role it picks* (the tools/controllers that
turn a role into concrete env actions). Read the modules below with that in mind.

---

> ## ⚠️ CORRECTION (2026-06-27) — the 4-level tower is NOT supported by the evidence; cap at 2 learned levels
>
> A 7-search literature review (consolidated in **`STRATEGY.md`** — see it for the full citation list) lands a
> verdict that **supersedes the "L4 strategy/phase tower is the next build" direction recorded above and in
> the L4 sections that follow.** Read the architecture below as the *design we explored*, with this correction
> on top.
>
> **The corrected design (what the evidence supports):**
> - **Cap structure at 2 LEARNED levels — a goal-selector over a GNN backbone — not a 4-level
>   strategy→role→skill→action stack.** Hierarchy's measured benefit is *exploration + temporally-extended
>   action*, **not** the structural tower itself (Nachum et al. 2019, arXiv:1909.10618); the only depth that
>   works at scale in the HRL literature is **2-level manager/worker** (FeUdal, Vezhnevets et al. 2017,
>   arXiv:1703.01161; HIRO, Nachum et al. 2018, arXiv:1805.08296; SOL, arXiv:2509.00338). A coarse spatial
>   **goal/region action head over the GNN** is the *one* temporal-abstraction the evidence cleanly backs — and
>   it is exactly the lever that fixes our 1-step-move ceiling. **This supersedes "the L4 gather is the next
>   build."** The L4 `{disperse↔gather}` *phase* and the discrete `{explorer, relay}` *skill library* are the
>   most anthropomorphic, least-evidenced layers — **do not hand-build them as brain levels.**
> - **Connectivity is a HARD CONSTRAINT / action-mask safety shell — NOT a brain level.** It encodes a true
>   world invariant (the comm graph must hold), so it belongs as a constraint the policy operates *inside*, not
>   as something the L4/L3 head reasons over (our own +20-pt hard-guardrail evidence; "RL Connectivity
>   Maintenance", arXiv:2109.08536).
> - **Roles AND phases EMERGE — they are measured outcomes, not authored modules.** Keep roles as a **learned,
>   regularized-to-be-readable latent** (ROMA-style MI — Wang et al. 2020, arXiv:2003.08039), let the
>   explorer/relay split and the disperse↔gather rhythm *fall out* of a near-flat shared policy under
>   **delivered-coverage** (optionally with a CDS-style diversity regularizer — Li et al. 2021,
>   arXiv:2106.02195). Swarm intelligence's founding result is that sophisticated collective behavior — even
>   *phase transitions* of exactly the gather/disperse kind — emerges from flat local rules with no cognitive
>   layers (boids: Reynolds 1987, SIGGRAPH; swarm↔torus↔parallel-group phase transitions from tuning ONE
>   parameter: Couzin et al. 2002, *J. Theor. Biol.*). The bitter lesson — "building in how we think we think
>   does not work in the long run" (Sutton 2019) — names the strategy→role→skill→action stack as a definitional
>   instance of the trap.
> - **Measure, don't legislate.** Roles/phases are graded as **outcomes** (MI(role;behavior), labour-division
>   redundancy → 1, a visible breathe-out/breathe-in; role-diversity per Hu et al. 2022, arXiv:2207.05683) on
>   the flat baseline FIRST. If role-diversity is low — as expected for *homogeneous* coverage — even one
>   explicit role level may be inert, so skip it.
> - **The program-specific reason this matters most:** a hand-designed hierarchy **short-circuits the very
>   phenomenon Zymera exists to study** — *whether* roles/phases emerge, *how* they propagate micro→macro, and
>   *how an adversary perturbs them*. Flat-policy + emergence keeps the macro structure a measured outcome (more
>   honest for a resiliency study) and gives the red team a *real emergent target* instead of an installed
>   scaffold.
>
> **Net:** the **L4 tower / discrete-skill-library framing in the sections below is downgraded to "explored,
> not endorsed."** Build the 2-level goal-selector-over-GNN, put connectivity in the hard shell, and let roles
> + phases emerge under delivered-coverage — then run the falsification test in `STRATEGY.md` before adding any
> third level. The barrier floor and delivered-coverage objective survive (they make the rhythm *emerge*); the
> *hand-built phase head* and *typed skill library* do not.

---

## Design principle — a multi-level cognitive platform (brain-like)

This is **not a flat pipeline** but a **multi-story platform** with **different levels and types of
cognition** — deliberately brain-like. Higher levels reason about **what** to do (slow, abstract) and
**delegate intent** to lower levels that decide **how** and then **act** (fast, concrete). Crucially,
**high levels do not micromanage** — they hand down a phase/role/goal and trust the level below to execute,
each level running at **its own timescale** (phase decisions are *very* sparse; role/goal decisions sparse;
control runs every step). This mirrors layered robot control and brain-inspired architectures (deliberative →
executive → reactive; subsumption; cortex → basal-ganglia / cerebellum → reflex) and maps cleanly onto
**hierarchical RL** (a high-level option/role selector over low-level controllers). Different **types** of
intelligence coexist across the levels — learned cognition, classical estimation, heuristic planning/reflex.

| Level | Cognition | Timescale | Modules |
|---|---|---|---|
| **L4 — strategy / mission-phase (NEW)** | *"what should the TEAM be doing right now"* — picks the **phase `{disperse ↔ gather}`** (≡ explore ↔ deliver) as a **temporally-extended option** | **very slow** (commit ~5–10 steps) | **phase selector** · reads connectivity-danger (λ̂₂ / barrier proximity) + belief |
| **L3 — deliberative / cognitive** | *"what's happening / what to do"* — abstract, mostly **learned**, *within the L4 phase* | slow (sparse) | goal · **role picker** · mission-safety reasoning · KB belief |
| **L2 — executive / tactical** | *"how to achieve the intent"* — planning & exchange, **heuristic or learned** | medium | the role's **tool** — A* / BFS *(heuristic)* or frontier-attention / λ₂-estimator *(learned)* · comm exchange |
| **L1 — reactive / reflexive** | *"act safely now"* — concrete, mostly **heuristic** | fast (every step) | movement · collision-avoidance · operation-action · raw perception + sensor-fusion |

The **L4 phase layer** sets the team rhythm; the **role picker (L3)** sets intent *within that phase*; an
**L2 tool** plans toward the goal; **L1 control** executes safely — **no level micromanages the one below.**

### Why L4 exists — resolving coverage ↔ connectivity in TIME *(added 2026-06-27)*

L4 was added after an empirical dead-end (see `JOURNEY.md` 2026-06-27 + `EXPERIMENT_PLAN.md` §0): **every
per-step connectivity guardrail** (hard action-mask, soft global-λ₂, local degree/edge-margin, under
fixed / Lagrangian / PID weighting) **failed to fix coverage at scale** — they all "hold" connectivity only
by **HUDDLING** (a dense clump satisfies the connectivity signal for free, so the mechanism's penalty/dual is
≈ 0 and it cannot push the team apart). **The fix is to resolve coverage ↔ connectivity in TIME, not every
step:** the team **fans out to cover (disperse), regroups to share (gather), and repeats.** L4 owns that
**phase as a temporally-extended option** — committed for ~5–10 steps, *not* re-decided per step. It is
**decentralized per-agent, cohering via the shared belief** — making the **micro→macro bridge an explicit
module** (each agent's local phase choice composes, through the belief, into a team-level rhythm). **Build
STAGED:** add the L4 phase head on top of the existing role-picker, keep gather/disperse **skills scripted
first (learn only the *timing*)**, grow learned-ness down the stack later — **do NOT train all 4 levels at
once.** The disperse→gather rhythm is reinforced by the **delivered-coverage objective** (coverage counts
only when in contact to share it — `PersistantNetwork`), which makes the breathing **emerge from the
objective with no connectivity penalty**, and floored by the **Hyper-Singularity barrier** (below).

---

## Design constraint — scale-invariance (so warm-start works)

**Warm-start** = train at a small scale, then use those weights to **initialize the next rung up the
ladder** (10²/2 → 16²/4 → 24²/6 → **32²/10**) as a curriculum — so the hard 32²/10 push is *bootstrapped*
from easier scales instead of trained from scratch. **For this to be possible at all, the agent must be
scale-invariant by construction** — the *same* parameters must apply across **both grid size and agent
count.** A hard constraint on every module:

- **Perception** — a **local / relative window** (fixed sensing radius), never the absolute full grid.
- **KB / belief** — **size-invariant** representation + update (GCRN-style message passing transfers across
  scales; a flat full-grid belief does **not**).
- **Comms / aggregation** — **agent-count-invariant** (GNN / attention / parameter-sharing over neighbors,
  not a fixed-`N` vector).
- **Role-picker, goal, tools** — operate on **local / relational** features, not absolute coordinates or a
  fixed agent count.

Then the **study (must validate):** does warm-start actually **transfer and help reach 90/90** at 32²/10
(vs from-scratch, vs zero-shot)? — logged per rung in `JOURNEY.md`.

---

## The agent — core modules

One agent = a composition of the modules below; the team is `N` such agents (decentralized, comms over a
limited range). Some modules are **learned stacks**, some are **heuristic tools**.

### Upstream — what feeds the role picker

1. **Perception / sensing stack** — the agent's **own firsthand sensing** of its surroundings — *like a
   camera / Lidar / IMU*. A **sensing-radius** parameter bounds the local view (e.g. a 3×3 window):
   occupancy / walls, frontier, neighbors, own position, etc. May include a **sensor-fusion** sub-stack
   (e.g. an EKF) when there are multiple/noisy sensors. **Independent of comms** — firsthand, not received
   over the radio.

2. **Knowledge base (KB)** — the agent's **memory / running belief** of world + team, fusing **independent
   streams**: (a) its **own perception** accumulated over time, (b) **information from neighbors over
   comms**, and (c) **priors / external knowledge** (beliefs, given maps, anything prior). Perception and
   comms are separate inputs — the KB is where they (and priors) meet.
   *Open:* the **fusion method**; how the policy reads/affects the KB; the KB's **shape / size / form /
   type**; and how an **adversary** or the KB's **limitations** affect the mission.

3. **Comms (radio) + comm stack** — the **radio**: what is transmitted / received over the topology.
   Involves a **ping-pong** (handshake/exchange) mechanism, since the **comm radius is usually larger than
   the sense radius**. The comm **stack** is one of (to decide / compare):
   - **simple message**,
   - **message composition**,
   - **learnable communication** — the agent learns *what* to send, or
   - **λ₂ (Fiedler) estimation under partial observation** — estimate the team's algebraic connectivity
     locally, from partial info.
   *(interp.)* feeds the KB. Learnable messaging is a lab contract extension (the *comms seam*);
   λ₂-estimation is a belief/perception block.

4. **Mission-safety module** — a **local** signal: how mission-safe the agent currently is, and **which
   actions would jeopardize it**. **Local by design**, so an agent with partial observability can compute
   it; a **local anomaly detector** can plug in here later. Method **TBD** (relates to connectivity health
   — degree / λ₂ / giant-component — and the hard guardrail).
   > **⚠️ Architecture gap — mission-safety is ENFORCEMENT-only today, NOT a brain INPUT** *(2026-06-27)*.
   > As intended above, mission-safety should be an **input the L4 phase / L3 role-picker ingests** (so the
   > brain *reads* connectivity danger and *decides* gather vs disperse / explorer vs relay accordingly). In
   > the current `ctde_v0` implementation it is **only an enforcement mechanism** — the `MissionSafety` config
   > block is an **action-mask / reward-penalty** (and the per-step variants are now **settled-closed** as a
   > coverage fix, see `EXPERIMENT_PLAN.md` §0) — while the **role head in `nets.py` conditions only on the
   > belief `z`**. **OPEN BUILD ITEM:** wire the **connectivity-danger signal (λ̂₂ / barrier proximity) as an
   > explicit INPUT** to the L4/L3 head. Until then the brain can *enforce* safety but cannot *reason about
   > it* when picking the phase/role.

5. **Goal** — the high-level objective the agent is pursuing; it **affects the mission-safety parameter**
   and is an input to the role picker. Must be **broad and generalizable** across a wide range of
   shared-exploration missions.
   *(interp.)* e.g. a frontier region for *explorer*, a bridge position for *relay*.

### The hub

6. **Role picker** *(central, learned, high-level)* — takes the **KB**, the **goal**, and the
   **mission-safety** signal, and **picks a role** ({explorer, relay} for now; open-ended), **within the
   current L4 phase** (a *gather* phase biases toward relay/regroup, a *disperse* phase toward explorer). It
   **generates each action indirectly**: the picked role **calls the tool/controller appropriate to it** —
   e.g. *explorer* → an A* / frontier tool; *relay (stop & hold the connection)* → a different tool — which
   produce the final actions the env consumes. This is the **key learning component** and the agent's hub.
   *(Empirically: roles are the proven huddle-fix at 16²/4 — `role_expl_relay` ≈ 90.9 % cov vs ≤5.6 % for
   `role_off` — but the win does not transfer up the ladder; see `JOURNEY.md` 2026-06-27. Hence L4 above.)*

### Downstream — what a role calls

7. **Role tools — control / planning / estimation** — to **achieve its goal**, the chosen role calls a
   **tool**, which may be **heuristic *or* a learnable stack**. Heuristic: path-finding (BFS / DFS / A*),
   movement direction, collision-avoidance. Learnable examples: a **frontier-attention** tool for
   *explorer*, a **λ₂ / Fiedler-estimator** tool for *relay*. Reflexive collision-avoidance still guards
   the final env-ready move. So a **role behaves like a skill / option** — itself heuristic or learned, and
   swappable (this is also where the "frontier-attention" and "λ₂" ideas plug in, as per-role tools).

8. **Operation action** — the **mission action itself**: literally *doing something* at the agent's
   current cell — e.g. a **deep scan** or **mining** — after which that cell is "done" (future: some cell
   types may require the operation **multiple times**). Inter-agent operation actions may be **homogeneous
   or heterogeneous** (TBD). Flexible-yet-simple for now; this is the mission interface and indicates the
   **output of the mission** (for shared exploration: the coverage / observation outcome).

### The connectivity FLOOR — Hyper-Singularity barrier *(reward term, added 2026-06-27)*

A **silent connectivity floor** that **composes UNDER the L4 phase rhythm** — *not* another per-step pull
inward. A per-agent reward term on **nearest-neighbour distance**, `f(x) = k·relu(x − a)² / (M − x)^p`
**CAPPED finite (RL-safe)**: **exactly 0 in the safe zone (`x < a`)**, an explosive-but-finite **wall as a
link nears the comm edge `M`** (the break range), saturating at `barrier_cap`. Config knobs
(`reward.barrier_*`): `barrier_weight = k` (**0 ⇒ OFF / no-op**), `barrier_a` (launch, default `comm_r·0.6`),
`barrier_M` (wall, default `comm_r`), `barrier_p`, `barrier_cap`. **It is a FLOOR the team can ride *out* to,
not a per-step force toward the centre — so it is NEVER tested in isolation** (alone it is itself a per-step
signal and would re-huddle, exactly the failure of `EXPERIMENT_PLAN.md` §0). It earns its keep only
*underneath* the L4 disperse↔gather rhythm and the delivered-coverage objective, as the hard backstop that
keeps a dispersing team from snapping the graph.

---

## Data flow (sketch) — role picker as the hub

```
  perception (own camera/Lidar/IMU, firsthand) ─┐
                                                ├─▶ KB (memory: own + neighbors + priors)
  comms (radio, ping-pong) ⇄ neighbors ─────────┘                 │
                                                                  ▼
   mission-safety / λ̂₂ / barrier-proximity ──▶  ┌───────────────────────────────┐
                              (OPEN: as INPUT)  │   L4 PHASE  {disperse↔gather} │   commit ~5–10 steps
                                                │   (temporally-extended option)│
                                                └───────────────────────────────┘
                                                                  │  sets the team phase
                                                                  ▼
            goal ──────────────────────────▶  ┌───────────────────────────────┐
                                              │   ROLE PICKER  (within phase)  │
   mission-safety (local) ───────────────────▶│   (central · learned)         │
                                              └───────────────────────────────┘
                                                                  │  picks role {explorer, relay, …}
                                                                  ▼
                               role calls its tool ──  explorer → A* / frontier
                                                    └─ relay    → hold-connection
                                                                  │
                                                                  ▼
                                    control / movement / collision  (heuristic)
                                                                  │
                                                                  ▼
                              env move   +   operation action ──▶ mission output
                                            (delivered-coverage: counts only when in contact)
```

KB (fed by perception + comms + priors), the **goal**, and the **mission-safety** signal all flow **into
the role picker**; the picked role **calls the matching tool/controller**, which produces the movement and
the operation action that realize the mission outcome.

---

## Learned vs. heuristic (current read)

- **Learned:** the **role picker** (central); optionally the **KB/belief** (fusion, λ₂/belief estimation),
  the **comm stack** (if learnable comms), and **per-role tools** — a role's tool can itself be a learnable
  stack (e.g. **frontier-attention** for *explorer*, a **λ₂-estimator** for *relay*). The **goal**
  selection may be learned or heuristic (TBD).
- **Heuristic:** path-finding / movement / collision-avoidance when a role uses a scripted tool (A* / BFS /
  DFS); sensor-fusion (EKF) is classical.
- A **role thus behaves like a skill / option** — heuristic or learned, swappable. **SuperBlue** = the
  *combination* of learned cognition (the picker) + the best per-role tools, tuned for the coverage ↔
  connectivity frontier.

## The lab ↔ experiment boundary

- **`zymera` (the lab) runs the WORLD only.** Given an action, it returns the next **observation + state +
  raw signals** (explored, seen-by, comm graph, metrics) and transports comms over the topology. Its whole
  job here: the **env**, the host **sensor**, and **`rollout`**. The env is **reward-agnostic**.
- **The experiment owns everything else.** *All* agent modules (perception → KB → role picker → tools →
  control → operation-action) are composed into the `policy(obs, state, key) -> (action, state)` **here**,
  and trained by a stack written **here**. One-way dependency: experiment → `zymera`; nothing flows back.
- **Reward engineering is totally separate.** The coverage ↔ connectivity objective and any shaping are
  **engineered in the experiment** from the env's raw signals — *not* baked into the env. (Blocks or reward
  terms that prove out may later **graduate** into `zymera.nets` / `zymera.missions_terms`, but they
  originate and live here first.)
- The role → tool/controller path uses the planned **goal→controller (hierarchy) seam**; learnable comms
  uses the planned **comms seam**.

## Build & run plan (direction)

- **The agent is a *configurable composition of swappable strategy modules*.** Each module above is a
  choice axis (KB type, comm stack, role set, per-role tool = heuristic vs learned, mission-safety method,
  …); a "strategy" is one selection across the axes.
- **Trainer-agnostic.** The agent exposes a parameter surface that **any trainer can optimize** — PPO
  (+CTDE / IPPO) and ES alike — so the *same* agent is driven by different learners and compared. The
  trainer is itself a run-plan axis.
- **A big run plan, not one MVP.** The goal is a **large, configurable sweep** that tests **many strategy
  combinations at once** to find the best coverage ↔ connectivity trade-off. The matrix gets designed
  next, together.

**Campaign target & setup (fixed).** Reach **90 / 90** (coverage / connectivity) at the **100-step**
horizon (mission requirement); scale ladder **10²/2 · 16²/4 · 24²/6 · 32²/10** (fixed density); a
**warm-start up the ladder** is a study (does small→large transfer help?); **every run carries a saved
config file** (exact params) so any result is reproducible. Daily log: `JOURNEY.md`.

**Strategy axes (candidate sweep dimensions):**

| Axis | Options (extensible) |
|---|---|
| **L4 phase layer (NEW)** | **off · scripted-timing · learned — `{disperse ↔ gather}` option, commit `k≈5–10` steps** |
| KB / belief | flat certainty-field · learned graph belief (GCRN) · relational belief-graph · (+priors) |
| Comm stack | simple msg · msg composition · learnable comms · λ₂ estimation |
| Role set | {explorer, relay} → more later |
| Explorer tool | A* / frontier *(heuristic)* · frontier-attention *(learned)* |
| Relay tool | hold-connection *(heuristic)* · λ₂-estimator *(learned)* |
| Objective | coverage · **delivered-coverage** (`PersistantNetwork` — makes disperse→gather emerge, no conn penalty) |
| **Connectivity floor (NEW)** | **Hyper-Singularity barrier `barrier_weight` {0=off · >0}** (+ `a/M/p/cap`) — silent floor, **composes UNDER L4, never alone** |
| Mission-safety | degree-floor · λ₂ · giant-component · learned — **TBD**; **⛔ per-step ENFORCEMENT settled at scale (§0); OPEN: wire as a brain INPUT** |
| Trade-off | ~~λ-sweep · constrained~~ ⛔ **per-step settled (§0)** · **resolve in TIME via L4 phase** |
| Trainer | PPO (IPPO / MAPPO+CTDE) · ES · **Quality-Diversity (MAP-Elites)** · multi-objective/constrained — **try all** |
| Scale | **10²/2 · 16²/4 · 24²/6 · 32²/10** (fixed density) · comm_r |
| Warm-start | none · warm-start up the scale ladder *(study)* |

---

## Open points (to refine — "more later")

- **Trade-off objective** — Pareto λ-sweep / preference-conditioned / constrained / single weighted reward.
  **Decide after the first slice runs** (once we've seen raw coverage/connectivity behavior).
- **KB** — fusion method, shape/size/form/type, how the policy uses/affects it, and how adversary /
  limitations affect the mission.
- **Mission-safety method** — how it's computed locally and how it gates actions; a **local anomaly
  detector** plugs in here later. **Needs its own discussion (TBD).**
- **Comm stack** — simple message / message composition / learnable comms / λ₂-estimation. *Choose / compare.*
- **Goal** — keep it broad & generalizable across shared-exploration variants.
- **Roles** — beyond {explorer, relay} later.
- **Operation action** — homogeneous vs heterogeneous across agents; cells that need multiple operations.
- **Learning paradigm(s)** — **trainer-agnostic agent**, **try all**: PPO (IPPO / MAPPO+CTDE) · ES ·
  Quality-Diversity (MAP-Elites) · multi-objective/constrained — the *same* agent, compared. (Per-module
  supervised pretraining possible.)
