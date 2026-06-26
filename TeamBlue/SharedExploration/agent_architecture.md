# SuperBlue — Shared-Exploration Agent Architecture (working notes)

**Goal.** A "superblue" TeamBlue agent for **shared exploration** that pushes the
**coverage ↔ connectivity** trade-off as far as achievable, by combining a set of mechanisms into one
modular agent (not a single trick — the *combination* is the point).
**Concrete target:** reach **≥ 90% coverage AND ≥ 90% connectivity** at the **100-step** horizon, across
the scale ladder (10²/2 · 16²/4 · 24²/6 · **32²/10**). Day-by-day progress is logged in `JOURNEY.md`.

**Status.** Understanding-capture from the ongoing design discussion (2026-06-25). This reflects the agent
**decomposition**; many internals and the learning-objective framing are deliberately open ("more later")
and collected in *Open points*. Lines marked *(interp.)* are my interpretation/mapping, not yet confirmed.

**The shape of the agent.** The **role picker is the central hub**. Everything upstream exists to inform
it (what the agent knows, what it's trying to do, how safe it is); everything downstream is *called by the
role it picks* (the tools/controllers that turn a role into concrete env actions). Read the modules below
with that in mind.

---

## Design principle — a multi-level cognitive platform (brain-like)

This is **not a flat pipeline** but a **multi-story platform** with **different levels and types of
cognition** — deliberately brain-like. Higher levels reason about **what** to do (slow, abstract) and
**delegate intent** to lower levels that decide **how** and then **act** (fast, concrete). Crucially,
**high levels do not micromanage** — they hand down a role/goal and trust the level below to execute, each
level running at **its own timescale** (role/goal decisions are sparse; control runs every step). This
mirrors layered robot control and brain-inspired architectures (deliberative → executive → reactive;
subsumption; cortex → basal-ganglia / cerebellum → reflex) and maps cleanly onto **hierarchical RL** (a
high-level option/role selector over low-level controllers). Different **types** of intelligence coexist
across the levels — learned cognition, classical estimation, heuristic planning/reflex.

| Level | Cognition | Timescale | Modules |
|---|---|---|---|
| **L3 — deliberative / cognitive** | *"what's happening / what to do"* — abstract, mostly **learned** | slow (sparse) | goal · **role picker** · mission-safety reasoning · KB belief |
| **L2 — executive / tactical** | *"how to achieve the intent"* — planning & exchange, **heuristic or learned** | medium | the role's **tool** — A* / BFS *(heuristic)* or frontier-attention / λ₂-estimator *(learned)* · comm exchange |
| **L1 — reactive / reflexive** | *"act safely now"* — concrete, mostly **heuristic** | fast (every step) | movement · collision-avoidance · operation-action · raw perception + sensor-fusion |

The **role picker (L3)** sets intent; an **L2 tool** plans toward the goal; **L1 control** executes
safely — **no level micromanages the one below.**

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

5. **Goal** — the high-level objective the agent is pursuing; it **affects the mission-safety parameter**
   and is an input to the role picker. Must be **broad and generalizable** across a wide range of
   shared-exploration missions.
   *(interp.)* e.g. a frontier region for *explorer*, a bridge position for *relay*.

### The hub

6. **Role picker** *(central, learned, high-level)* — takes the **KB**, the **goal**, and the
   **mission-safety** signal, and **picks a role** ({explorer, relay} for now; open-ended). It **generates
   each action indirectly**: the picked role **calls the tool/controller appropriate to it** — e.g.
   *explorer* → an A* / frontier tool; *relay (stop & hold the connection)* → a different tool — which
   produce the final actions the env consumes. This is the **key learning component** and the agent's hub.

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

---

## Data flow (sketch) — role picker as the hub

```
  perception (own camera/Lidar/IMU, firsthand) ─┐
                                                ├─▶ KB (memory: own + neighbors + priors)
  comms (radio, ping-pong) ⇄ neighbors ─────────┘                 │
                                                                  ▼
            goal ──────────────────────────▶  ┌───────────────────────────────┐
                                              │   ROLE PICKER                 │
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
| KB / belief | flat certainty-field · learned graph belief (GCRN) · relational belief-graph · (+priors) |
| Comm stack | simple msg · msg composition · learnable comms · λ₂ estimation |
| Role set | {explorer, relay} → more later |
| Explorer tool | A* / frontier *(heuristic)* · frontier-attention *(learned)* |
| Relay tool | hold-connection *(heuristic)* · λ₂-estimator *(learned)* |
| Mission-safety | degree-floor · λ₂ · giant-component · learned — **TBD (own discussion)** |
| Trade-off | λ-sweep · preference-conditioned · constrained · single — **decide after first slice** |
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
