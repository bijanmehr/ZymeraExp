# Zymera — Agent Architecture & Its Evolution

The canonical record of the Zymera agent: what it is **now**, the **substrate** every version shares, and
**how the design got here** — through a multi-level "brain", a hard correction, a rejected divide-and-conquer
branch, and finally the T5 three-clock. Written to replace the scattered June design notes
(`agent_architecture.md`, `LOCALITY_DESIGN.md`, `COGNITION_DESIGN.md`) with one document; the day-by-day
narrative lives in `../journal/JOURNEY.md`, the current parameters in `T5_DESIGN.md`.

---

## 1. The shared substrate (constant across every version)

One agent = a composition of modules; the team is `N` such agents, decentralized, communicating over a
limited range. These pieces did **not** change as the "brain" above them was redesigned — they are the
environment's *bridge* made concrete.

| Module | What it is |
|---|---|
| **Perception** | Firsthand local sensing — a small window (sense_r), never the full grid. Independent of comms. |
| **KB / belief** | The agent's running, size-invariant memory: own perception over time **+** neighbor messages **+** priors, fused into one occupancy/coverage belief. |
| **Comms (bridge)** | A range-limited, time-varying graph. Neighbors gossip the belief; aggregation is **trust-by-default**. Carried by an **LPAC-KB GNN** (CNN local-perception → 2-round message passing), which is what makes the policy **size-invariant** — same parameters at 16²/4 and 32²/10. |
| **Mission-safety** | A *local* connectivity-danger signal (λ̂₂ / neighbor-distance). Today it **enforces** (soft λ₂ guardrail, no hard mask by preference); wiring it as an explicit **brain input** is still open. |
| **Operation / move** | The actual env action + a hard collision veto. |

**Design invariant — scale-invariance by construction.** Perception is local, the belief is
count-invariant, aggregation is permutation-equivariant. This is what lets warm-start up the ladder work and
lets a frozen policy transfer to unseen sizes.

---

## 2. The current architecture — the T5 three-clock

Each agent runs three "clocks" over the shared belief `z`. Only **one level is learned end-to-end** (L3); L2
is a distilled network that behaves like a classical planner; L1 is scripted.

```
   belief z  (LPAC-KB GNN over the comm graph)
        │
        ▼
  ┌────────────────────────────────────────────┐
  │ L3 · GOAL     "where to go"                 │   learned head off the belief      ← the open lever
  └────────────────────────────────────────────┘
        │  goal cell / region
        ▼
  ┌────────────────────────────────────────────┐
  │ L2 · PLANNER  "how to get there"            │   distilled MVProp value-iteration ← done + proven
  │               V = max(r, γ·w·max_4nbr V)    │   (routes around known walls)
  └────────────────────────────────────────────┘
        │  γ^dist potential field
        ▼
  ┌────────────────────────────────────────────┐
  │ L1 · TRACKER  "the actual move"             │   climb the field + collision veto ← done (scripted)
  └────────────────────────────────────────────┘
        │
        ▼
     env move
```

- **L2 is solved and proven.** The distilled planner's forward pass *is* value iteration over one learned
  passability field `w = σ(conv[blocked, goal])`; measured **w-recovery 98–99.6%** on unseen maps, so it
  recovers the exact shortest-path wavefront — optimal by construction, not by luck.
- **L3 is the open lever.** The remaining coverage headroom on walled maps is a *connectivity tax* + weak
  team coordination, i.e. the goal level, **not** navigation.
- Trained with **MAPPO / CTDE** (central critic, discarded at deploy); connectivity held by a **soft λ₂
  guardrail** (no hard mask, by preference — a hard mask would kill the resilience study).

---

## 3. How it evolved

### v0 — flat reactive CTDE *(pre-June)*
A single shared policy with a **one-step move head** over the LPAC-KB backbone. Covered open floors well
(~86%) but got stuck on unseen walled maps (~25% zero-shot). The one-step head is the thread the whole
redesign eventually pulls on.

### v1 — the multi-level cognitive platform *(2026-06, `agent_architecture.md`)*
The first deliberate architecture: a brain-like **4-level stack**, each level slower and more abstract than
the one below, no level micromanaging the next.

| Level | Cognition | Timescale | Modules |
|---|---|---|---|
| **L4 — strategy / phase** | *what should the TEAM do* — pick `{disperse ↔ gather}` as a committed option (~5–10 steps) | very slow | phase selector |
| **L3 — deliberative** | *what to do* — mostly learned, within the phase | slow | goal · **role-picker (hub)** · mission-safety · KB |
| **L2 — executive** | *how* — planning / exchange, heuristic or learned | medium | the role's tool: A*/BFS or frontier-attention / λ₂-estimator |
| **L1 — reactive** | *act safely now* | fast | move · collision-avoidance · perception |

The **role-picker** was the hub: it read the belief + goal + safety and picked a role (`{explorer, relay}`),
which *called a tool* to produce the move. Data flow:

```
  perception ─┐
              ├─▶ KB (own + neighbors + priors) ─┐
  comms ⇄ ────┘                                  ▼
  mission-safety / λ̂₂ ─▶ [ L4 PHASE {disperse↔gather} ] ─▶ [ ROLE PICKER (learned hub) ] ─▶ role's tool
                                                                            ▲                    │
                                                        goal ───────────────┘                    ▼
                                                                                  control + collision ─▶ env move
```

The L4 phase layer was added specifically as the intended fix for the **scale huddle** — resolve
coverage↔connectivity *in time* (fan out, regroup, repeat) rather than every step — supported by a
delivered-coverage objective and a "Hyper-Singularity" connectivity-floor barrier.

### The correction — cap at two learned levels *(2026-06-27, `STRATEGY.md`)*
A 7-search literature review retracted the tower **before it was built**. The load-bearing verdicts:

- **Hierarchy's benefit is exploration, not the structural tower** (Nachum 2019); only 2-level manager/worker
  scales. A coarse **goal head over the GNN** is the one temporal abstraction the evidence backs — and it is
  exactly the fix for the one-step-move ceiling.
- **Connectivity is a hard constraint, not a brain level** — it encodes a world invariant, so the policy
  operates *inside* it.
- **Roles and phases must EMERGE** — graded as measured outcomes, not authored. Boids/Couzin show even
  gather/disperse *phase transitions* fall out of flat local rules.
- **The rule:** impose structure only where it encodes a true task/world invariant, never a guess about how a
  mind should think. The program-specific reason: a hand-designed hierarchy *short-circuits the very
  phenomenon Zymera exists to study* — whether roles/phases emerge and how they propagate.

### v2 — the divide-and-conquer branch *(2026-06-28, `LOCALITY_DESIGN.md`) — REJECTED*
A competing design: *"a big world is many small worlds."* A **parallel local toolbox** — Voronoi partition +
in-cell frontier + difference rewards + a **hardcoded context rule** (no learned selector) — with the sharp
diagnosis that **the coverage wall is a spatial-partitioning / credit problem, and adding agents floods
rather than divides** (redundancy 3.7 → 7.9). The D&C toolbox was rejected, but three things survived it.

### v3 — the T5 three-clock *(2026-07 → 08, current)*
The honest **T1–T5** re-run localized the ceiling on the **one-step move head**, not comms. So: keep the
system fully learned, but replace the reactive head with a **distilled planner**. What carried through from
every prior version:

- **from the correction** → cap learned levels (T5 has exactly one, L3), connectivity as a soft constraint, emergence-as-a-measured-target;
- **from the D&C branch** → **difference rewards** (the exact submodular contribution) + the flooding-not-dividing diagnosis;
- **dropped for good** → the L4 phase tower, the role-picker hub, the discrete skill library, and the barrier floor.

---

## 4. Where each block-diagram / figure lives

| Artifact | Path |
|---|---|
| Actor / critic two-panel block diagram + model data-sheet | `../../report/architecture/architecture_panels.html`, `architecture_depiction.html` |
| Three-stack architecture diagram (animated) | `site/assets/figures/three-stack.gif` |
| Network-architecture figure | `presentations/dl-talk/network_architecture.png` |
| L3 reasoner design | `L3_REASONER_whitepaper.html` (this folder) |
| Current parameters | `T5_DESIGN.md` (this folder) |
| The formalism (micro/bridge/macro, Dec-POMDP) | `Docs/formalism.{tex,pdf}`, `site/theory.html` |

---

*Supersedes the deleted June design notes. The evolution's day-by-day detail — with exact numbers and the
negative results that drove each turn — is in `../journal/JOURNEY.md` (see the "architecture decision process"
entry). Related memory: `mvprop-planner-distillation-solution`, `emergence-framing-conditional-cooperation`,
`obligate-collaboration-taxonomy`, `architecture-audit-two-stacks`.*
