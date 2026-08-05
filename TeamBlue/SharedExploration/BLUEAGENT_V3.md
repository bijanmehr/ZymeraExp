# SuperBlue v3 — design ("pay differently, aim differently, one wire")

Converged 2026-07-04 through the v1/v2 critique arc. Supersedes the role-picker-central framing
of `COGNITION_DESIGN.md` for the coverage substrate. Verdict lineage: the stack was built from
single-agent *discovery + navigation* primitives (frontier + A\*) that structurally fight
connectivity and cause the flood; the mission's native primitives are **partition + marginal
credit**. The lever is **signal/formulation + warm-start, not new modules.**

**Governing rule.** Parameter-shared policy is KEPT (for scale/transfer). Symmetry is broken by
**inputs + reward, never weights**. Every change below is a signal, a capability, or one wire —
no new learned module except a single small decision head.

## Locked decisions (2026-07-04)
- **Regime: `cover_r=0` (visit-only).** Coverage credited only for cells actually *visited*.
  Oracle caps ~72% here → the metric is **%-of-oracle**, not absolute 90/90. This is the
  discriminating regime: the go-*through* rake is essential and A\*-dart under-covers, so it
  directly tests the executor claim.
- **Partition source is a first-class fork** (run both, both connectivity-constrained):
  - **P-scaffold** — scripted CVT partition as scaffold + policy learns only the connectivity
    *residual* (when to hold / hand off), paid by Dᵢ. Symmetry broken *by construction*.
  - **P-learned** — policy learns to divide space from scratch + Dᵢ + connectivity constraint.
  - The P-scaffold vs P-learned gap *is* the emergent-vs-authored test, scoped to the partition.
- **Connectivity is soft** (β·λ₂ inside G / Lagrangian), **no hard mask** (breakage must stay
  possible for the resilience thesis; near-perfect connectivity is a *target blue optimizes*,
  not a guarantee).

## Data flow
`belief → CVT partition (your cell) → learned decision {rake · hold · handoff} → scripted motion → Dᵢ credit`

## Capabilities (scripted, always-on, 0 learned params)
| Capability | Status | Role in v3 |
|---|---|---|
| SLAM / occupancy belief | KEEP | the map (a met prerequisite, not a lever) |
| geodesic field (BFS on belief) | KEEP | true reach cost (walls make Euclidean lie) |
| **CVT partition** of connected-reachable region by agent position | **ADD** | each agent owns a distinct cell → symmetry broken by construction; replaces frontier as the coverage aim |
| **boustrophedon rake** over your cell | **ADD** | go-*through* coverage motion; replaces A\*-dart (essential at cover_r=0) |
| **λ̂₂ / Fiedler-centrality** of self in comm graph | KEEP cap · **ADD as input** | "how load-bearing am I" — the symmetry source, now wired to the decision (see `FidlerValueEstimation`) |
| A\* | DEMOTE | transit + warm-start teacher only, not the coverage motion |
| frontier | SCOPE | discovery only (find undiscovered rooms → extend the partition), never the coverage signal |
| role module / ES individuation | **REMOVE** | roles are outcomes of partition+credit, not authored |
| extra perception features (attn/einfo/boundary) | **FREEZE** | proven null levers; stop feeding perception |

## The one learned decision (shared policy, KEEP)
- **in:** `my cell · λ̂₂-centrality · neighbor claim-bits · my cell's coverage-saturation`
- **out:** select `{rake my cell · hold my hinge · advance to a claim-request}`
- Explorer/relay are **labels we measure** on this selection, not inputs. Arm B tests whether
  even this minimal selection needs an explicit role head.

## Coordination (message, ADD)
**Claim-bit** — broadcast "covering cell X" (dedup) + "need relay at Y" (handoff request).
Converts the joint relay-handoff into two *unilateral* moves that Dᵢ can credit (unilateral
marginal credit is structurally blind to joint deviations; the claim-bit linearizes them).

## Training signal (the real lever, ADD)
- **Difference reward** `Dᵢ = G(team) − G(team∖i)`, `G = coverage + β·λ₂` (smooth λ₂, not the
  connected/not indicator → Dᵢ_conn = agent's Fiedler-centrality: smooth, bounded, computable in
  CTDE). Pays hinges; self-sizes relays (a redundant relay's Dᵢ≈0 → it defects to cover).
- **Warm-start** — BC from the connected-sweep oracle rake → Dᵢ fine-tunes. The proven ~26× lever.

## Build order (oracle-gated)
1. **Connected-sweep oracle (cover_r=0).** Settles partition-vs-frontier (does it rake or chase
   edges?), the true relay count, and the coverage ceiling. *Is* the warm-start teacher **and**
   the %-of-oracle denominator **and** the scripted baseline (task #56).
2. **v3-min** = partition + rake + Dᵢ + λ̂₂-input + claim-bit + warm-start, P-scaffold first.
3. **Ablation ladder** (below).

## Ablation ladder — each arm isolates one verdict-claim (no confounds)
Additive stack, P-scaffold partition:
- `A0` v2 control (frontier + A\* + shared reward)
- `A1` +partition (frontier→CVT) — explore-signal swap alone
- `A2` +Dᵢ — does marginal credit kill the flood?
- `A3` +λ̂₂ input — does the source wire help hinge-holding?
- `A4` +claim-bit — does handoff-linearization help?
- `A5` +warm-start = **full v3**

Forks at the top of the stack:
- `A5-learned` — swap P-scaffold → P-learned (the emergent-vs-authored partition test)
- `B` — +explicit role head on A5 (falsification: if it adds nothing, roles are confirmed dead)

**Metrics per arm:** %-of-oracle coverage · connectivity-hold rate · per-agent Dᵢ distribution ·
realized relay-count vs oracle · role-MI (is the selection bimodal / does labor divide?).

## Scale ladder
Build/tune at **16²/4**, validate transfer at **32²/10** (density-pinned), then a fixed-world
N-sweep (does adding agents divide labor or flood — the redundancy→1 test).

## Codebase touchpoints (verify API before building)
- Sim/env + rollout: `zymera` pkg (comm-coverage env, `rollout`, comm_graph).
- Executor/agent stack: `ctde_v0` (current CTDE PPO); `es.py` retired from the critical path.
- λ̂₂ estimator: `FidlerValueEstimation` study (potential-adjacency + decentralized power-iteration).
- Oracle/ceiling probe: extend `compass_ceiling.py` (god-view oracle) to emit the connected-sweep
  rake + relay assignment at cover_r=0.
- Metric: %-of-oracle already the house metric (`experiment-findings-consolidated`).
