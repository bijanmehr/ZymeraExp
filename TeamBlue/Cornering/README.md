# Cornering — mission workspace (hazard-cordon)

**Mission #3** beside `../SharedExploration/` (coverage) and `../TetherRelay/`. **Status: not yet
designed** — parked for its own design session per Bijan (`SharedExploration/docs/planning/
T5_DECISIONS.md` §135, open TODO **#8**). No spec exists; this folder collects everything known so
far so the design session starts warm. When written, the spec lives here.

**Family: Act-dominant composite** (reclassified 2026-08-10): the deliverable is an intervention on
external agents (containment) via divided work under shared constraints, carried on an Organize
substrate — so with SharedExploration (Sense) and TetherRelay (Organize), the three missions span all
three taxonomy families.

## One line (current sketch — TODO #8 + the overview doc)

A pursuer team **splits into squads, each cornering one evader** (team vs team; sketched at ~10
pursuers vs 6 evaders). Reward = **MIN-payoff containment** of gap-seeking evaders — the team scores
on its worst-contained evader — so **every agent is pivotal** and the labor *must* divide. The covert
red is a pursuer that **opens a small gap on the escape side** while otherwise complying.

## Why it's the obligate-collaboration flagship (memory `obligate-collaboration-taxonomy`)

- **Litmus:** a mission *obligates* collaboration iff agent i's optimal action depends on agent j's
  **intention/plan**, not just j's state. Coverage fails the litmus (anti-overlap on shared state
  reaches the team optimum — the canonical *reducible* mission).
- Cornering is **reducible under shared/common-knowledge belief** (everyone runs the same
  deterministic perimeter partition) but **obligate under divergent private beliefs** — the
  interaction layer (sparse comms + partial obs) *manufactures* the irreducibility. Dead-center of
  the micro↔macro-via-interaction thesis.
- **Difference rewards collapse here:** when everyone is pivotal, `d_i = V(team) − V(team∖i) = V` for
  *all* i — zero per-agent differentiation. The mission therefore *requires* intention-sharing /
  correlated policies, not a better critic. This is where the deferred rich-L3 intention-sharing
  earns its keep (`T5_DECISIONS.md` §43).

## Two explicit gates before building

1. **The 4c horizon probe** (`SharedExploration/docs/planning/T5_EXPERIMENT_PLAN.md` §50): sweep the
   horizon on *coverage* and measure the flat-vs-rich-L3 gap. Tight time = *soft* obligation. If the
   gap opens → collaboration has a home on coverage **without** building cornering; if the null
   survives at short horizons → coverage is robustly reducible and cornering is **required**.
2. **Occlusion precondition** (memory `lit-coop-mission-selection-verdict`): cornering is
   comms-coupled **only if occlusion forces shared-target-belief *fusion* over comms** — with
   line-of-sight, mutual observation substitutes for comms and the coupling dies. Occlusion is
   currently **OFF** in runs (through-wall comms; TODO #15, memory `wall-rf-occlusion-comms`) —
   effectively a prerequisite build.

## Substrate & lit anchors

- Buildable on the existing red/blue substrate: `GroupedMission` + `RandomKofN` route per-group
  objectives (team-vs-team); evaders fit the sim's `mission.update` NPC seam. This mission doubles as
  the roadmap's **team-vs-team** *and* **second non-coverage** entry.
- **SUB-PLAY** (CCS 2024, arXiv:2402.03741, code released) — pursuit red/blue + partial-obs sweep
  harness, but the adversary is **external**, not a covert insider: adapt, don't adopt. Also
  forkable: PettingZoo SISL `Pursuit` (grid — matches zymera), MPE2 `simple_tag`, Hüttenrauch swarm
  code (Δ-disk local obs), VMAS scenarios. (Memory `mission-spec-catalog-marl`.)
- **Covert-peer cross-validation** ranked as the natural *defense inside* cornering
  (memory `obligate-collaboration-taxonomy`).

## Related

- `../TetherRelay/` — sibling mobile-target mission (protect a *friendly* mover's link vs contain a
  *hostile* mover): the natural B-coupling pair.
- Published one-liner: `site/index.html` — "Cornering & patrol — reward shapes that make every agent
  pivotal, so the labor has to divide."
- Memories: `obligate-collaboration-taxonomy` · `lit-coop-mission-selection-verdict` ·
  `mission-spec-catalog-marl` · `lit-coma-too-sparse-use-difference-rewards`.
