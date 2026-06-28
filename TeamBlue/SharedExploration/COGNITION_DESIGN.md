# Two-Level Cognition — Design (converged 2026-06-28)

The agent architecture for connectivity-constrained coverage, converged through the
2026-06-27/28 design dialogue. This is the **target design**; experiment design follows.
Board: `OPEN_THREADS.md`. Honest campaign record: `CAMPAIGN_REVIEW.md`. Method verdicts +
citations: `STRATEGY.md`.

## Core principle — capabilities (rich) vs. the selector's menu (small)

The failure mode to avoid is a flat pile of ten "skills." The fix is to separate:

- **Capabilities** — what an agent *can do*, always on. Rich; harvest freely from
  multi-robot systems. NOT chosen by the selector.
- **The menu** — the small set of *behaviors* the selector toggles between. Disciplined
  (~3), mission-specific.

An agent can carry ten capabilities; the selector still picks among only ~three behaviors.
SLAM, λ₂/Fiedler estimation, A\*, attention are **capabilities** (or computational
mechanisms), NOT menu items — which is exactly why they don't blow up the menu.

## The layers — where every MRS capability files

| Layer | Always on? | Selected? | Examples (harvest from MRS) |
|---|---|---|---|
| Perception (substrate) | yes | no | SLAM/mapping, **λ₂/Fiedler estimation** (our aux head), relative localization, belief fusion (GNN message-passing) |
| Execution (primitive) | yes | no | **A\*** / RRT, collision avoidance, potential-field nav |
| Computation (inside nets) | yes | no | attention, GNN aggregation |
| **The menu (behaviors)** | no | **yes** | **disperse · flock · hold** (coverage); search/pursue (other missions) |
| Coordination | — | — | market/auction → our **learned congestion price**; consensus, hysteresis |

## The two learned levels

- **Level 2 — the selector ("what to do").** Off each agent's belief (post-GNN graph
  state), a small head picks an abstract **mode** — *unnamed slots* (0/1/2/3), meaning
  EMERGES. This is the "general role-picker, let it evolve" — and our confirmed novel gap:
  belief-graph-driven role *emergence* for connectivity swarms (everyone else hardcodes
  relay/frontier).
- **Level 1 — the executor ("how to do it").** The chosen behavior sets a goal off the
  belief; A\* drives there as a valid move. Plumbing; the *mode* is what shifts behavior.

Pipeline: **SLAM → belief → selector picks behavior → behavior sets goal → A\* executes.**

## For the coverage mission (locked)

- **Substrate:** GNN belief + λ₂ aux head (the Fiedler capability — deep-dived in the
  `FidlerValueEstimation` study) + A\* executor.
- **Menu (3 behaviors):** **disperse** (coverage / frontier-seek) · **flock/gather**
  (connectivity / cohere) · **hold/relay** (a stationary bridge). This trio *is* the
  coverage↔connectivity↔structure trade-off, expressed as skills.
- **Skills are hybrid:** scripted where a primitive is obviously correct (A\*, hold),
  learned where behavior is subtle (disperse via frontier-attn, flock). One uniform
  `belief → goal` interface so the selector treats them identically.

## Individuation — in the selector, not the skills

Skills are shared code (everyone's "disperse" is the same disperse). What differs per agent
is *how often / when* it reaches for each skill — its **selector**. Each agent gets an
identity that biases its selector → agents specialize into different skill-mixes WITHOUT
duplicating the library. "Unique to some degree," and exactly what ES evolves: a population
of individuated selectors over one shared skill set.

## Anti-collapse — three complementary forces

Options/skill setups love to collapse (one mode wins, or the skills smear together). Defenses:

1. **Diversity pressure** — SND / role_div (mode–behavior MI), already shipped; keeps modes distinct.
2. **Individuation** — per-agent ES selectors; different agents prefer different modes.
3. **The free market (learned congestion price)** — each agent sees, *locally*, how crowded
   its chosen mode/region is among neighbors; crowding LOWERS that mode's value, so agents
   spread. Market behavior, but local + learned + decentralized (NO global bid broadcast).
   Seed: the anti-overlap reward. The *scripted* global auction stays a baseline + red-team
   surface (#2/#3) — it smuggles global comms and hardcodes the allocation we want to emerge.

## Training — ES + gradient, side by side (MERL / feudal-evolutionary)

A **level split**: **ES evolves the small selectors** (sparse team-level credit, escaping
the huddle local optimum, individuation) · **gradient/CTDE trains the large skills +
perception** (dense per-step signal). Different modules → nothing to interchange; they
*compose*, not compete (this dissolves the earlier ES-vs-CTDE warm-start worry).

## Emergence hypothesis — test, don't prescribe

The **gather/disperse rhythm** (disperse early, contract late) should FALL OUT as the
selector switching modes over time (the belief carries coverage-saturation + distance-to-
team). If it emerges, that's far stronger than hardcoding it. The running **B-fork / B-dico**
arms are the *static-diversity precursor*: do separated / individuated agents differentiate
at all? Their answer gates the selector work.

## Locked experimental setup (2026-06-28)

The agent spec is **tweakable once, then frozen across every experiment** (the comm_r lesson).

- **Weak = can't solo.** A single agent cannot complete the mission alone — enforced *structurally*
  (limited vision + range + the 100-step budget), not by a capability cap. Prove it with a one-off
  single-agent run that fails the goal.
- **Vision: 3×3 (sense_r = 1).** Kept small on purpose — bigger vision only makes coverage *faster*
  (a discovery-rate dial), trivializing the 16² rung; and the skills read the accumulated **KB** (the
  neighbor-fused map), not the raw sensor, so a bigger patch buys them nothing.
- **Memory = the KB.** The accumulated, fused map belief is the agent's memory; no extra per-agent
  recurrence by default.
- **Comms: comm_r = 5** every rung. **Horizon: 100 steps** every experiment.
- **Start:** random world location, clustered → a fully-connected (clique) graph at t=0. Obstacle-free for now.
- **Connectivity success = team in one piece** (giant component = N, on the real comm graph); λ₂ reported
  as a margin, not graded.
- **The "numbers" test = both:** the density-pinned ladder (16→24→32, N grows with the world) AND a
  fixed-world N-sweep (more agents, same world, same 100 steps — does adding agents *divide labor* or *flood*).

## Status

Design converged **2026-06-28**. **Experiment design is next.** Parked (per the user): the
positioning **literature review** (#2 — "where this sits between everything else") and the
**resilience / red-team** program (#3) — the scripted skills + scripted auction become their
baselines + attack surfaces.
