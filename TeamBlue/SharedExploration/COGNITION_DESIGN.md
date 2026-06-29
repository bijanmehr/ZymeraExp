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

## Training — ES + gradient, side by side (MERL / feudal-evolutionary) — RAN, mechanism confirmed

A **level split**: **ES evolves the small selectors** (sparse team-level credit, escaping the huddle
local optimum, individuation) · **gradient/CTDE trains the large skills + perception** (dense per-step
signal). Different modules → they *compose*, not compete. **Confirmed empirically** (16²/4, 80 rounds,
`es.py` + `run_es.py`; numbers in `CAMPAIGN_REVIEW.md §8`): the two co-adapt *upward* without either
collapsing.

### Why it's possible — the mechanism (remember this)
The coexistence rests on **three load-bearing properties; all must hold:**

1. **Disjoint parameters.** ES perturbs ONLY `selector_head`; the gradient updates ONLY the executor
   (backbone + goal head + critic), `selector_head` frozen under the gradient pass. The two optimizers
   never write the same weights → zero parameter-level interference. (Shared weights would let ES noise
   corrupt the gradient's credit assignment and vice-versa.)
2. **One shared, aligned objective.** Both maximize the *same* team return J(π) — ES over selector
   perturbations, CTDE over executor params (policy-gradient surrogate + central critic). Shared, not
   adversarial (unlike a GAN) → improvements *add* rather than fight. This is the "same centralized-training
   principle": both see the team-level signal.
3. **Timescale separation (the feudal/MERL trick).** Executor trains FAST (K dense gradient steps per outer
   round); selector evolves SLOW (one ES generation per round, the population evaluated against the *current*
   executor). The slow outer loop treats the fast inner loop as part of its environment — bilevel
   **two-timescale stochastic approximation** (Borkar): slow ES outer, fast gradient inner.

**Why ES for the selector (not gradient):** (a) the selector output is a **discrete** skill choice
(m ∈ {disperse,flock,hold}) → discrete decisions give high-variance/biased policy gradients (REINFORCE/Gumbel
hacks); **ES is gradient-free**, indifferent to the non-differentiable argmax. (b) the selector is **small**
(`W→3` head) and ES scales poorly with dimension → fits a tiny head, not the whole net. (c) ES explores in
**parameter space** → naturally yields **individuated** selectors and escapes the **huddle local optimum**
(sparse team-level credit, where dense gradient collapsed — cf. the DTE result).

**Why gradient/CTDE for the executor:** perception + goal head are **high-dimensional** and need **dense
per-step credit** (which move gained coverage). PPO + a central critic is sample-efficient and low-variance
there; and the A/B batch proved the **central critic is load-bearing** (the DTE collapse).

**The risk (watch this):** selector and executor are **moving targets** for each other → non-stationarity →
the executor-coverage **volatility (35–59% in the late rounds)** IS that co-adaptation noise. It held at
16²/4 but is the failure mode to watch at scale.

### Canonical references (the mechanism's lineage)
- **ERL** — Khadka & Tumer, *Evolution-Guided Policy Gradient in RL*, NeurIPS 2018 (population EA + gradient,
  shared replay; EA explores, gradient exploits).
- **CEM-RL** — Pourchot & Sigaud, *CEM-RL: Combining Evolutionary and Gradient-Based Methods for Policy
  Search*, ICLR 2019 (CEM on the actor + TD3 gradient — closest to our CEM/ES + gradient split).
- **CERL** — Khadka et al., *Collaborative Evolutionary RL*, ICML 2019 (portfolio of learners + evolutionary
  outer loop).
- **OpenAI-ES** — Salimans et al., *Evolution Strategies as a Scalable Alternative to RL*, 2017 (the ES
  estimator we use).
- **FeUdal Networks** — Vezhnevets et al., ICML 2017 (slow manager sets goals / fast worker executes — the
  timescale-separation our selector/executor split mirrors; origin: Dayan & Hinton, *Feudal RL*, NeurIPS 1993).
- **Options** — Sutton, Precup & Singh, *Between MDPs and semi-MDPs*, Artificial Intelligence 1999
  (skills-as-temporal-abstractions — the menu's grounding).

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
