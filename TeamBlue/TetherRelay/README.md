# TetherRelay — mission workspace

**Mission #2** beside `../SharedExploration/` (coverage). **Status: v0 build spec approved 2026-08-14 —
`zymera_lab/docs/specs/2026-08-14-tether-relay-v0-design.md`** (24² · 5 relays + NPC explorer ·
s–t Menger-k health · zymera2 + `experiments/tether_v0/`). Concept spec:
[`tether_relay_mission.md`](tether_relay_mission.md) (2026-08-05; moved here from
`SharedExploration/missions/` on 2026-08-09). This folder is the mission's home — notes and spec now,
runs/reports/checkpoints when the build starts. Implementation-level lit sweep (6 angles, 2026-08-13/14):
memories `lit-tether-implementation-sweep` + `lit-resilience-metrics-verdict`.

## One line

A decentralized relay team keeps a **moving lead** connected back to **base** through a comms-denied
**looped maze**; the science is **k-redundant connectivity** — `k` node-disjoint base→lead paths
(Menger) — under a **covert cut-vertex adversary**. In plain terms: hold a chain of radios through a
cave to a walker who keeps walking deeper, while one relay quietly lies about its links.

## Why it leads the mission portfolio

Per the 2026-08-06 lit map (memory `lit-coop-mission-selection-verdict`): **adopt, ranked #1** — the
only candidate that intrinsically passes both selection criteria: (A) decentralization is
**mandatory** — the mission *builds* the very link a dispatcher would need, so no center is reachable
by definition; and (B) the **topology is the interaction layer and the only attack surface**.
Whitespace: no released MARL benchmark exists for k-connectivity/redundancy or covert-on-connectivity
(memory `mission-spec-catalog-marl`).

## Load-bearing decisions already made (full set in the spec)

- Health = base↔lead **s–t node-connectivity `k`** (min-cut), **not** global λ₂ — λ₂ hides *where*
  the tether is fragile. **`k` IS RQ3**: a chain has `k*=1` (one covert node = silent total cut).
- **Looped maze is a hard precondition** — a tree maze forces `k=1` and kills the redundancy science.
  Keep a tree-maze variant as the pure-fragility (`k=1`) study.
- Lead = goal-directed **stochastic**, oblivious to relays, depth ≲ `N·comm_r`; **not** a random walk
  (zero drift never stretches the tether). Train = randomized + curriculum; demo = scripted path;
  stress = adversarial lead (eval only).
- Red = covert **link-liar on a cut vertex**. ⚠️ Corrected 2026-08-14: the "SNR-lie" is OUR threat
  model on ACHORD's documented **trust-by-default surface** (self-reported link metrics propagated
  unaudited) — no misreport *instance* is documented in the SubT literature; do not cite one.
  Detector = neighbour cross-check + end-to-end heartbeat; the hard part is **localizing the hop**.

## Build templates & substrate

- k-metric math: **Luo–Sycara k-CMCS** (IROS 2019) — edge critical iff ≥ `k+1` node-disjoint paths
  (max-flow test); no code. Nearest MARL: **Galliera Dynamic Network Bridging** (arXiv:2404.01557 —
  base paper; 2404.01551 is the safety follow-up) — single-path, no redundancy term, and its learned
  policy loses to a centralized line heuristic (63.9% vs 83.2% uptime); **extending to `k` is the open
  gap** (verified 2026-08-13; nearest learning+k work = Tokekar FCR, global-k one-shot restoration).
- Reuse from the built substrate: `local_edge_margin`, the Lagrangian dual, occlusion `d_eff`, the
  grid/sensing/comms sim. New to build: moving lead (the sim's `mission.update` NPC seam) · s–t
  min-cut metric · looped-maze worldgen · link-liar red + chain-localization detector ·
  reachability/`k` reward (replaces coverage).

## Open questions (decide at build time)

Reward shape `k≥1` vs `k≥2` (soft dual / PID on the min-cut) · how relays *learn to choose*
redundancy-vs-reach · detector learnable end-to-end vs a classical module · lead position broadcast
vs known only through the network (realistic = through the network, like intermittent-delivery).

## Related

- `../Cornering/` — the sibling mobile-target mission (contain a *hostile* mover vs protect a
  *friendly* mover's link): the natural B-coupling pair.
- `../PersistantNetwork/` — placeholder for the delivered-coverage objective (ctde_v0
  `EXPERIMENTS.md` dial 14′).
- Intermittent-delivery (memory `project-intermittent-delivery-mission`) — the time-varying-graph /
  store-carry-forward relay sibling: endpoints move and the graph *cannot* be held.
- Narrative: `../SharedExploration/docs/journal/JOURNEY.md` (closing section — the thesis builds here
  next, "connectivity as an obligate deliverable rather than a slack constraint").
- Memories: `project-tether-relay-mission` · `lit-coop-mission-selection-verdict` ·
  `mission-spec-catalog-marl` · `lit-energy-turnover-relay-bridge` · `lit-real-missions-sparse-organize`.
