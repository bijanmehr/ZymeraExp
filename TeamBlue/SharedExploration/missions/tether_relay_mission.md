# Tether-Relay Mission — design notes

**Status:** designed, not built. The **second mission** beside shared exploration — the non-coverage demo.
**Date:** 2026-08-05.

## One line
A decentralized relay team keeps a **moving lead** connected back to **base** through a
**comms-denied looped maze**. The science is maintaining **`k`-redundant connectivity** (`k`
node-disjoint base→lead paths) under a **covert cut-vertex adversary**. Realistic anchor: DARPA
**SubT** / NASA **CADRE** forward-comms tether (the ACHORD "SNR-lie" red is a documented instance).

## Why this mission (realistic, and it *demands* the complexity)
Not invented — it is the forward-comms problem in comms-denied deployments (underground, planetary,
disaster interiors). Every complication is **forced**, not chosen:
- **Decentralized is forced.** The mission is to *build* the base↔lead link, so no central
  controller can position the relays — it would need the very link that does not exist yet. The
  center is unreachable *by definition of the problem.* (This is the strongest possible "why
  decentralized is a must.")
- **Multi-agent is forced.** base–lead distance > `comm_r` and the lead moves; one body cannot span it.
- **Reactive is forced.** Unknown maze + unpredictable lead → nothing can be precomputed; relays
  reconfigure to the geometry as it is discovered.
- **Covert adversary is natural and catastrophic.** The tether is a *sequence of trust hops*; each
  relay can only trust its neighbours' link reports. One liar on a cut vertex silently cuts the whole
  tether while every local check passes — covert-micro → macro-failure in its most literal form.

## The formalism (this is the science, not the mechanics)
- **Health = base↔lead node-connectivity `k`** — the minimum number of relays whose removal
  disconnects base from lead = the number of node-disjoint base→lead paths (**Menger's theorem**).
  Use this **s–t min-cut**, *not* global λ₂ (λ₂ is a proxy that hides *where* the tether is fragile).
- **`k` IS RQ3.** The minimum compromise size `k*` that fails the mission = the tether's
  connectivity. A chain → `k*=1`; a `k`-redundant tether → `k*=k`. **The network structure is the
  break budget.**
- Mission healthy while `k ≥ target`; degrading as `k → 1`; failed at `k = 0` (cut).
- **Redundancy is the defense.** `k ≥ 2` ⇒ no single covert node is catastrophic (the other path
  carries traffic while you localize the liar). Red's optimal play = occupy a **cut vertex** (a relay
  on *every* path), which collapses `k` to 0 regardless of how many relays exist elsewhere. Blue's
  tradeoff: spend relays on **redundancy** (survive compromise) vs **reach** (follow the lead deeper)
  = the **stealth–damage / break-budget frontier**, made concrete and measurable.

## Six-slot scope
- **Gate (obligate decentralized):** one agent can't span; **no center is reachable** (the killer
  necessity); relays pivotal (a cut vertex is total).
- **Coupling:** **B — topology.** Maintain a relational structure (the connected subgraph) with a
  redundancy (`k`-disjoint-paths) objective.
- **Interaction contract:** relays exchange positions + link-health reports; the team **trusts** each
  "my links are healthy / I am holding position." The trusted payload is the attack surface.
- **Role contract:** each relay holds a segment so base↔lead stays `k`-connected; the lead advances
  to its goal (exogenous, not a teammate).
- **Success + health:** success = fraction of steps `k ≥ target` (100% at `k ≥ 1`; *resilience* =
  holding `k ≥ 2`). Health Φ = `k` over time (+ the width of the weakest cut).
- **Redundancy / m-of-n:** `k` node-disjoint paths; **m-of-n = k**.
- **Covert red + detector:** red = a compliant relay that reports healthy links / "holding position"
  but drifts or lies (the SNR-lie), ideally sitting on a cut vertex; a stealth budget bounds its
  per-report deviation. Detector = local cross-check (neighbour-measured position vs claimed) +
  end-to-end base↔lead heartbeat, with the hard part = **localizing which hop drops it.**

## Environment
- **Looped maze — HARD PRECONDITION.** The maze must have alternate routes / cycles so `k ≥ 2` is
  achievable. A single narrow corridor (a tree maze) *forces* `k = 1`: you cannot lay two node-disjoint
  paths down one hallway, so redundancy is impossible and the whole science is moot. Keep a **tree-maze
  variant** for the pure-fragility (`k=1`) study; use a **looped maze** to make redundancy a *choice*.
- Base (fixed cell), walls with **occlusion** (`d_eff = d + c·k`), `comm_r`.
- **Lead motion:** goal-directed **stochastic** — a random deep goal per episode + greedy/A* nav +
  per-step noise. **Oblivious** to the relays (exogenous environment dynamics, does NOT wait when the
  tether stretches — keeps tether-holding entirely the relays' job). Depth capped at **≲ N · comm_r**
  (feasibility; deeper is unwinnable). Near the limit ⇒ every relay pivotal. **NOT a random walk**
  (zero drift → never stretches the tether → degenerate).
  - phase variants: **train** = randomized + difficulty curriculum (slow/shallow/straight → fast/deep/turning);
    **demo** = a fixed scripted path (legible, repeatable); **stress/eval** = an adversarial lead that
    maximally strains the tether (eval only — training against it destabilizes with a 2nd adversary).

## Reuse vs new
- **Reuse (verified this session):** `local_edge_margin`, the Lagrangian dual, occlusion `d_eff`,
  `kb_adjacency`; the grid + sensing + comms substrate.
- **New:** (1) a moving endpoint (the lead) to tether to; (2) the **s–t min-cut / `k` metric**
  (replaces the global-λ₂ grade); (3) a **looped-maze generator**; (4) the **covert link-liar red +
  chain-localization detector**; (5) reward = base↔lead reachability / `k` instead of coverage.

## Open questions (decide at build time)
- Reward shape: reachability (`k ≥ 1`) vs redundancy (`k ≥ 2`) — a soft dual on `k`? PID on the min-cut?
- How relays *learn to choose* redundancy-vs-reach (reward weighting vs curriculum).
- Is the detector (heartbeat + localize) learnable end-to-end, or a separate classical module?
- Lead position: known-to-relays (broadcast) or only inferable through the network (like
  intermittent-delivery)? Realistic = through the network.

## Relationship to the other missions
- **Not coverage** (no ground to sweep — the sole objective is holding a chain to a mover).
- **Not cornering** (protect a *friendly* mover's link vs contain a *hostile* mover) — but shares the
  mobile-target substrate; the two make a natural **B-coupling pair** (tether = keep-linked-to-mover,
  cornering = contain-mover). See `../missions/` for cornering when written.
