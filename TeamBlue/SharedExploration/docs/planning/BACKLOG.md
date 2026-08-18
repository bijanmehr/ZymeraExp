# BACKLOG — what we have, what's running, what's left

*Consolidated 2026-08-18 from `IDEAS.md`, `TODO.md`, `NEARTERM_EXPERIMENTS.md`, `T5_DECISIONS.md`,
`T5_STATUS.md`, `ctde_v0/EXPERIMENTS.md` (the dial ledger), the L3/zymera2 design specs, and memory.
Living doc — update status as things land.*

The project runs on **two prongs**: **Prong 2** = the blue-team substrate (mostly built) · **Prong 1** =
the covert-adversary / resilience thesis (the actual research goal, almost entirely ahead of us).

**Governing rule (do not violate):** *topology must emerge* — we supply objectives + capacity +
information, never the authored solution. Route-station proposers, authored chains, and hard connectivity
masks are rejected; failure to emerge is a finding, not a licence to author.

**Metric of record:** coverage = **1×1 stood-on** (the operated cell); 3×3 is sensing range only.

---

## ✅ HAVE — done / settled

**Substrate & architecture**
- Coverage mission (Sense×Organize) @ 32²/10, corner spawn; occlusion model `d_eff=d+c·k` spec'd (c=comm_r/3).
- 3-clock **L3 (goal) / L2 (planner) / L1 (tracker)** decomposition — settled design.
- LPAC-KB backbone (CNN→GNN→64-d KB) + GCRN size-invariant belief (zero-shot 16²→32²).
- **Goal-region head** (L3) — the action-representation fix (replaced 1-step move head).
- Count-invariant critic (setpool ≈ setattn > conv); MAPPO-CTDE trainer + V and λ̂₂ critic heads.

**Planner (L2)**
- MVProp distilled navigator — validated 4 ways (w-recovery 98–99.6%, field↔oracle corr 0.96–0.999,
  goal-invariant, optimal-by-construction). Arm matrix: mvprop ≈ gppn ≈ highway ≫ msp.
- Distillation-recipe ablation done (best = mixed + K128 + γ0.99; deployed K32/γ0.9 is suboptimal).

**Credit & connectivity**
- Exact **difference reward** (submodular Dᵢ) = the division-of-labor fix (≈ doubles coverage).
- Connectivity via **reward shaping only** (reach + degree), soft, no hard mask. Reach-weight frontier
  mapped — reach-3 knee ≈ 38.5% stood-on / 92% one-piece (giant 98–99%).

**Mission #2 & red**
- `tether_v0` BUILT (98 tests) — mission-generality started (invariance only at matched net dials).
- Red arsenal built + tested: ① map poison, ② coverage-hole action sabotage, ③ embedding poison; RQ4
  contribution detector; stealth–damage frontier sweep. *(open bug: pose-spoof diverges cross-platform on gaius.)*
- Infra: balthar (RTX PRO 6000) + gaius (2×A6000, 2026-08-18); overnight tmux harness.

---

## 🔴 RUNNING
- `gscratch` — reach-2 & reach-3 with the guard stack (PID + degree-floor 2 + collision) **from scratch**,
  10k, 2 seeds. Tests whether guardrails-in-training beat the warm-start version (which was a null:
  ~38% / ~90%, 2/8 seeds collapsed).

---

## 🕗 LATER — the backlog

### Prong 1 — resilience thesis (the goal; barely started)
1. **Run the red study** — covert red → mission failure (machinery built, unrun; waits on locking the blue operating point).
2. **k\*** — minimum m-of-n compromise that degrades the mission past threshold (RQ3).
3. **Propagation model** — one-step influence `Infl^{i→j}` + multi-hop bound (RQ2).
4. **RQ4 detector** — KB-audit linear probe + ensemble disagreement. (EBM energy-detector DEMOTED — build/calibrate/freeze only; Bijan's conditional-baseline objection stands.)
5. **Stealth–damage Pareto + break budget** — per-edge KL forces the frontier; break budget = min-cost D forcing failure w.p. ≥ 1−δ.
6. **Position/role amplification** — H1 (comms amplify at cut-vertices/relays), H2 (adaptive ≫ random at equal budget).
7. **Resilience metrics** — R_rob, brittleness frontier/index, elasticity, recovery ratio/time. *Build measurement BEFORE the attack.*
8. **Potential-game null model** — nominal difference-reward team ≈ exact potential game; covert deviation = stationary-distribution shift.

### Prong 2 — open substrate levers
1. **L3 goal-head G×C×X ablation** — *highest-value open build.* goal-repr {stencil · spatial-softmax · top-K proposer+scorer} × cadence {per-step · commit-K} × coordination {none · claim-round · intention-visibility}. 18 cells, backbone/L2/L1/PPO frozen. Primary bets: C1 (commit-hold) + G1 (spatial-softmax).
2. **Cadence** — learned termination (option-critic) + deliberation/energy cost. Currently fixed commit_k=10 + reached/invalid interrupts; learned cadence built but OFF. Sweep K∈{5,10,25} and/or flip `--learned-cadence`.
3. **Coverage: representation vs credit** — 2×2 (sector8/kbattn × shared/difference), 3 seeds — is the 32² failure the 8-sector bottleneck (H_repr) or diffuse credit (H_signal)? Pre-registered interpretation table exists.
4. **kbattn explorer** — attention over whole KB vs fixed 8-sector reduction (init as residual on frontier prior).
5. **Learnable message content** — channel is 1 bool today; richer content is a P3 comms seam AND a red attack surface.
6. **Delivered-flow credit** for relays (lands on tether/intermittent; Shapley still zeros relays).
7. **HAPPO / heterogeneous updates** — sequential-update decomposition to *initiate* division-of-labor (shared-policy suboptimality; middle ground HyperMARL/Kaleidoscope).
8. **Occlusion-aware comms** — model decided, run pending (today occlusion OFF overstates connectivity).
9. **Distillation-recipe retrain** — retrain deployed planner with the better recipe; does *mission* coverage move (belief-bound, maybe small)?
10. **Finish planner-arm matrix** — mixed terrain missing 5 seed-cells (~6.5 h).
11. **Make the connectivity guardrail bind** — τ≥~2, cut reach reward. *(gscratch is a version of this — degree_target=2 made λ move.)*
12. **Per-agent criticality input** — route `∂λ₂/∂me` into the goal head (same λ̂₂ head, new target).

### Missions backlog
- **Tether-relay** study (built, unrun) · **intermittent-delivery** (out-disjoint journeys) · **cornering** (obligate-collaboration flagship, Act-dominant; own design session) · **fusion-poisoning** (C-seam attack) · a **second non-coverage** mission (chase/patrol/formation) · **time-budget/collaboration probe** (does a tight horizon open a coordination gap ample time hides?).

---

## ⛔ DEAD ENDS — do not re-run (settled-fail / rejected)
- **Per-step connectivity mechanisms at scale** — mask / soft-λ / Lagrangian / PID / degree / edge-margin. Huddle satisfies the signal, dual λ flat (0.30→0.30, viol ≈0.001). *The guardrail campaign re-confirmed this in the spread regime.*
- **Hard connectivity action-mask** — kills the resilience study + emergence.
- **Fiedler eigendecomp-oracle / adaptive-λ** — as-is.
- **COMA / transformer / attention critic** — too sparse for set-cover (COMA credit↔truth r≈0.13).
- **L4 4-level tower & disperse→gather phase layer** — no emergence; cap at 2 learned levels.
- **Discrete role head** — flat ≥ role at honest spec (relay = "hold", emerges label-free).
- **MSP / VIN / diffusion / decision-transformer planners** — below floor on structured maps / blow the per-step budget.
- **Occupancy-belief extra channels** — net noise.
- **Zone divide-and-conquer · competitive relay/explorer · tiny-LM** — rejected.
- **Energy token / battery** — skip for 90/90 (may revisit under an energy-native mission; seeds cadence arm-b).

---

**One-line read:** the blue substrate is ~80% done; the one big open piece is the **L3 goal-head
(G×C×X, incl. cadence)**; the entire **red/resilience thesis is still ahead** — which is exactly what the
built-but-parked red arsenal is waiting on.
