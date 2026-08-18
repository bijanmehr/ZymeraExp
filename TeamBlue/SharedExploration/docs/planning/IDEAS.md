# Zymera — Ideas & Backlog (living)

The project's idea registry: what we've pitched, what happened to it, and what's still open. Consolidated
from the June coverage backlog **plus** the adversarial program, the missions, and the current build threads
that used to live scattered across other docs, memory, and the task list. Keep this current — it's the one
place to look before proposing something.

**Status tags:** ✅ tried → resolved · ◐ partly tried · 🟢 open / live · ❌ untried · ⛔ rejected — don't re-propose.
Full outcomes + numbers live in `../journal/JOURNEY.md` and the memory corpus; this is the map, not the record.

---

## 1 · Coverage substrate — the original 11 ideas & their fate

The June backlog. Nine of eleven were tried; the outcomes are settled.

| # | Idea | Status | Outcome |
|---|---|---|---|
| 3 | Frontier attention | ✅ | ~98% cov but disperses the swarm (conn 32%); **T1: collapses by 24²** — representational, not comms |
| 4 | CBF / Lagrangian / constrained connectivity | ⛔ | **Per-step guardrails huddle at scale** (§0 ledger in JOURNEY); T3b dual ≡ 0, null. Settled-closed. |
| 5 | Physics-inspired nets | ✅ | Adopted as the **LPAC backbone + MVProp value-iteration planner** (PINNs never pursued) |
| 6 | Network fundamentals (type/size/HP) | ✅ | **Count-invariant critic > conv; setattn ≈ setpool** (T4); backbone ablations done |
| 8 | Per-role toolkits / roles | ✅ | **Flat ≥ role** at the honest spec (T2); discrete role head dropped for coverage |
| 9 | Reward engineering (intrinsic/info-gain) | ✅ | **Difference rewards adopted** (exact submodular); info-gain (`einfo`) was a null/reward-hack |
| 10 | Graph-embedding KB | ✅ | **GCRN belief transfers zero-shot** (later unplugged); the "Fiedler estimator" was an eigendecomp oracle |
| 2 | Learnable comms (GNN) | ◐ | Neighbor-attention aggregation → **null lever**; learned message *content* still untried (channel = 1-bool gossip) |
| 7 | Multi-phase cycle (disperse→gather) | ◐ | L4 phase tower **dropped** (no emergence — coordination diagnostic); its delivered-coverage half → the **relay missions** |
| 1 | Energy token / battery | ❌ | **Deferred** — lit says skip for 90/90; "energy turnover isn't novel" |
| 11 | Diffusion models | ❌ | **Dropped** — per-step cost vs the 100-step budget; no size-invariant graph-diffusion survived |

*The deep-research verdict tables (Round 1/2/3, the 7-search method review) that used to fill this file are all
in memory (`lit-*`) and the §0 empirical block is in JOURNEY — not repeated here.*

---

## 2 · The adversarial / resilience program — the thesis (mostly OPEN)

The actual research question, barely started. Coverage was the substrate; **this is the point.**

- 🟢 **Covert internal misbehavior → mission failure.** The core claim: stealthy micro-deviation by internal
  agents propagates through the interaction layer and surfaces as macro (mission) failure, defeating robust
  aggregation (W-MSR). *(Memory: `project-rq-covert-misbehavior-resilience`.)*
- 🟢 **Minimum m-of-n (`k*`).** In a team of `n`, the smallest compromised set that degrades the mission past a
  threshold (RQ3) — tie `(2f+1)`-graph-robustness to tolerable faults.
- 🟢 **Propagation model (RQ2).** One-step influence `Infl^{i→j}` and the multi-hop bound (data-processing +
  TV triangle) — how deviation spreads; generalizes linear-consensus error propagation.
- 🟢 **Local anomaly detection (RQ4).** The defender's side. The **KB audit** (linear-probe of the shared
  belief) and **ensemble-disagreement** as a covert-anomaly detector; contribution-as-detector.
  *(`lit-potential-ebm-hetcredit-verdict`.)*
- ❌ **Energy-score detector (EBM) — revisit-if, red phase only (demoted 2026-08-10).** A one-class
  "surprise meter": train on nominal traffic only, flag per-edge departures; detection power ties to the
  ε_s KL budget (Neyman–Pearson); the lit corner is unclaimed (nearest = Gaussian/GP/trust statistics).
  **Bijan's objection stands:** the baseline is conditional and shifts on unseen maps, and the stealth
  bound caps ANY distributional detector by construction — honest role = measuring the detectability–damage
  frontier, not catching red. Red-free phase can only build/calibrate/freeze it (pre-registration hygiene)
  and measure the false-alarm side. Revisit when red exists and the frontier needs quantifying.
- 🟢 **Stealth–damage Pareto frontier + break budget.** The stealth constraint (per-edge KL) forces a
  frontier; the break budget = min-cost `D` that forces failure w.p. ≥ 1−δ.
- 🟢 **Potential-game null model (formalism import, 2026-08-09).** The nominal difference-reward team IS an
  exact potential game with Φ = team welfare (Marden school — *not* in Wolpert–Tumer), so log-linear-style
  play has Gibbs stationary `p ∝ exp(Φ/τ)` and Vetta's ½-PoA floors the nominal equilibria. Covert deviation
  becomes a *measurable stationary-distribution shift* under the ε_s KL budget — that insider corner is OPEN
  (the Marden-school adversary cluster is overt/external). Caveat: the guarantees assume discrete repeated
  play + revision protocols; the bridge to PPO-trained policies is itself a gap, not a free import.
  *(`lit-potential-ebm-hetcredit-verdict`.)*
- 🟢 **Position/role amplification (H1).** Comms amplify a small compromise at influential graph positions
  (cut-vertices, relays); adaptive ≫ random attacks at equal budget (H2).
- 🟢 **Resilience metrics.** Robustness `R_rob`, brittleness frontier + index, elasticity, recovery ratio/time
  — trajectory-based, threat-parameterized. Build the measurement before the attack.

---

## 3 · Missions backlog

Each coupling is a distinct attack surface; the roadmap pushes outward along the Sense/Organize/Act taxonomy.

- ✅ **Coverage (Sense × Organize)** — the built substrate: connectivity-aware cooperative coverage.
- 🟢 **Tether-relay (Organize-heavy).** A relay chain keeps a moving lead base-connected through a comms-denied
  maze; **k-redundant connectivity (Menger)** under a covert cut-vertex red. *(`project-tether-relay-mission`.)*
- 🟢 **Intermittent-delivery.** Relays carry payloads between moving endpoints over an *un-holdable* graph
  (store-carry-forward / time-varying-graph journeys); redundancy = out-disjoint journeys.
- 🟢 **Fusion-poisoning (the C-seam attack).** Bias the local→shared belief fusion — Boeing standards-based-
  teaming inspired; one bad input corrupts the fused picture. *(Task #21.)*
- 🟢 **Hazard-cordon / mobile-target cornering.** The **obligate-collaboration flagship** — reward-shape ×
  partial-obs makes every agent pivotal, so labor *must* divide (~10 pursuers vs 6 evaders in squads).
- ❌ **Second non-coverage mission (chase / patrol / formation)** as a generalization/falsification test.

---

## 4 · Open build & method threads

- 🟢 **L3 goal head — the real coverage lever, barely tuned.** T5's L2 planner is *solved and proven*; the
  remaining walled-map headroom is **L3 relay coordination**, not planning (the connectivity tax). This is
  the highest-value open build. **Lit verdict (2026-08-09, 4 scouts): direction CONFIRMED, three components
  indicted** — the 9-stencil is SAM's losing "steering commands" arm; per-step goal resampling is supported
  nowhere (commit K≈10–25 + event interrupt); no-coordination is refuted (claim-round ranked #1: ½-bound +
  deterministic symmetry-breaker + Grimsman comm-graph degradation = adversary handle; spatial intention
  channel #2 = the covert attack surface). Mission-general form = pluggable candidate PROPOSER (as data) +
  shared scorer; backbone/L2/L1 stay mission-blind. Connectivity-aware goal selection remains unclaimed in
  every family. **→ the G×C×X ablation** (goal repr {stencil · ego spatial-softmax · top-K proposer+scorer}
  × cadence {per-step · commit-K} × coordination {none · claim-round · intention channel}) — the head-to-head
  the literature lacks. *(`lit-l3-goal-representation-verdict`.)*
  **Criteria for the L3 brainstorming session (fixed 2026-08-10):** (1) **SLM-portable contract** — L3 =
  "K proposed candidates + teammates' claims → choose & commit", so a future language-model head is a swap
  (claim-round = turn-taking protocol; intention channel = message); numeric L2/L1 stay numeric. (2)
  **Emergent roles + label-free credit** — no role head; delivered-flow marginal credit pays by function,
  and the per-agent credit distribution doubles as the role-emergence order parameter AND the load-bearing
  map for later red placement (H1). (3) **Mission-generality demos = SharedExploration · TetherRelay ·
  Cornering** — reclassified 2026-08-10: cornering is **Act-dominant** (deliverable = intervention on
  external agents via divided work under shared constraints, on an Organize substrate), so the three demos
  **span all three taxonomy families** (Sense · Organize · Act); mission = {proposer, candidate features,
  reward terms} as data. (4) Evidence base = the G×C×X matrix above.
- 🟢 **Distillation recipe upgrade.** Deployed K32/γ0.9/rects is suboptimal; **mixed-maps + K128 + γ0.99** wins
  (maze 10→29, rooms 70→100). End-to-end retrain with the better recipe. *(`mvprop-distillation-ablation`.)*
- 🟢 **Connectivity-weight sweep / Pareto trace.** ~6-point cov↔conn frontier (soft, no mask), each point
  decomposed by per-agent contribution.
- 🟢 **Occlusion-aware comms (env realism).** Walls attenuate comm range (soft `d_eff = d + c·k`); today
  occlusion is OFF (through-wall comms), which overstates connectivity. *(Task #15; `wall-rf-occlusion-comms`.)*
- 🟢 **Connectivity signal: critic-only vs a distilled λ₂ estimator.** Does the actor need an explicit λ̂₂, or
  does the CTDE critic + loss suffice? *(Task #5.)*
- 🟢 **Heterogeneous-agent updates (HAPPO/HARL) — first axis of the MAPPO session (task #7).** Sequential-
  update decomposition (Kuba arXiv:2109.11251, Lemma 1) is the proof-backed fix for "symmetric credit cannot
  INITIATE division" — per-agent baselines differ by construction, and Prop. 1 shows shared policies are
  *exponentially* suboptimal at division-of-labor. Tension: it abandons parameter sharing (vs our
  count-invariance + T2/T4) → shared-yet-diverse middle ground: HyperMARL 2412.04233, Kaleidoscope
  2410.08540. Note: Shapley does NOT fix initiation (symmetry axiom ⇒ identical shares from symmetric states).
- 🟢 **Per-agent introspection — distributions + per-agent traces as tracked figures (Bijan, 2026-08-10).**
  See inside the team, not just team-mean curves: log per agent per step {position, goal choice k, commit
  age, contribution/credit dᵢ, delivered-flow share, soft-degree, component id}; per run emit distribution
  figures (credit bimodality = the role-emergence order parameter), per-agent traces, and claim/conflict
  stats into the run dir + `make_report` panels. Prerequisite instrumentation for P1/P2 and the G×C×X
  ablation — build FIRST so every subsequent run is introspectable.
- 🟢 **Delivered-flow credit for relays (2026-08-09).** Relay under-credit is objective MIS-SPECIFICATION,
  not allocation — Shapley on raw coverage still zeros relays. Credit by *delivered* flow (COIN
  packet-routing precedent; DSAC dual-price-as-learned-reward; danmox SOCP duals) — unclaimed in learned
  connectivity-constrained swarms. Lands on TetherRelay / PersistantNetwork.
  *(`lit-potential-ebm-hetcredit-verdict`.)*
- 🟢 **L3 intention-sharing / rich collaboration** — deferred until an *obligate* mission makes it pay
  (it's inert when agents can solo). *(Task #13.)*
- 🟢 **Reproducibility & transfer prong** — the 2nd contribution (Prong 2): measured reproducibility +
  level-resolved covert-amplification, size/count generalization. *(`project-two-prong-contribution-framing`.)*
- 🟢 **Learnable message content** — the one untried half of idea #2 (the channel currently carries 1 bool).
  A P3 comms seam + a richer adversarial attack surface.

---

## 5 · Rejected / settled — do NOT re-propose

Kept so we don't re-pitch dead directions. Reasons in memory + JOURNEY.

- ⛔ **Per-step connectivity mechanisms at scale** (mask / soft-λ₂ / degree / Lagrangian / PID) — the huddle
  satisfies the signal, dual stays flat. *(§0.)*
- ⛔ **Hard connectivity action-mask** — kills the resilience study + emergence; soft/learned + emergent relay only. *(pref.)*
- ⛔ **L4 strategy→role→skill→action tower** — hierarchy's benefit is exploration, not the structure; cap at 2 learned levels.
- ⛔ **Transformer / attention critic / COMA** — too sparse for set-cover (COMA credit↔truth r≈0.13) → use exact difference rewards.
- ⛔ **Occupancy / boundary belief** — clean negative, net noise at 2000 iters.
- ⛔ **Zone divide-and-conquer · competitive relay/explorer · tiny-LM** — *(`rejected-design-directions`.)*
- ⛔ **Fiedler eigendecomp-oracle as-is · adaptive-λ as-is** — don't retry unchanged.
- Deferred/dropped (may revisit under the right mission): **energy tokens · diffusion**.

---

*Housekeeping: this file replaces the old June-only "Ideas to Verify" backlog (deep-research verdict dumps
removed → memory). Pairs with `../journal/JOURNEY.md` (the narrative) and `T5_STATUS.md` (the current build).*
