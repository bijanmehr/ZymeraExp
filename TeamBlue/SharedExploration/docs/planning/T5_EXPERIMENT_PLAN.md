# T5 — Experiment Plan

Companion to `T5_DECISIONS.md`. Ordered to answer one question at a time, change one variable per arm, and gate early on the make-or-break result.

## Guiding questions

- **Q1 — Generalization.** Does a fully-learned planner-based architecture close the zero-shot gap (reactive ctde_v0 ≈ 25%) under *free-space* comms, toward the classical yardstick (79%)?
- **Q2 — Planner.** Which learned planner (MVProp / GPPN / MSP) generalizes best, and does distillation-from-oracle beat RL-only?
- **Q3 — Occlusion.** As comms become obstacle-sensitive (sweep `c`), can the team hold coverage *under* connectivity, and do relay-at-doorway behaviors emerge? How does it degrade?
- **Q4 — Connectivity input.** Does routing the actor-side λ̂₂ (or per-agent criticality) into the goal head help — and only when connectivity is *hard* (high `c`)?
- **Q5 — Secondary.** Do critic arch (setpool vs setattn) and learned cadence matter?

## Fixed protocol (every experiment)

- **16 rollouts, never reduced.** **≥4 seeds** per arm.
- **No warm-start (for now).** From-scratch everywhere — the cleanest generalization claim (no pretrained-backbone confound). Trade-off: from-scratch × many arms × 32²/10 is a heavy serial queue, so prioritize the critical path (0→1→2→3) and defer secondary arms.
- **Clean train/test separation (provable):**
  - **Train** = procedurally-generated random maps, **fresh per episode** (infinite draws → no map memorization), from **train-generators only**: rooms + `ConnectedClutter` + `Pillars` (walls mandatory; rooms-only overfits to room topology).
  - **Test** = a **fixed** held-out set from **different generators** (SAR 30-map set: floor-plan / maze / cave / warehouse / collapsed / urban), on a **disjoint seed range**, verified non-overlapping (hash check). Zero-shot.
  - Optional 2nd test tier: unseen *seeds* of the train-generators — separates instance-generalization from distribution-transfer.
- **Primary metric:** zero-shot % coverage on unseen SAR. **Secondary:** connected-fraction, mean λ₂.
- **One variable per arm.** Shared eval harness; report as units land (HTML + GIF gallery).

## Phases

### Phase 0 — Build the T5 substrate (blocks everything)
Isolated `t5/`: backbone (warm-start) → **L3 goal-region head** → **L2 MVProp+Highway planner** → **L1 reactive tracker**. Occlusion `d_eff = d + c·k` behind a flag (`c`, default `comm_r/3`). Connectivity-input plumbing (actor λ̂₂ → goal head) behind a flag. Heuristic replan trigger (event + floor).
**Exit:** a training run completes and zero-shot eval runs end-to-end with sane rollouts.

### Phase 1 — Baseline / GO-NO-GO (Q1)
**1a — minimal plug-in first (de-risk the planner).** Before building the full 3-clock stack, take the *current working* ctde_v0 policy and **plug MVProp into its existing pipeline** (the `Actor` already has a `goal_logits` head — route KB+goal through the planner to produce moves), changing that *one* thing. Zero-shot eval. This isolates "does a learned planner help" in a known-good system: if it stalls here, it's the planner, not a new architecture. Build the full stack only if the plug-in shows life.
**1b — full reference config.** MVProp planner, goal head, **free-space (`c=0`)**, connectivity = **critic-only**, **setpool** critic, heuristic cadence, from-scratch. Zero-shot SAR eval.
Compare against: reactive ctde_v0 (~25%) and the classical yardstick (78.9%).
**Success:** coverage ≫ 25%, approaching the yardstick; connectivity maintained.
**GATE:** if the learned planner doesn't clearly beat the reactive baseline (in 1a already), stop and diagnose — the T5 premise (the gap is *planning*) is on trial here.

### Phase 2 — Planner ablation (Q2) → resolves ledger open #1, #2
Arms (one change each): planner ∈ {MVProp+Highway, GPPN+Highway, MSP} × teacher ∈ {distill-from-oracle, RL-only}. **Control:** no-planner reactive L1 (the old head).
**Metric:** zero-shot coverage. **Output:** the winning planner + teacher, carried into Phase 3.

### Phase 3 — Occlusion + connectivity study (Q3, Q4) — **the core contribution**
- **3a — occlusion.** Best planner at **`c = comm_r/3`** (the realistic setting) vs the Phase-1 `c=0` baseline → a **2-point free-vs-occluded degradation** signal. Coverage, connected-fraction, λ₂; watch for **relay-at-doorway emergence** (order parameter + GIFs). *Full `c ∈ {0, ⅓, ½, 1}` sweep = OPTIONAL follow-up for the degradation curve (the resilience result) — not needed to start.*
- **3b — connectivity-input ablation** at each `c`: {critic-only, distilled λ̂₂ input, per-agent criticality input}.
  **Hypothesis:** the input helps at high `c` (occlusion, connectivity hard) and is ~null at `c=0` — i.e. it earns its keep only when connectivity binds.
**Output:** the coverage-under-connectivity degradation curve + the connectivity-input verdict (ledger open #3).

### Phase 4 — Secondary ablations (Q5) → resolves ledger open #5, #6
- **4a — Critic:** setpool vs setattn (train-only, cheap). Prior: tie.
- **4b — Cadence:** heuristic vs learned-termination + energy vs learned-termination + deliberation-reward. Also report **replans saved / compute**, not just coverage.
- **4c — Time-budget / collaboration probe (Bijan).** Sweep the horizon (steps); measure the **flat vs coordinated (rich-L3) gap** on coverage. Hypothesis: tight time opens a coordination gap that ample time hides (T2's flat≥role null was at 100 steps — maybe not tight enough). *Soft* obligation (reactive degrades under a deadline), not hard like cornering. If the gap opens → collaboration has a home on **coverage** without building cornering; if the null survives at short horizons → coverage is robustly reducible, cornering required.

### Phase 5 — Deferred (later)
Scale-ladder warm-start to bigger maps; MAPPO-detail tuning (more episodes for 32²); **cornering** mission (its own design session, task #8).

## Critical path

```
  0 (build)  →  1 (GATE: does planning generalize?)  →  2 (which planner)  →  3 (occlusion + connectivity)  [CORE]
                                    └→ 4 (critic / cadence) can run after 1, in parallel
                                                                                          5 (scale / 2nd mission) later
```

## The two-part T5 claim (what "success" means)
1. **Phase 1:** a fully-learned planner-based agent *closes the generalization gap* the reactive policy couldn't (25% → ≫).
2. **Phase 3:** and *holds coverage-under-connectivity* as comms become obstacle-realistic (`c` sweep), with relay emerging — the thing coverage-only classical planning could not do.
