# T5 — status, what's done, what's left

*Last updated 2026-08-08. Living doc; the git cleanup + balthar-side reconciliation are deferred (noted at the end).*

## What T5 is

T5 is the **fully-learned, multi-level agent** for connectivity-aware cooperative coverage that must
**generalize to unseen maps**. It exists because of the T5 Stage-0 finding: on unseen maps the
generalization gap was **planning, not learning** — a classical frontier+planner hit ~79% zero-shot
while the learned 1-step reactive move head hit ~25%. T5 keeps the system fully learned but replaces the
reactive move head with a **distilled planner**, so navigation generalizes without hand-coded search.

Each agent runs three "clocks":

| level | job | what it is | status |
|---|---|---|---|
| **L3 — goal** | *where to go* | learned goal/region head off the belief | **open** — the real lever, barely tuned |
| **L2 — planner** | *how to get there* | **distilled MVProp** value-iteration net (routes around known walls) | **done + proven** |
| **L1 — controller** | *the actual move* | climb the field + hard collision veto (scripted) | done |

Supporting substrate (all built): **LPAC-KB backbone** (per-agent CNN → 2-round GNN over the comm graph),
**CTDE central critic** (discarded at deploy), **MAPPO** training, a **soft λ₂ connectivity guardrail**
(no hard mask, by preference), and a **gossip comms + SLAM belief** layer the whole thing reads.

---

## What's done

### L2 planner — done, and now *proven* (this session's focus)
The MVProp planner is validated at three levels of rigor:

1. **Benchmark (single-agent, 50 seeds/map-type, bounded).** %Success reaching goals on real MovingAI
   maps + independent generators, bracketed by BFS-teacher (upper=100) and naive Manhattan (lower).
   Conv planners (mvprop≈gppn≈highway) clear the Manhattan floor everywhere, reach 94–96 on open game
   maps; **MSP is at/below the naive floor on every structured map** → dropped.
   *(planner_study_report.html · ms_bench.py)*

2. **Multi-agent bracket.** Same L1 swapped inside the real 10-agent env with a fixed god-view L3,
   bracketed by greedy (lower) and god-view BFS (upper). Conv planners reach the upper bound on
   clutter/mixed; rooms leaves ~15pt headroom (corridor terrain). MSP below the greedy floor.
   *(planner_bracket_report.html · ma_planner_bracket.py)*

3. **Mechanistic proof (the real validation).** MVProp's forward *is* value iteration over one learned
   quantity, passability `w = σ(conv[blocked,goal])`; if `w = 1−wall` the recurrence gives the exact BFS
   wavefront. Measured: **w-recovery 98–99.6%** on every map type (incl. OOD), **w is goal-invariant**
   (|Δw|≈0.001), **field↔oracle correlation 0.96–0.999** (vs MSP 0.79–0.93). So MVProp is *optimal by
   construction* and generalizes because "passable iff not a wall" is map-independent physics.
   *(mvprop_proof_report.html · validate_mvprop.py)*

4. **Distillation recipe ablation.** K (depth) × γ (discount) × training-distribution. The deployed
   recipe (K32, γ0.9, rectangles) is **suboptimal on corridor/maze** terrain; a better recipe is more
   depth + diverse training + higher γ *together* (best found: mixed+K128+γ0.99 → maze 10→29, rooms
   70→100). γ→1 alone backfires. *(distill_ablation_report.html · ablate_distill.py)*

### Substrate — built and characterized
- **Full pipeline trained** on open/clutter/rooms/mixed @32²/10 (the `parm_32_*` runs + greedy baselines,
  on balthar). Coverage: open ~0.72, clutter ~0.56, rooms/mixed ~0.30. Connectivity_real (λ₂>0.5):
  clutter ~0.84, rooms ~0.97, mixed ~0.85 — **not** the ~1.0 that connectivity_pct misleadingly showed.
- **Comms subsystem fully read from source** (see `comms_report.html` / memory `comms-subsystem-facts`):
  Chebyshev-disk graph, gossip channel (lossless in these runs), one bool SLAM belief (sense_r=1 → wall
  knowledge is ~all gossip), 2-round max-aggregate GNN (size-invariant, trust-by-default), λ₂ Fiedler.
- **Connectivity tax quantified.** Low walled-map coverage is the *cost of staying connected*, not weak
  planning: a god-view controller with no connectivity constraint hits ~0.61 on rooms by shattering the
  team (λ₂→0.10). Corrected the coverage ceiling (the old greedy god-view baseline was broken on walls).

---

## What's left

**Ordered by leverage.**

1. **The real lever is L3 + belief, not L2.** Every angle (single-agent, multi-agent, isolated, in-team)
   says the planner is solved and *not* the mission bottleneck. In deployment agents plan on a sparse,
   comms-fused belief (~40% of walls known, near-blind early), so richer belief (comms/connectivity) and
   smarter goal assignment (L3, relay coordination) are where mission coverage will move. **This is the
   next real work.**
2. **End-to-end recipe check.** Retrain the deployed planner with the better distillation recipe and see
   if rooms/corridor *mission* coverage actually moves (belief-bound, so maybe small — but rooms is where
   both planner and mission are weakest). One balthar run.
3. **Connectivity-weight sweep.** `w_connectivity ∈ {0, 0.5, 1, 2}` on a walled terrain → quantify how
   much coverage the connectivity tax costs, and whether the policy even reaches the Pareto frontier.
4. **Verify the connectivity *mechanism* per run.** Config default is `action_mask` — a **hard mask** —
   which contradicts the no-hard-mask stance and would make connectivity unbreakable (killing the
   resilience study). Confirm which mechanism each `parm_32` run actually used.
5. **Finish the planner-arm matrix.** Mixed terrain is missing 5 seed-cells (balthar, ~6.5 h).
6. **Prong 2 — the adversary.** The covert-red thread on comms is barely started. The comms read surfaced
   clean, buildable attack surfaces: a **lying / asymmetric edge** (the graph is symmetric, trust-by-
   default, no anomaly gate) and a **poisoned gossip payload** (one shared belief, no provenance).

---

## Load-bearing verdicts (don't re-litigate)

- Distilled MVProp is the L2 planner. MSP (transformer) is out — no value-iteration structure, doesn't
  route. mvprop ≈ gppn ≈ highway; keep MVProp (simplest).
- Planner architecture is *not* the mission lever; belief/comms + L3 are.
- Connectivity must stay soft (no hard mask) — a team that can't fracture has nothing to study.
- Report %connectivity as the **real** bar (λ₂>0.5), never `connectivity_pct` (λ₂>0.001, ~always 1.0).

---

## Assets & pointers

- **Reports (local `report/`, also published as artifacts):** planner_study, planner_bracket,
  mvprop_proof, distill_ablation, planner_matrix, planner_demo, comms.
- **Analysis scripts — currently in the session scratchpad, NOT yet in the repo (rescue pending):**
  `planner_verify.py` (external benchmark + generators + MovingAI loader), `ms_bench.py`,
  `ma_planner_bracket.py`, `validate_mvprop.py` (the proof), `ablate_distill.py`, `comms_demo.py`,
  plus the MovingAI maps and `ms_bench.json` / `ablate_distill.json`.
- **Code:** MVProp `t5lab/mvprop.py`, distillation `t5lab/distill.py`, planner arms `t5lab/planner_arms.py`,
  render `t5lab/render_mvprop.py`; env/comms/connectivity in `ctde_v0/{env_utils,ppo,nets,config}.py`.
- **Runs (balthar):** `ZymeraExp-nearterm/.../runs/mvprop/parm_32_*` (arm matrix), `*32_greedy` /
  `*32_mvprop` (L1 comparison). Distilled planners in `planners/{arch}_distilled.eqx`.
- **Memory:** `planner-arm-matrix-result`, `mvprop-distillation-ablation`, `comms-subsystem-facts`,
  `connectivity-tax-and-corrected-ceiling`, `mvprop-planner-distillation-solution`.

## Deferred housekeeping (both GPU boxes are up + idle as of 2026-08-08)
- Rescue the scratchpad scripts into `SharedExploration/planner_study/` (time-sensitive — scratchpad is
  session-scoped).
- Commit the session's reports + rescued scripts on branch `t5-impl` (57 uncommitted currently; 0 unpushed).
- Clean `zymera_lab` stray files (Untitled.canvas/.base, 2026-07-08.md). Push `zymera_env` (2 unpushed).
- Reconcile the diverged `ZymeraExp-nearterm` on balthar; pull any `/tmp` render results before reboot.
- Prune the stale task list.
