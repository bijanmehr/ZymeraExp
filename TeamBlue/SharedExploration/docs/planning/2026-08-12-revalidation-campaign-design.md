# Re-validation campaign — design sheet (SIGNED 2026-08-13; launch on Bijan's go)

**Status: DESIGN LOCKED — answers ruled in the 2026-08-13 session.** Balthar offline tonight; launch
when it returns. Substrate: zymera2 (certified) + ctde_v1 full study stack (L1 fixed
follower · L2 distilled planner · L3 seam with G/C/X/roles dials · MP/GCRN modes).

**Env, fixed for the whole campaign** (the corner-trained G0 setting): CORNER spawn (tl), 32²/10 mixed maps
(rooms+clutter, doors clear), horizon 100, comm_r 5, sense_r 1, clean radio
(delay 0, dropout 0), occlusion c = comm_r/3, rollouts 16, fresh train-pool worlds
per iteration, disjoint eval pool (16 maps), 3000 iters/run unless noted.
Every run reports: stood-on cov, sensed cov, deg≥1, connected-ticks, giant share.

## Stage 0 — S-L2: pick the campaign planner (supervised only, no RL)
| Cell | K | γ | teacher maps |
|---|---|---|---|
| P1 (v0-replica) | 32 | 0.9 | rects/clutter |
| P2 (prior winner) | 128 | 0.99 | mixed-diverse |
| P3 (intermediate) | 64 | 0.99 | mixed-diverse |

Selection: planner-only nav success on 100 held-out mixed maps (BFS-reachable pairs;
no greedy bracket — BFS upper only). Gate: winner ≥ 0.95 nav success. Cost: minutes.

## Stage 1 — S-GNN + S-critic: backbone axes (2+2+3 cells, reference arm = goal9/commit10)
- MP: delayed-1 **vs** sametick-2 (declared: 1 tick = 2 comm cycles)
- REC: ff **vs** gcrn (does the graph-fused latent add on top of the explicit KB?
  judged at zero-shot 40²/10 transfer — the setting where GCRN won in v0)
- CRITIC: setpool **vs** conv **vs** attn (t4 re-validation on honest physics;
  all three implemented + global-info/perm-invariance tested 2026-08-13)
Seeds: 3. Carry the winners forward.

## Stage 2 — S-G: the head axis (the plateau-breaker bet)
| Arm | Head | Notes |
|---|---|---|
| G0 | move5 | DONE (47.1% stood-on, the measured baseline) |
| G1 | goal9 + L2 field | v0-lineage stencil |
| G2 | goal_coarse (5×5, stride 4) | wider reach |
| G3 | goal_cand (top-8 frontier) | KB-native candidates |
Cadence fixed at commit_k=10 for all goal arms (C studied next). Seeds: 3.

## Stage 3 — S-C: cadence on the G winner  [RULED]
per-step (k=1) vs commit-10 vs commit-25 vs **LEARNED** (option-critic termination
head, commit_k=25 cap, deliberation cost 0.01 — implemented + consistency-tested
2026-08-13). 4 cells, seeds 3.

## Stage 4 — S-X + roles: coordination toward near-100% connectivity
2×2: {X0 none, X2 intention-plane} × {flat, explorer/relay roles}, each under
ALL 6 conn objectives {w=0.3 budget · reach 2.0 · reach 5.0 (STRONG — aimed at the
near-100% target) · soft_lambda 1.0 · lagrangian · pid} — RULED "everything":
adaptive duals first-class (runaway risk accepted, behaviorally verified). 24 cells,
seeds 3.

**MISSION REQUIREMENT (re-affirmed 2026-08-14): near-100% connectivity.**
Operationalized: connected-ticks ≥ 90% AND mean giant share ≥ 95%. Stage-4 winner
selection is CONSTRAINED: max coverage SUBJECT TO the bar; full cov↔conn Pareto
reported. If NO arm clears the bar, the campaign output = the measured frontier +
the binding-constraint identification (radio vs arena geometry; counterfactual
comm_r curves) → a data-backed mission-spec decision (comm_r ↑ / N ↑ / regime-B
intermittent definition) — Bijan's call, made with numbers. THE stage that answers the near-100% requirement with relay
capability actually present. (Runaway dual stays retired; fixed-λ = precautionary.)

## Stage 5 — confirmation + transfer  [RULED]
Winner at **10 seeds** (rliable CIs) + zero-shot suite: {open, rooms, clutter40} ×
{24²/6, 32²/10, 40²/10-at-caps-48} on frozen checkpoints.

## Bill
RULED iters: **3000/run + stage winners extended to 8000** before carry-forward.
~7 (S1) + 3 (S2) + 4 (S3) + 24 (S4) = 38 cells × 3 seeds = 114 runs × ~4.5 min
≈ 8.6 GPU-h + winner extensions (~0.7h) + Stage 0 + 10-seed confirmation +
zero-shot ≈ **~10.5 GPU-h total** (an evening on balthar).

## Sign-off record (2026-08-13 session)
1. S1 reference arm: **goal9/commit10** (recommendation accepted by default).
2. Stage-4 objectives: **EVERYTHING** — all 5 mechanisms × the 2×2, first-class.
3. Cadence grid: **{1, 10, 25, LEARNED}** (learnable termination — Bijan's addition).
4. Iters: **3000 + extend stage winners to 8000**.
5. Confirmation seeds: **10**.
6. Spawn: **corner** (ruled earlier; scatter = hard-variant column).
7. Horizon: **100 hard** (train + inference).
6. ~~spawn mode~~ **RULED (2026-08-13): corner spawn** — agents initialize bunched
   at one side of the world (the v0 test convention; tl default). Implemented as
   the default everywhere; scatter/cluster remain dials for robustness studies.
   Consequence: ALL prior v2 runs (anchor, w-sweep, parity) were scatter-spawned
   → the campaign re-establishes every baseline under corner spawn; scatter
   numbers are kept as the harder-variant column, not discarded.
