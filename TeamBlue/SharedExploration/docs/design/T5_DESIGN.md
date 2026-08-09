# T5 — Planner-based multi-robot exploration (isolated)

## Why
Zero-shot on unseen SAR maps: the ctde_v0 reactive policy stalls (~20%, "stuck not slow",
sublinear with 2x budget). The learned 1-step move can't route around unseen walls.
Fix = frontier-based coordinated exploration + a discovered-map planner (the multi-robot
SLAM/exploration standard), with the KB doing coordination, not navigation.

## Architecture (3 clocks)
- L1 (fast, every step): belief update (sense+gossip) + PLANNER re-route to current goal
  over the known map (BFS/wavefront/D*-Lite; optimistic on unknown). Take first step.
- L3 (slow, every K steps): read KB -> select a connectivity-safe frontier; coordinate so
  agents take DIFFERENT frontiers (cost-utility / auction / Voronoi).
- KB (L2 substrate): fused belief = own sensing + gossiped neighbor discoveries + neighbor
  positions. Updated fast; read for the deliberate selection.
- Connectivity: a CONSTRAINT on selection (don't pick a frontier that fragments the graph),
  shaped in training by the Lagrangian/PID guardrail PAIRED WITH a reduced reach reward so
  it actually binds. Roles (explorer/relay) DROPPED — relay behavior emerges from "hold".

## Stages
- Stage 0 (NOW, no training, CPU): classical frontier explorer (this file set). Yardstick.
- Stage 1 (later, training): learned LPAC-KB -> frontier assignment; classical planner kept;
  count-invariant (setpool) critic.

## Isolation
Own dir. `from ctde_v0 import env_utils` (READ-ONLY, for the env only). No edits to ctde_v0.
