# Open Topics — living agenda (updated 2026-06-29 PM)

## ⏳ RUNNING (balthar pipeline: r9090 → bench → occ → warmbig, all jobs=1)
- **r9090** (7/12) — best arch @cover_r=0 × open/rooms/mixed/crowded × 3 seeds → cov-vs-optimal +
  real-conn. ETA ~5:45 PM. Live in `report/pages/best-arch.html` (DATA-DRIVEN table — never hardcode).
- **bench** (0/36, waits for r9090 via benchchain tmux) — reproducibility `role/base × soft/lag ×
  bump/flat` × open/rooms/crowded × 3 seeds. ~9 h → ~3 AM.
- **occ A/B** (waits for bench via occchain tmux) — base(SLAM) vs occ(+sense_free+boundary) × 4 terrains
  × 2 seeds @32²/10, 16 runs ~4 h → ~7 AM. THEN **warmbig** — 16²→32² warm-start, occ vs base,
  open+rooms × 2 seeds, ~3 h → ~10 AM.
- **Renders** (auto crons): 9 AM r9090+bench, 11 AM occ+warmbig. Data-driven tables, merge-not-overwrite.
- **Occupancy belief BUILT** (ZymeraLab 1932594 + ZymeraExp d89a291 on main): sense_free occupancy +
  occ_frontier (Yamauchi) + boundary channels; 235 lab tests green. local_frontier collapses under
  occupancy → occ_frontier is the real explore signal.

## ✅ SETTLED (don't relitigate)
- **Metric = cover_r=0 REAL coverage scored as % of OPTIMAL (oracle)** — not sensed, not a fixed 90%.
  Optimals @32²/10: open 72% · rooms 37% · mixed 31% · crowded 40%. 90% is *impossible* on obstacle maps;
  goal = approach optimal while holding real connectivity.
- **Hard connectivity mask = OUT** (kills the resilience study + emergence). Soft/learned + emergent relay only.
- **SLAM = ON always** — prerequisite for the occupancy/boundary mechanism (null *isolated* coverage lift,
  but foundational for real frontier exploration + mission-field containment).
- **Warm-start WORKS** — ≈26× faster convergence + slightly higher final; scale-transfer (32²→16²) confirmed.
  Use the small→large ladder.
- **Best recipe across all cuts:** role-split + bump-explore (`ebump`) + soft connectivity (`role_soft_bump`).
- **Null / off-limits levers:** neighbor-attention (multihead) · SLAM-isolated coverage · info-gain (`einfo`,
  reward-hacks) · hard-mask · the learned skill-selector (`sel`, collapses at scale).
- **Compass:** routes fine; the gap is BELIEF, not routing. Frontier-exploration optimizes *sensing*
  (≠ cover_r=0 *visiting* = a coverage-path-planning problem).
- **Position-blindness:** the LPAC backbone strips absolute coords (for scale-invariance) → agents can't
  orient / know the field extent. The boundary/occupancy/mission-field idea is the proposed grounding.

## ⚖️ DECIDE / DISCUSS
1. **Frontier sweep** — 6-point cov↔conn Pareto trace (soft mechanism, no mask), each point decomposed by
   per-agent contribution. Launch after the benchmark, or hold?
2. **Connectivity mechanism inconsistency** — same Lagrangian gives 33% conn (open) vs 84% (corridors).
   Fix it, or is the relay structure the answer?
3. **Ensemble + multi-head** — REFRAMED: not for *size* (scale-invariance already there + selector collapsed),
   but for **resilience** (ensemble disagreement = covert-adversary anomaly detection) + **multi-mission**
   (soft-blended heads / mission token). Discuss after the benchmark.

## 🔨 BUILD QUEUE (the levers)
- **Contribution metric (#70) — the keystone.** Both the resilience measurement AND the relay incentive
  (difference-rewards). Local build.
- **Relay structure** — push the frontier *out* (difference-rewards + delivered-coverage → relays
  self-organize, self-balancing via marginal contribution). Parked for design.
- **Occupancy-grid / mission-field belief** — give the learned policy free/occupied/unknown + the boundary,
  for real frontier exploration + in-field containment. (SLAM prerequisite now in place.)
- **Attention-critic (MAAC)** — as the COMA counterfactual engine for #70.
- **Whole-team teleop warm-start (#69)** — the distillation version that survives (drive the team → BC → warm-start).
- **Position-grounding** — normalized boundary-relative position; does re-grounding help directed exploration?

## 🧭 THESIS-LEVEL
- **When to pivot to the adversarial/resilience layer** — the actual thesis (covert misbehavior → mission
  failure). Contribution metric + degradation curve → minimum m-of-n is the bridge.
- **Mission #2 / mission token** — you have missions in mind. Which (chase/patrol/formation)? Unlocks the
  generalist + token.
- **Non-MARL baseline panel (#56)** — scripted competitors for the scorecard.
