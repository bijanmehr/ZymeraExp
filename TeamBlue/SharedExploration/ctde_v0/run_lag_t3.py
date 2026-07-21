"""T3 connectivity-mechanism test — does the soft dual hold a GLOBAL-λ₂ target @32²/10?

The question: how high can we push REAL, GLOBAL connectivity (team algebraic connectivity
λ₂ ≥ a target θ) WITHOUT collapsing coverage — and does the learned dual actually do the work?

Why the target is GLOBAL λ₂ and not per-agent degree (the correction):
  Connectivity is topological — the whole team in ONE connected piece. A spread-out spanning
  tree/chain is connected (λ₂ > 0) with most agents holding only 1-2 neighbours. Penalising low
  *degree* would fight the very spreading coverage needs (and degree ≥ 1 per agent does not even
  guarantee one component — the team can fragment into pairs). So the constraint is on the true
  global λ₂. It is computed by the TRAINER from all positions (centralised training) and enters
  only through the REWARD; the agents never observe it and act on local info at execution (CTDE).
  NO hard action mask ever (hard mask makes connectivity unbreakable → kills the resilience study).

The mechanism (thermostat): a per-step reward penalty ``-λ · relu(θ - λ₂)`` is subtracted; the
policy (primal) descends the PPO loss on that shaped reward, and the dual λ (dual ascent) rises
whenever the target is violated until connectivity reaches θ. We sweep the target θ so it BINDS
(a slack target never fires λ → tests nothing) and to trace the coverage↔connectivity frontier.

  MECHANISM (3, ladder of adaptiveness):
    soft_lambda    — λ is a FIXED constant (brittle baseline: one weight for all scales).
    lagrangian     — adaptive dual ascent λ←relu(λ+lr·v); auto-tunes but MONOTONE (only ratchets up).
    pid_lagrangian — PID dual (Stooke 2020); responsive — P/D let λ rise AND fall, tracks θ stably.
  TARGET θ (the λ₂ floor, swept to bind): 0.5 (real-connectivity bar) · 0.7 (demanding).

  ARMS (5):
    soft_th5  : soft_lambda           · θ=0.5              (fixed-weight baseline, penalty=1.0)
    lag_th5   : lagrangian            · θ=0.5 · lr=0.05
    lag_th7   : lagrangian            · θ=0.7 · lr=0.05
    pid_th5   : pid_lagrangian        · θ=0.5 (default PID gains)
    pid_th7   : pid_lagrangian        · θ=0.7 (default PID gains)
  Fixed: 32²/10 · comm_r 5 · collision-mask on · frontier-attn · role split (expl_relay) ·
    up-weighted coverage (--w-coverage 3) · cover_r 0 · conn-signal global_lambda2 · the FULL
    7-channel SLAM occupancy KB belief (--sense-walls --sense-free --boundary:
    free/occupied/unknown + occ_frontier + boundary; was 5-channel coverage-trail).

Per seed: 5 runs. Run-dir <out>/seed{S}/{arm}. FRESH (no warm-start) — this is the screen.

We log the two curves that ARE the test — the dual λ(t) (``dual_lambda_next``) and the realised
connectivity λ₂(t) (``dual_violation`` / connectivity_real; emitted by ppo.py ~1147-1151) land in
each run's history.json — so we can SEE whether λ actually rises to enforce θ and whether λ₂
converges to it.

GRADE: the coverage↔connectivity_real frontier (does forcing λ₂→θ collapse coverage?) + does the
dual track θ. Expectation: pid holds θ more cleanly than monotone lagrangian; both beat fixed
soft_lambda. If λ climbs but λ₂ never reaches θ, local info isn't enough → relay / λ₂-estimator.

    # see the whole plan first (no training):
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python \
        -m ctde_v0.run_lag_t3 --dry-run --seeds 3

    # the real batch on balthar (drop JAX_PLATFORMS so it uses the GPU), 3 at a time:
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        -m ctde_v0.run_lag_t3 --out runs/lag_t3 --seeds 3 --iters 2000 --rollouts 16 --jobs 3

===>>> This launcher LAUNCHES REAL TRAINING; run it deliberately. <<<===
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _schedule

# The honest fixed spec — IDENTICAL on every arm. frontier-attn explorer + hard collision-mask
# (movement only) + role split ON + up-weighted coverage + cover_r 0 @ 32²/10, and the GLOBAL λ₂
# signal (the true team connectivity, computed centrally, folded into reward). The belief is now
# the FULL 7-channel SLAM occupancy KB: --sense-walls + --sense-free + --boundary give
# free/occupied/unknown occupancy + a Yamauchi occ_frontier + the field-edge boundary (5→7 ch).
_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--w-coverage", "3", "--sense-walls", "--sense-free", "--boundary",
          "--horizon", "100", "--cover-r", "0",
          "--role-picker", "expl_relay", "--conn-signal", "global_lambda2"]

# --- the 5 arms (id -> the extra train_ctde flags that DEFINE that arm) ---
#   soft_lambda    · θ=0.5           (fixed-weight baseline; --soft-lambda-penalty defaults 1.0)
#   lagrangian     · θ∈{0.5,0.7}     (adaptive dual, lr=0.05)
#   pid_lagrangian · θ∈{0.5,0.7}     (responsive dual, default PID gains)
_ARMS: dict[str, list[str]] = {
    "soft_th5": ["--mechanism", "soft_lambda", "--constraint-threshold", "0.5"],
    "lag_th5":  ["--mechanism", "lagrangian", "--constraint-threshold", "0.5", "--lambda-lr", "0.05"],
    "lag_th7":  ["--mechanism", "lagrangian", "--constraint-threshold", "0.7", "--lambda-lr", "0.05"],
    "pid_th5":  ["--mechanism", "pid_lagrangian", "--constraint-threshold", "0.5"],
    "pid_th7":  ["--mechanism", "pid_lagrangian", "--constraint-threshold", "0.7"],
}
# fixed arm order (deterministic listing / launch order)
_ARM_IDS = ["soft_th5", "lag_th5", "lag_th7", "pid_th5", "pid_th7"]


class Unit:
    """One train_ctde run: id, run-dir, seed, the extra flags. FRESH — no warm-start, no deps."""
    def __init__(self, uid, run_dir, seed, extra):
        self.uid = uid
        self.run_dir = run_dir
        self.rung = (32, 10, 5)              # (grid, n_agents, comm_r)
        self.seed = seed
        self.extra = extra
        self.init_from_dir = None
        self.needs_dir = None
        self.proc = None

    @property
    def init_from(self):
        return None

    def ready(self):
        return True

    def cmd(self, iters, rollouts):
        g, n, r = self.rung
        return [sys.executable, "-u", "-m", "ctde_v0.train_ctde",
                "--grid", str(g), "--n-agents", str(n), "--comm-r", str(r),
                "--iters", str(iters), "--rollouts", str(rollouts), "--seed", str(self.seed),
                "--run-dir", self.run_dir, "--ckpt"] + _FIXED + self.extra


def _build_units(out, seeds):
    """5 arms × each seed, every unit FRESH. Run-dir <out>/seed{S}/{arm}."""
    units = []
    for s in seeds:
        for aid in _ARM_IDS:
            rd = os.path.join(out, f"seed{s}", aid)
            units.append(Unit(f"s{s}/{aid}", rd, s, list(_ARMS[aid])))
    return units


def main(argv=None):
    p = argparse.ArgumentParser(
        description="T3 connectivity test — can the soft dual hold a global-λ₂ target @32²/10 "
                    "(soft/lagrangian/pid × θ∈{0.5,0.7}, 5 arms × seeds)")
    p.add_argument("--out", type=str, default="runs/lag_t3",
                   help="parent dir for per-seed/per-arm run-dirs (rel to SharedExploration)")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..S-1)")
    p.add_argument("--iters", type=int, default=2000, help="PPO iters per run")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per iter")
    p.add_argument("--jobs", type=int, default=3,
                   help="max concurrent train_ctde subprocesses (balthar ~3 @32²/10)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    a = p.parse_args(argv)

    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))
    units = _build_units(out, seeds)

    print(f"=== T3 conn-target: {len(units)} runs (5 arms × {len(seeds)} seed) @32²/10, "
          f"iters={a.iters}, jobs={a.jobs}, out={out} ===", flush=True)
    print(f"    fixed: {' '.join(_FIXED)}", flush=True)
    print(f"    target θ (λ₂ floor) swept {{0.5, 0.7}}; mechanisms soft/lagrangian/pid; "
          f"watch dual_lambda_next (λ) + connectivity_real (λ₂) in history.json", flush=True)

    if a.dry_run:
        for u in units:
            print(f"  {u.uid:16s} {u.extra}", flush=True)
        print(f"\n[dry-run] {len(units)} runs planned; nothing launched.", flush=True)
        return

    t0 = time.time()
    done = _schedule(units, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== T3 done: {len(done)}/{len(units)} produced model.eqx "
          f"({round(time.time() - t0, 1)}s) ===", flush=True)
    print(f"compare per-arm history.json under {out}/seed*/ "
          f"(coverage↔connectivity_real frontier + dual_lambda_next tracking θ).",
          flush=True)


if __name__ == "__main__":
    main()
