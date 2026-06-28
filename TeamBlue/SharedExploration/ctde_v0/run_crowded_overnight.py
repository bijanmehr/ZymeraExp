"""Overnight CROWDED-MAP curriculum batch — learn dense obstacle fields, not zero-shot.

Curriculum warm-start ladder (16²/4 → 24²/6 → 32²/10) on the CONNECTIVITY-SAFE crowded
distribution ``--terrain crowded_mix`` (per-reset draw over clutter / pillars / rooms+clutter;
ctde_v0.terrains guarantees the free space is one component, so coverage stays well-posed).
Each rung warm-starts the previous (the scale-invariant transfer the campaign is about), and
obstacle DENSITY is held ~constant (~15% of cells) by scaling ``--n-obstacles`` per rung so the
warm-start sees the same clutter level at every scale. comm_r FIXED 5 on the honest spec.

Two cornerstones × two exploration drives (the obstacle-batch lesson: coverage-bump helps):

  ARM (2):     role  (--role-picker expl_relay, the cognition-batch winner) | base (reference).
  EXPLORE (2): eoff  (default reward) | ebump (--w-coverage 3).

  LADDER (3, curriculum-chained): 16x16/4 → 24x24/6 → 32x32/10, rung i warm-starts rung i-1
  (same arm+explore+seed) via --init-from.

Per seed: 2 arms × 2 explore × 3 rungs = 12 runs (4 curriculum chains). Run-dir
``<out>/seed{S}/{arm}_{explore}/{grid}x{grid}x{n}`` — a rung is skipped once it holds model.eqx
(resumable), and a rung stays blocked until its predecessor's model.eqx exists.

    # plan only:
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python \
        ctde_v0/run_crowded_overnight.py --dry-run --seeds 1

    # on balthar (GPU; memory-bound → keep --jobs small when 32²/10 is in play):
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_crowded_overnight.py --out runs/crowded --seeds 1 --iters 1200 \
        --rollouts 16 --jobs 2

===>>> This launcher LAUNCHES REAL TRAINING; run it deliberately. <<<===
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _done, _schedule

# honest fixed spec (identical every run) + the crowded distribution.
_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--mechanism", "soft_lambda", "--horizon", "100",
          "--terrain", "crowded_mix", "--rooms", "3",
          "--pillar-spacing", "4", "--pillar-size", "2"]

_ARMS = {"role": ["--role-picker", "expl_relay"], "base": []}
_EXPLORES = {"eoff": [], "ebump": ["--w-coverage", "3"]}
_LADDER = [(16, 4), (24, 6), (32, 10)]            # (grid, n_agents); comm_r FIXED 5
_DENSITY = 0.15                                    # obstacle fraction held ~constant per rung

_ARM_IDS = ["role", "base"]
_EXPLORE_IDS = ["eoff", "ebump"]


def _nobs(g):
    return round(_DENSITY * g * g)


class Unit:
    """One curriculum rung: id, run-dir, (grid, n, comm_r), seed, extra flags, the predecessor
    rung's run-dir to warm-start from / wait on (None for the first rung)."""
    def __init__(self, uid, run_dir, rung, seed, extra, prev_dir):
        self.uid = uid
        self.run_dir = run_dir
        self.rung = rung
        self.seed = seed
        self.extra = extra
        self.init_from_dir = prev_dir
        self.needs_dir = prev_dir
        self.proc = None

    @property
    def init_from(self):
        return os.path.join(self.init_from_dir, "model.eqx") if self.init_from_dir else None

    def ready(self):
        return self.needs_dir is None or _done(self.needs_dir)

    def cmd(self, iters, rollouts):
        g, n, r = self.rung
        c = [sys.executable, "-u", "-m", "ctde_v0.train_ctde",
             "--grid", str(g), "--n-agents", str(n), "--comm-r", str(r),
             "--n-obstacles", str(_nobs(g)),
             "--iters", str(iters), "--rollouts", str(rollouts), "--seed", str(self.seed),
             "--run-dir", self.run_dir, "--ckpt"] + _FIXED + self.extra
        if self.init_from:
            c += ["--init-from", self.init_from]
        return c


def _build_units(out, seeds, max_grid=32):
    """Curriculum chains. ``max_grid`` caps the ladder (e.g. 24 → run only 16²/24²,
    deferring the memory-heavy 32² rungs); the warm-start chain still links across
    launches because a later full-ladder run finds the 24² model.eqx and continues."""
    units = []
    for s in seeds:
        for aid in _ARM_IDS:
            for eid in _EXPLORE_IDS:
                prev = None
                for (g, n) in _LADDER:
                    if g > max_grid:
                        continue
                    extra = _ARMS[aid] + _EXPLORES[eid]
                    run_dir = os.path.join(out, f"seed{s}", f"{aid}_{eid}", f"{g}x{g}x{n}")
                    uid = f"s{s}/{aid}_{eid}/{g}x{g}x{n}"
                    units.append(Unit(uid, run_dir, (g, n, 5), s, extra, prev))
                    prev = run_dir
    return units


def main(argv=None):
    p = argparse.ArgumentParser(
        description="overnight crowded-map curriculum batch — ARM × EXPLORE × LADDER "
                    "(2×2×3 = 12 curriculum runs per seed) on terrain=crowded_mix")
    p.add_argument("--out", type=str, default="runs/crowded",
                   help="parent dir for per-seed run-dirs (rel to SharedExploration)")
    p.add_argument("--seeds", type=int, default=1, help="number of seeds (0..S-1)")
    p.add_argument("--iters", type=int, default=1200, help="PPO iters per rung")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per iter")
    p.add_argument("--jobs", type=int, default=2,
                   help="max concurrent train_ctde subprocesses (memory-bound: ~3 @32/10)")
    p.add_argument("--max-grid", type=int, default=32,
                   help="cap the ladder at this grid size (e.g. 24 defers the 32² rungs)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    args = p.parse_args(argv)

    out = args.out if os.path.isabs(args.out) else os.path.join(_PKG_PARENT, args.out)
    seeds = list(range(args.seeds))
    units = _build_units(out, seeds, max_grid=args.max_grid)

    print(f"=== crowded-map curriculum batch: {len(units)} runs "
          f"({len(_ARM_IDS)} arms × {len(_EXPLORE_IDS)} explore × {len(_LADDER)} rungs "
          f"= {len(_ARM_IDS)*len(_EXPLORE_IDS)*len(_LADDER)}/seed × {len(seeds)} seeds), "
          f"out={out}, iters={args.iters}, jobs={args.jobs} ===", flush=True)
    print(f"    fixed spec: {' '.join(_FIXED)}", flush=True)
    print(f"    density ~{int(_DENSITY*100)}% per rung → n_obstacles "
          f"{[(f'{g}²', _nobs(g)) for g, _ in _LADDER]}", flush=True)

    if args.dry_run:
        for u in units:
            g, n, r = u.rung
            dep = f"  <- {os.path.basename(os.path.dirname(u.needs_dir))}/{os.path.basename(u.needs_dir)}" if u.needs_dir else "  (fresh)"
            print(f"  {u.uid:28s} grid={g} N={n} nobs={_nobs(g)} extra={u.extra}{dep}", flush=True)
        print(f"\n[dry-run] {len(units)} runs planned; nothing launched.", flush=True)
        return

    t0 = time.time()
    done = _schedule(units, args.iters, args.rollouts, max(1, args.jobs))
    print(f"\n=== batch done: {len(done)}/{len(units)} produced model.eqx "
          f"({round(time.time() - t0, 1)}s) ===", flush=True)
    print(f"compare per-rung history.json under {out}/seed*/ "
          f"(coverage_pct / connectivity_real / snd / role_div).", flush=True)


if __name__ == "__main__":
    main()
