"""Frontier sweep — trace the coverage<->connectivity Pareto by sweeping the COVERAGE
reward weight (the cov:conn trade-off dial), SOFT mechanism only, NO action_mask.

~6 points x N seeds @32^2/10, OPEN terrain, cover_r=0. Each point = one --w-coverage setting
against a fixed soft connectivity penalty; raising w-coverage tilts the balance toward coverage
(and should shed connectivity), tracing the frontier. Report RAW coverage + REAL connectivity per
point -> the empirical Pareto curve (NOT '% of optimal' — that yardstick is retired). Anchors
already on the curve: slamattn ~43/73 (conn-dominant), old frontier_attn ~98/32 (cov-dominant).

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_frontier.py --out runs/frontier --seeds 4 --iters 2000 --jobs 1
"""
from __future__ import annotations

import argparse
import os
import sys

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _schedule

# SOFT connectivity only (no hard mask); open terrain (no --terrain). Role split + frontier-attn,
# collision-mask, 100-step horizon, SLAM on — held fixed; only --w-coverage moves.
_FIXED = ["--role-picker", "expl_relay",
          "--mechanism", "soft_lambda", "--conn-signal", "local_edge_margin",
          "--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--horizon", "100", "--sense-walls"]
_WCOV = [1.0, 2.0, 3.0, 5.0, 8.0, 12.0]            # the 6 frontier points (coverage emphasis)


class Unit:
    def __init__(self, uid, run_dir, seed, wc):
        self.uid = uid; self.run_dir = run_dir; self.rung = (32, 10, 5)
        self.seed = seed; self.extra = ["--w-coverage", str(wc)]
        self.init_from_dir = None; self.needs_dir = None; self.proc = None

    @property
    def init_from(self): return None

    def ready(self): return True

    def cmd(self, iters, rollouts):
        g, n, r = self.rung
        return [sys.executable, "-u", "-m", "ctde_v0.train_ctde",
                "--grid", str(g), "--n-agents", str(n), "--comm-r", str(r),
                "--iters", str(iters), "--rollouts", str(rollouts), "--seed", str(self.seed),
                "--run-dir", self.run_dir, "--ckpt"] + _FIXED + self.extra


def _build(out, seeds):
    units = []
    for s in seeds:
        for wc in _WCOV:
            rd = os.path.join(out, f"seed{s}", f"wcov{wc}")
            units.append(Unit(f"s{s}/wcov{wc}", rd, s, wc))
    return units


def main(argv=None):
    p = argparse.ArgumentParser(description="6-point cov-conn frontier sweep (soft, no mask)")
    p.add_argument("--out", default="runs/frontier")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))
    units = _build(out, seeds)
    print(f"=== frontier: {len(units)} runs ({len(_WCOV)} w-coverage points x {len(seeds)} seeds) "
          f"@32^2/10 OPEN cover_r=0 SOFT no-mask, jobs={a.jobs} ===", flush=True)
    print(f"    w-coverage points: {_WCOV}", flush=True)
    if a.dry_run:
        for u in units:
            print(f"  {u.uid:16s} {u.extra}")
        return
    done = _schedule(units, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== frontier done: {len(done)}/{len(units)} ===", flush=True)


if __name__ == "__main__":
    main()
