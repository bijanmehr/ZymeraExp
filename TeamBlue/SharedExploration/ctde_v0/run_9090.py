"""Best architecture toward 90/90 under the UNCERTAINTY-REDUCTION (sensed-coverage) metric.

Reframe (2026-06-29): the mission is information-gathering, so "coverage" = *sensed* coverage =
`coverage_pct` at cover_r=1 (credit for the sensed 3x3 footprint), NOT cells physically stood on.
This runs the shootout-winning architecture — role split + learned-Lagrangian SOFT connectivity
(NO hard action_mask) + frontier-attn — at cover_r=1 across DIFFERENT MAPS, 3 seeds each, and
reports coverage_pct (= sensed / uncertainty reduced) + connectivity_real. Let the numbers talk.

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_9090.py --out runs/r9090 --seeds 3 --iters 2000 --jobs 1
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _schedule

# Shootout-winning architecture + the sensing objective. SOFT connectivity only (no mask).
_FIXED = ["--role-picker", "expl_relay",
          "--mechanism", "lagrangian", "--conn-signal", "local_edge_margin",
          "--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--horizon", "100", "--w-coverage", "3",
          "--cover-r", "1", "--sense-walls"]
_MAPS = {
    "open":    [],
    "rooms":   ["--terrain", "rooms", "--rooms", "3"],
    "mixed":   ["--terrain", "mixed", "--n-obstacles", "40", "--rooms", "3"],
    "crowded": ["--terrain", "crowded_mix", "--n-obstacles", "40"],
}
_MAP_IDS = ["open", "rooms", "mixed", "crowded"]


class Unit:
    def __init__(self, uid, run_dir, seed, extra):
        self.uid = uid; self.run_dir = run_dir; self.rung = (32, 10, 5)
        self.seed = seed; self.extra = extra
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
        for mid in _MAP_IDS:
            rd = os.path.join(out, f"seed{s}", mid)
            units.append(Unit(f"s{s}/{mid}", rd, s, _MAPS[mid]))
    return units


def main(argv=None):
    p = argparse.ArgumentParser(description="best-arch toward 90/90 (sensed-coverage) across maps")
    p.add_argument("--out", default="runs/r9090")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))
    units = _build(out, seeds)
    print(f"=== r9090: {len(units)} runs ({len(_MAP_IDS)} maps × {len(seeds)} seeds) "
          f"@32²/10 cover_r=1 (sensed), jobs={a.jobs} ===", flush=True)
    print(f"    fixed: {' '.join(_FIXED)}", flush=True)
    if a.dry_run:
        for u in units:
            print(f"  {u.uid:14s} {u.extra}")
        return
    t0 = time.time()
    done = _schedule(units, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== r9090 done: {len(done)}/{len(units)} ({round(time.time() - t0)}s) ===", flush=True)


if __name__ == "__main__":
    main()
