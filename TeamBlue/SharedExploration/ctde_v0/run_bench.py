"""Reproducibility benchmark — re-run the methods-that-matter under ONE fixed protocol, many seeds.

Most prior rankings were single-seed; this nails them down. Fixed protocol for EVERY run:
  @32²/10 · cover_r=0 (real visited coverage, scored vs the per-map oracle optimal) · SLAM ON
  (sense_walls) · frontier-attn · collision-mask · 100-step horizon · SOFT/learned connectivity only
  (NO hard action_mask).

Varies the 3 axes that actually moved the needle — role-split, connectivity (soft vs learned-
Lagrangian), and the bump-explore term — as ablations around the best, across 3 terrains, N seeds.
Reports coverage_pct + connectivity_real per (method, terrain), with seed spread.

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_bench.py --out runs/bench --seeds 4 --iters 2000 --jobs 1
"""
from __future__ import annotations

import argparse
import os
import sys

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _schedule

_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--horizon", "100", "--sense-walls"]          # SLAM always ON; cover_r=0 (default)
_SOFT = ["--mechanism", "soft_lambda", "--conn-signal", "local_edge_margin"]
_LAG = ["--mechanism", "lagrangian", "--conn-signal", "local_edge_margin"]
_ROLE = ["--role-picker", "expl_relay"]
_METHODS = {
    "role_soft_bump": _ROLE + _SOFT + ["--w-coverage", "3"],   # the best — full recipe
    "base_soft_bump":         _SOFT + ["--w-coverage", "3"],   # ablate role-split
    "role_lag_bump":  _ROLE + _LAG  + ["--w-coverage", "3"],   # learned-Lagrangian vs soft
    "role_soft_flat": _ROLE + _SOFT + ["--w-coverage", "1"],   # ablate the bump-explore
}
_TERRAINS = {
    "open":    [],
    "rooms":   ["--terrain", "rooms", "--rooms", "3"],
    "crowded": ["--terrain", "crowded_mix", "--n-obstacles", "40"],
}
_M_IDS = list(_METHODS); _T_IDS = list(_TERRAINS)


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
        for m in _M_IDS:
            for t in _T_IDS:
                rd = os.path.join(out, f"seed{s}", f"{m}__{t}")
                units.append(Unit(f"s{s}/{m}/{t}", rd, s, _METHODS[m] + _TERRAINS[t]))
    return units


def main(argv=None):
    p = argparse.ArgumentParser(description="reproducibility benchmark @32²/10 cover_r=0")
    p.add_argument("--out", default="runs/bench")
    p.add_argument("--seeds", type=int, default=4)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))
    units = _build(out, seeds)
    print(f"=== bench: {len(units)} runs ({len(_M_IDS)} methods × {len(_T_IDS)} terrains × "
          f"{len(seeds)} seeds) @32²/10 cover_r=0 SLAM-on, jobs={a.jobs} ===", flush=True)
    if a.dry_run:
        for u in units:
            print(f"  {u.uid:22s} {u.extra}")
        return
    done = _schedule(units, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== bench done: {len(done)}/{len(units)} ===", flush=True)


if __name__ == "__main__":
    main()
