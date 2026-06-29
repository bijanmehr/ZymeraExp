"""Occupancy A/B — does the occupancy belief beat the SLAM-only baseline, per terrain?

Same fixed best recipe both arms (role explorer/relay split + learned-Lagrangian SOFT
connectivity + frontier-attn + SLAM wall-sensing, NO hard mask); the ONLY difference is
the belief the policy sees:
  base : sense_walls only — walls in the belief, coverage-trail frontier            (5 obs ch)
  occ  : + --sense-free --boundary — the FULL sensed region enters the belief (a real
         free/occupied/unknown occupancy map), a true Yamauchi `occ_frontier` to chase,
         and the field-edge `boundary` channel                                       (7 obs ch)

@32²/10, cover_r=0 (REAL visited coverage, scored vs the per-map oracle optimal:
open 72 / rooms 37 / mixed 31 / crowded 40), 100-step horizon. Reports coverage_pct +
connectivity_real per (arm, terrain) over N seeds — so occupancy's lift (or null) is clean
per map. The corridor maps (rooms/mixed/crowded) are where walls + boundary should matter.

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_occ.py --out runs/occ --seeds 2 --iters 2000 --jobs 1
"""
from __future__ import annotations

import argparse
import os
import sys

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _schedule

_FIXED = ["--role-picker", "expl_relay",
          "--mechanism", "lagrangian", "--conn-signal", "local_edge_margin",
          "--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--horizon", "100", "--w-coverage", "3", "--sense-walls"]    # SLAM always on; cover_r=0
_ARMS = {
    "base": [],                                    # SLAM-only (the current best)
    "occ":  ["--sense-free", "--boundary"],        # + occupancy belief + field boundary
}
_TERRAINS = {
    "open":    [],
    "rooms":   ["--terrain", "rooms", "--rooms", "3"],
    "mixed":   ["--terrain", "mixed", "--n-obstacles", "40", "--rooms", "3"],
    "crowded": ["--terrain", "crowded_mix", "--n-obstacles", "40"],
}
_A_IDS = list(_ARMS); _T_IDS = list(_TERRAINS)


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
        for a in _A_IDS:
            for t in _T_IDS:
                rd = os.path.join(out, f"seed{s}", f"{a}__{t}")
                units.append(Unit(f"s{s}/{a}/{t}", rd, s, _ARMS[a] + _TERRAINS[t]))
    return units


def main(argv=None):
    p = argparse.ArgumentParser(description="occupancy A/B @32²/10 cover_r=0")
    p.add_argument("--out", default="runs/occ")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))
    units = _build(out, seeds)
    print(f"=== occ: {len(units)} runs ({len(_A_IDS)} arms × {len(_T_IDS)} terrains × "
          f"{len(seeds)} seeds) @32²/10 cover_r=0 SLAM-on, jobs={a.jobs} ===", flush=True)
    print(f"    arms: base={_ARMS['base']}  occ={_ARMS['occ']}", flush=True)
    if a.dry_run:
        for u in units:
            print(f"  {u.uid:20s} {u.extra}")
        return
    done = _schedule(units, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== occ done: {len(done)}/{len(units)} ===", flush=True)


if __name__ == "__main__":
    main()
