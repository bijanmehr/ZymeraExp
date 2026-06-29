"""Warm-start into a BIGGER world — does the occupancy belief make small→large transfer
better? (User hypothesis: occupancy/boundary should pay off most when you scale up, because
the policy learns general "chase the occupancy frontier, stay inside the field" rules that
don't depend on grid size.)

Phase 1 trains @16²/4; phase 2 warm-starts @32²/10 from the MATCHING phase-1 checkpoint
(`--init-from`; the LPAC backbone is scale-invariant, so a 16² policy loads into a 32² run).
Two arms, held consistent across phases so the obs-channel count matches on load:
  base : SLAM-only                       (5 obs channels)
  occ  : + --sense-free --boundary       (7 obs channels: occupancy + occ_frontier + boundary)
Read against the from-scratch 32² numbers from run_occ — the question is whether the occ
warm-start climbs higher / faster (% of the per-map optimal) than the base warm-start.
cover_r=0, 100-step horizon, SLAM on, no hard mask.

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_warmbig.py --out runs/warmbig --seeds 2 --jobs 1
"""
from __future__ import annotations

import argparse
import os
import sys

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _schedule

_FIXED = ["--role-picker", "expl_relay",
          "--mechanism", "lagrangian", "--conn-signal", "local_edge_margin",
          "--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--horizon", "100", "--w-coverage", "3", "--sense-walls"]
_ARMS = {"base": [], "occ": ["--sense-free", "--boundary"]}
_TERRAINS = {"open": [], "rooms": ["--terrain", "rooms", "--rooms", "3"]}
_A_IDS = list(_ARMS); _T_IDS = list(_TERRAINS)


class Unit:
    def __init__(self, uid, run_dir, seed, extra, rung):
        self.uid = uid; self.run_dir = run_dir; self.rung = rung
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


def main(argv=None):
    p = argparse.ArgumentParser(description="warm-start into a bigger world (16²→32²), occ vs base")
    p.add_argument("--out", default="runs/warmbig")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--iters", type=int, default=2000)          # phase-2 (big) iters
    p.add_argument("--iters-small", type=int, default=1500)    # phase-1 (small) iters
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))

    p1, p2 = [], []
    for s in seeds:
        for arm in _A_IDS:
            for t in _T_IDS:
                small_rd = os.path.join(out, "small16", f"seed{s}", f"{arm}__{t}")
                big_rd = os.path.join(out, "warm32", f"seed{s}", f"{arm}__{t}")
                base_extra = _ARMS[arm] + _TERRAINS[t]
                p1.append(Unit(f"s{s}/{arm}/{t}/16", small_rd, s, base_extra, (16, 4, 5)))
                # phase 2 warm-starts from the SAME-arm 16² checkpoint (channel counts match)
                p2.append(Unit(f"s{s}/{arm}/{t}/32", big_rd, s,
                               base_extra + ["--init-from", os.path.join(small_rd, "model.eqx")],
                               (32, 10, 5)))
    print(f"=== warmbig: phase1 {len(p1)} @16²/4, phase2 {len(p2)} @32²/10 warm "
          f"({len(_A_IDS)} arms × {len(_T_IDS)} terrains × {len(seeds)} seeds), jobs={a.jobs} ===",
          flush=True)
    if a.dry_run:
        for u in p1 + p2:
            print(f"  {u.uid:18s} {u.rung} {u.extra}")
        return
    print("--- phase 1: train @16²/4 (the small-world seed policies) ---", flush=True)
    d1 = _schedule(p1, a.iters_small, a.rollouts, max(1, a.jobs))
    print(f"--- phase 1 done: {len(d1)}/{len(p1)} ---", flush=True)
    print("--- phase 2: warm-start @32²/10 from the 16² checkpoints ---", flush=True)
    d2 = _schedule(p2, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== warmbig done: phase1 {len(d1)}/{len(p1)}, phase2 {len(d2)}/{len(p2)} ===",
          flush=True)


if __name__ == "__main__":
    main()
