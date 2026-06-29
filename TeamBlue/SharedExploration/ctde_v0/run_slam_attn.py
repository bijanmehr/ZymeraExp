"""SLAM × neighbor-attention head-to-head @32²/10 — the 'ready' transformer experiment.

Two new knobs, on the shootout-winning base (role split + learned-Lagrangian connectivity):
  AGG (2):        max (current pooling) vs multihead (softmax attention over neighbours = GAT).
  PERCEPTION (2): --no-sense-walls (old wall-blind belief) vs --sense-walls (SLAM occupancy).

The 2×2 isolates the attention lift, the wall-perception lift, and their interaction. `max_blind`
≈ reproduces the connectivity-shootout's role+lag (sanity anchor at ~42cov/85conn). FRESH (no
warm-start) — the screen; the winner gets the laddered + multi-seed follow-up.

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_slam_attn.py --out runs/slamattn --seeds 3 --iters 2000 --jobs 4
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _schedule

# Shootout winner held fixed: explorer/relay roles + learned-Lagrangian (RCPO) connectivity,
# frontier-attn explorer, hard collision-mask, 100-step horizon, up-weighted coverage.
_FIXED = ["--role-picker", "expl_relay",
          "--mechanism", "lagrangian", "--conn-signal", "local_edge_margin",
          "--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--horizon", "100", "--w-coverage", "3"]
# 2×2: aggregation × perception.
_ARMS = {
    "max_blind":  ["--agg", "max",       "--no-sense-walls"],   # old baseline (anchor)
    "max_slam":   ["--agg", "max",       "--sense-walls"],      # + wall perception
    "attn_blind": ["--agg", "multihead", "--no-sense-walls"],   # + neighbor attention
    "attn_slam":  ["--agg", "multihead", "--sense-walls"],      # + both
}
_ARM_IDS = ["max_blind", "max_slam", "attn_blind", "attn_slam"]


class Unit:
    """Single-scale 32²/10, fresh (no warm-start) — mirrors run_conn_shootout.Unit."""

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


def _build_units(out, seeds, terrain_extra):
    units = []
    for s in seeds:
        for aid in _ARM_IDS:
            rd = os.path.join(out, f"seed{s}", aid)
            units.append(Unit(f"s{s}/{aid}", rd, s, _ARMS[aid] + terrain_extra))
    return units


def main(argv=None):
    p = argparse.ArgumentParser(description="SLAM × neighbor-attention head-to-head @32²/10")
    p.add_argument("--out", default="runs/slamattn")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--seed-start", type=int, default=0,
                   help="first seed index (for chained extensions, e.g. --seed-start 3 --seeds 3)")
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=4)
    p.add_argument("--terrain", default="open",
                   help="terrain for ALL arms (open/rooms/walls/clutter/pillars/mixed/crowded_mix). "
                        "Use mixed/crowded_mix to actually test SLAM perception (needs walls present).")
    p.add_argument("--n-obstacles", type=int, default=0)
    p.add_argument("--rooms", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seed_start, a.seed_start + a.seeds))
    terrain_extra = ["--terrain", a.terrain, "--n-obstacles", str(a.n_obstacles), "--rooms", str(a.rooms)]
    units = _build_units(out, seeds, terrain_extra)
    print(f"=== slam×attn: {len(units)} runs (2 agg × 2 perception × {len(seeds)} seed) "
          f"@32²/10, iters={a.iters}, jobs={a.jobs} ===", flush=True)
    print(f"    fixed: {' '.join(_FIXED)}", flush=True)
    if a.dry_run:
        for u in units:
            print(f"  {u.uid:16s} {u.extra}")
        return
    t0 = time.time()
    done = _schedule(units, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== slam×attn done: {len(done)}/{len(units)} produced model.eqx "
          f"({round(time.time() - t0)}s) ===", flush=True)


if __name__ == "__main__":
    main()
