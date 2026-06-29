"""Warm-start A/B — from-scratch vs --init-from a checkpoint, side by side.

Runs at 16²/4 (small enough to fit alongside a 32² job). Warm arm inits actor+critic from
--ckpt (the LPAC backbone is scale-invariant, so a 32² checkpoint loads into a 16² run). Same
best architecture both arms; the only difference is the init. Reports coverage/connectivity so
we can see whether warm-start converges higher/faster than current-code-from-scratch.

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_warmab.py --ckpt runs/slamattn/seed0/max_slam/model.eqx --seeds 2 --jobs 1
"""
from __future__ import annotations

import argparse
import os
import sys

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _schedule

_FIXED = ["--role-picker", "expl_relay",
          "--mechanism", "lagrangian", "--conn-signal", "local_edge_margin",
          "--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--horizon", "100", "--w-coverage", "3", "--agg", "max", "--sense-walls"]


class Unit:
    def __init__(self, uid, run_dir, seed, extra):
        self.uid = uid; self.run_dir = run_dir; self.rung = (16, 4, 5)
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
    p = argparse.ArgumentParser(description="warm-start vs from-scratch A/B @16²/4")
    p.add_argument("--ckpt", required=True, help="checkpoint to warm-start the warm arm from")
    p.add_argument("--out", default="runs/warmab")
    p.add_argument("--seeds", type=int, default=2)
    p.add_argument("--iters", type=int, default=1500)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    units = []
    for s in range(a.seeds):
        units.append(Unit(f"scratch_s{s}", os.path.join(out, f"scratch_s{s}"), s, []))
        units.append(Unit(f"warm_s{s}", os.path.join(out, f"warm_s{s}"), s, ["--init-from", a.ckpt]))
    print(f"=== warmab: {len(units)} runs (scratch vs warm) @16²/4, iters={a.iters}, "
          f"ckpt={a.ckpt}, jobs={a.jobs} ===", flush=True)
    if a.dry_run:
        for u in units:
            print(f"  {u.uid:12s} {u.extra}")
        return
    done = _schedule(units, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== warmab done: {len(done)}/{len(units)} ===", flush=True)


if __name__ == "__main__":
    main()
