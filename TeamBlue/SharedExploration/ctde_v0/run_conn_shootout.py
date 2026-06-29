"""Connectivity-mechanism shootout — the 90/90 assault at 32²/10.

Throws every connectivity mechanism the trainer has at the coverage<->connectivity wall, to
find the one that holds connectivity while maximising coverage. Directly tests the deep-research
verdict: a LEARNED Lagrangian connectivity constraint (lagrangian / pid_lagrangian — RCPO /
PID-RCPO) should beat the brittle fixed barrier-penalty and the soft penalty; the hard action
mask is the other strong candidate (our prior "guardrail beats soft by ~20pts").

  MECHANISM (4): maskhard (action_mask, hard guardrail) · soft (soft_lambda penalty) ·
    lag (lagrangian = learned dual) · pidlag (pid_lagrangian = PID dual, Stooke 2020).
  ARM (2): role (explorer/relay) · base (homogeneous).
  Fixed: 32²/10 · comm_r 5 · collision-mask on · frontier-attn · 100 steps · up-weighted
  coverage (--w-coverage 3) · conn-signal local_edge_margin (per-agent degree ≥ 1).

Per seed: 4 × 2 = 8 runs. Run-dir <out>/seed{S}/{arm}_{mech}. FRESH (no warm-start) — this is
the screen; the winner gets pushed (longer + laddered + multi-seed) in the follow-up.

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_conn_shootout.py --out runs/connshoot --seeds 1 --iters 2000 --jobs 3
"""
from __future__ import annotations

import argparse
import os
import sys
import time

from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _done, _schedule

_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--horizon", "100", "--w-coverage", "3"]
_MECHS = {
    "maskhard": ["--mechanism", "action_mask"],
    "soft":     ["--mechanism", "soft_lambda", "--conn-signal", "local_edge_margin"],
    "lag":      ["--mechanism", "lagrangian", "--conn-signal", "local_edge_margin"],
    "pidlag":   ["--mechanism", "pid_lagrangian", "--conn-signal", "local_edge_margin"],
}
_ARMS = {"role": ["--role-picker", "expl_relay"], "base": []}
_MECH_IDS = ["maskhard", "soft", "lag", "pidlag"]
_ARM_IDS = ["role", "base"]


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


def _build_units(out, seeds):
    units = []
    for s in seeds:
        for aid in _ARM_IDS:
            for mid in _MECH_IDS:
                rd = os.path.join(out, f"seed{s}", f"{aid}_{mid}")
                units.append(Unit(f"s{s}/{aid}_{mid}", rd, s, _ARMS[aid] + _MECHS[mid]))
    return units


def main(argv=None):
    p = argparse.ArgumentParser(description="connectivity-mechanism shootout @32²/10")
    p.add_argument("--out", default="runs/connshoot")
    p.add_argument("--seeds", type=int, default=1)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=3)
    p.add_argument("--dry-run", action="store_true")
    a = p.parse_args(argv)
    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))
    units = _build_units(out, seeds)
    print(f"=== conn shootout: {len(units)} runs (4 mech × 2 arm × {len(seeds)} seed) "
          f"@32²/10, iters={a.iters}, jobs={a.jobs} ===", flush=True)
    print(f"    fixed: {' '.join(_FIXED)}", flush=True)
    if a.dry_run:
        for u in units:
            print(f"  {u.uid:18s} {u.extra}")
        return
    t0 = time.time()
    done = _schedule(units, a.iters, a.rollouts, max(1, a.jobs))
    print(f"\n=== shootout done: {len(done)}/{len(units)} produced model.eqx "
          f"({round(time.time() - t0)}s) ===", flush=True)


if __name__ == "__main__":
    main()
