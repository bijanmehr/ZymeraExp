"""Serial-32 orchestrator — run every 32²/10 unit ONE AT A TIME, pack the light units in parallel.

WHY: one 96 GB card. A Lagrangian 32²/10 training needs ~50 GB (dual + aux λ̂₂ estimator
back-propagated through the 100-step rollout); the fixed-penalty variant ~17 GB; 24²/6 ~8.8 GB;
16²/4 ~3.5 GB. Two heavy units = ~100 GB > 96 → OOM. So the 32² work MUST be serialized. The
light units are cheap and can safely run alongside the single heavy one.

TWO TIERS (launched as two coexisting tmux sessions by run_serial32.sh):
  --tier heavy : every grid==32 unit across T2/T3/T4, jobs=1  → strictly ONE at a time (~50 GB).
  --tier light : every grid<32  unit across T2/T3/T4, jobs=J  → packed (~J×9 GB).
Peak = 1 heavy (~50 GB) + J light (~J×9 GB). J=3 → ~77 GB, comfortably under 96 with compile
headroom. Because 'heavy' is jobs=1 and 'light' only ever holds grid<32 units, the number of
32² trainings on the card is provably ≤ 1 at all times — bulletproof, no cross-session race.

Both tiers write checkpoints into the SAME runs/{roles_t2,lag_t3,critic_t4} tree, so the existing
_report_for.sh renders each suite unchanged, and every unit is resumable (a run-dir with a
model.eqx is skipped). Each unit carries its own suite's iters (T2=1500, T3/T4=2000). All units
are FROM SCRATCH (no warm-start, no cross-unit deps) so scheduling is a plain bounded pool.

    # plan only (no training):
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        -m ctde_v0.run_serial32 --tier heavy --dry-run

    # the tiers (run_serial32.sh launches both as tmux zheavy/zlight):
    ... -m ctde_v0.run_serial32 --tier heavy --jobs 1 --seeds 3 --rollouts 16
    ... -m ctde_v0.run_serial32 --tier light --jobs 3 --seeds 3 --rollouts 16

===>>> This launcher LAUNCHES REAL TRAINING; run it deliberately. <<<===
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

from ctde_v0 import run_roles_t2, run_lag_t3, run_critic_t4

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)                       # .../SharedExploration
_FIEDLER = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "FiedlerValueEstimation"))

# (subdir under runs/, the runner module that builds its units, that suite's PPO iters)
_SUITES = [
    ("roles_t2",  run_roles_t2,  1500),
    ("lag_t3",    run_lag_t3,    2000),
    ("critic_t4", run_critic_t4, 2000),
]


def _done(run_dir):
    p = os.path.join(run_dir, "model.eqx")
    return os.path.exists(p) and os.path.getsize(p) > 0


def _run_env():
    env = dict(os.environ)
    pp = [_PKG_PARENT, _FIEDLER]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    return env


def _build_all(out_base, seeds):
    """Every training unit across the three suites, each tagged with its suite name + iters.
    out_base is the runs/ root; each suite builds its run-dirs under out_base/<subdir>."""
    units = []
    for name, mod, iters in _SUITES:
        out = os.path.join(out_base, name)
        us = mod._build_units(out, seeds)
        for u in us:
            u._iters = iters
            u._suite = name
        units.extend(us)
    return units


def _schedule(units, rollouts, jobs):
    """Bounded-parallel pool (all units independent). Skips done run-dirs; launches up to
    `jobs` at once with each unit's own iters; polls to completion."""
    env = _run_env()
    pending = [u for u in units if not _done(u.run_dir)]
    running = []
    done = 0
    for u in units:
        if _done(u.run_dir):
            print(f"[skip] {u._suite}:{u.uid} (model.eqx present)", flush=True)
            done += 1
    while pending or running:
        while pending and len(running) < jobs:
            u = pending.pop(0)
            os.makedirs(u.run_dir, exist_ok=True)
            cmd = u.cmd(u._iters, rollouts)
            print(f"[launch] {u._suite}:{u.uid}  ({u.rung[0]}²/{u.rung[1]}) iters={u._iters}",
                  flush=True)
            u.proc = subprocess.Popen(cmd, cwd=_PKG_PARENT, env=env)
            running.append(u)
        time.sleep(3)
        for u in list(running):
            rc = u.proc.poll()
            if rc is None:
                continue
            running.remove(u)
            ok = _done(u.run_dir)
            print(f"[finish] {u._suite}:{u.uid}  rc={rc}  {'ok' if ok else 'NO-CKPT'}",
                  flush=True)
            if ok:
                done += 1
    return done


def main(argv=None):
    p = argparse.ArgumentParser(
        description="Serial-32 orchestrator: 32² units one-at-a-time (heavy tier), "
                    "light 16²/24² units packed in parallel (light tier)")
    p.add_argument("--tier", choices=["heavy", "light"], required=True,
                   help="heavy = every grid==32 unit (jobs=1); light = every grid<32 unit")
    p.add_argument("--out", type=str, default="runs",
                   help="runs/ root (suite subdirs appended); rel to SharedExploration")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..S-1)")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per PPO iter")
    p.add_argument("--jobs", type=int, default=None,
                   help="concurrent trainings (default: heavy=1, light=3)")
    p.add_argument("--shard", type=str, default=None,
                   help="I/N round-robin split (e.g. 0/2, 1/2): run only units with index%%N==I, "
                        "for splitting a tier across GPUs/machines (skip-on-done keeps shards "
                        "non-overlapping and resumable)")
    p.add_argument("--dry-run", action="store_true", help="print the split, launch nothing")
    a = p.parse_args(argv)

    out_base = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))
    units = _build_all(out_base, seeds)

    is_heavy = (a.tier == "heavy")
    tier_units = [u for u in units if (u.rung[0] >= 32) == is_heavy]
    if a.shard:
        si, sn = (int(x) for x in a.shard.split("/"))
        tier_units = [u for k, u in enumerate(tier_units) if k % sn == si]
        print(f"    shard {si}/{sn}: {len(tier_units)} of this tier's units on this process", flush=True)
    jobs = a.jobs if a.jobs is not None else (1 if is_heavy else 3)

    n_heavy = sum(1 for u in units if u.rung[0] >= 32)
    n_light = len(units) - n_heavy
    print(f"=== serial32 [{a.tier}]: {len(tier_units)} units "
          f"(total {len(units)} = {n_heavy} heavy 32² + {n_light} light), "
          f"jobs={jobs}, rollouts={a.rollouts}, seeds={seeds} ===", flush=True)

    if a.dry_run:
        for u in tier_units:
            print(f"  {u._suite:9s} {u.uid:22s} grid={u.rung[0]:2d} N={u.rung[1]:2d} "
                  f"iters={u._iters}", flush=True)
        print(f"\n[dry-run] {len(tier_units)} '{a.tier}' units; nothing launched.", flush=True)
        return

    t0 = time.time()
    done = _schedule(tier_units, a.rollouts, jobs)
    print(f"\n=== SERIAL32 {a.tier} DONE: {done}/{len(tier_units)} produced model.eqx "
          f"({round(time.time() - t0, 1)}s) ===", flush=True)


if __name__ == "__main__":
    main()
