"""T2 role-picker baseline runner — flat vs hand-coded role split on the OPEN density ladder.

The T2 baseline asks a single clean question: does the hand-coded explorer/relay role split
(``--role-picker expl_relay``) beat the flat homogeneous shared explorer on the honest shared
spec, held FIXED across a density-pinned OPEN world ladder? Every run is trained FROM SCRATCH
(no warm-start, no inter-run deps) and graded on the two metrics the trainer logs every iter:
``coverage_pct`` and the REAL connectivity bar ``connectivity_real`` (λ₂>0.5).

Belief: the fixed spec now carries the FULL 7-channel SLAM occupancy KB as the baseline belief
(``--sense-walls --sense-free --boundary``: free/occupied/unknown + occ_frontier + boundary),
so both arms see occupancy rather than the old 5-channel coverage-trail belief.

TWO axes, crossed in full (2 arms × 3 worlds = 6 units per seed):

  ARM (2) — what coordinates the team:
    flat  : the homogeneous shared explorer, no extra flags. The REFERENCE.
    role  : the hand-coded explorer/relay split (--role-picker expl_relay).

  WORLD (3) — density-pinned OPEN ladder (grid / n_agents / comm_r), all open terrain:
    o16   : 16×16, 4 agents,  comm_r 5, open.
    o24   : 24×24, 6 agents,  comm_r 5, open.
    o32   : 32×32, 10 agents, comm_r 5, open.

Per seed: 2 arms × 3 worlds = 6 units. With the DEFAULT ``--seeds 0,1,2`` that is 18 runs.
Each unit's run-dir is ``<out>/seed{S}/{world}/{arm}`` (e.g. ``runs/roles_t2/seed0/o24/role``).

Every unit is INDEPENDENT and FRESH: the run-dir is skipped once it holds a ``model.eqx``;
nothing warm-starts anything (init_from=None, needs_dir=None for every unit). ``--jobs N`` runs
up to N train_ctde subprocesses at once (balthar is MEMORY-BOUND — keep N small if o32 is in
play). ``--dry-run`` prints the full 18-unit plan (id, grid/N, all extra flags) and launches
nothing.

    # see the whole plan first (no training):
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python \
        -m ctde_v0.run_roles_t2 --dry-run

    # the real batch on balthar (drop JAX_PLATFORMS so it uses the GPU), 2 at a time:
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        -m ctde_v0.run_roles_t2 --out runs/roles_t2 --iters 1500 --rollouts 16 --jobs 2

===>>> This launcher LAUNCHES REAL TRAINING; run it deliberately. <<<===
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)                       # .../SharedExploration
_FIEDLER = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "FiedlerValueEstimation"))

# The honest fixed spec — IDENTICAL on every run. frontier-attn explorer + hard collision-mask
# + lagrangian mechanism on the local_edge_margin connectivity signal, heavier coverage reward,
# 100-step horizon. cover_r=0 is stated EXPLICITLY here (the confound fix — the god-view Voronoi
# budget ceiling depends on cover_r; it must never drift silently). The belief is now the FULL
# 7-channel SLAM occupancy KB: --sense-walls + --sense-free + --boundary give free/occupied/
# unknown occupancy, a Yamauchi occ_frontier, and the field-edge boundary channel (5→7 ch).
_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--mechanism", "lagrangian", "--conn-signal", "global_lambda2",
          "--constraint-threshold", "0.5",
          "--w-coverage", "3", "--sense-walls", "--sense-free", "--boundary",
          "--horizon", "100", "--cover-r", "0"]

# --- the two factorial axes (id -> the extra train_ctde flags that DEFINE that level) ---

# ARM: what coordinates the team (flat = no extra; role = hand-coded explorer/relay split).
_ARMS = {
    "flat": [],
    "role": ["--role-picker", "expl_relay"],
}

# WORLD: density-pinned OPEN ladder; grid / n_agents / comm_r, all open terrain.
#   id -> (grid, n_agents, comm_r, terrain_extra_flags)
_WORLDS = {
    "o16": (16, 4, 5, ["--terrain", "open"]),
    "o24": (24, 6, 5, ["--terrain", "open"]),
    "o32": (32, 10, 5, ["--terrain", "open"]),
}

# fixed cross order (arm × world) — every combination.
_ARM_IDS = ["flat", "role"]
_WORLD_IDS = ["o16", "o24", "o32"]


def _done(run_dir):
    p = os.path.join(run_dir, "model.eqx")
    return os.path.exists(p) and os.path.getsize(p) > 0


class Unit:
    """One train_ctde run: an id, its run-dir, the extra flags, the seed/world, and the
    run-dir of a predecessor whose model.eqx must exist before it can start (here always
    None — every obstacle run is FRESH)."""
    def __init__(self, uid, run_dir, rung, seed, extra, init_from_dir, needs_dir):
        self.uid = uid
        self.run_dir = run_dir
        self.rung = rung                     # (g, n, r)
        self.seed = seed
        self.extra = extra
        self.init_from_dir = init_from_dir   # run-dir whose model.eqx warm-starts this, or None
        self.needs_dir = needs_dir           # run-dir that must be done first, or None
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
             "--iters", str(iters), "--rollouts", str(rollouts), "--seed", str(self.seed),
             "--run-dir", self.run_dir, "--ckpt"] + _FIXED + self.extra
        if self.init_from:
            c += ["--init-from", self.init_from]
        return c


def _build_units(out, seeds):
    """The full cross of runs across seeds: ARM × WORLD, every unit FRESH (no warm-start, no
    deps). Run-dir <out>/seed{S}/{world}/{arm}."""
    units = []
    for s in seeds:
        for wid in _WORLD_IDS:
            g, n, r, world_extra = _WORLDS[wid]
            for aid in _ARM_IDS:
                extra = _ARMS[aid] + world_extra
                run_dir = os.path.join(out, f"seed{s}", wid, aid)
                uid = f"s{s}/{wid}/{aid}"
                units.append(Unit(uid, run_dir, (g, n, r), s, extra, None, None))
    return units


def _run_env():
    env = dict(os.environ)
    pp = [_PKG_PARENT, _FIEDLER]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    return env


def _schedule(units, iters, rollouts, jobs):
    """Dependency-aware bounded-parallel runner: repeatedly launch any not-done unit whose
    predecessor ckpt exists, up to ``jobs`` at once; poll; stop when all are done/blocked.
    (Obstacle units have no deps, so this is just a bounded-parallel pool.)"""
    env = _run_env()
    pending = [u for u in units if not _done(u.run_dir)]
    running = []
    skipped = [u for u in units if _done(u.run_dir)]
    for u in skipped:
        print(f"[skip] {u.uid} (model.eqx present)", flush=True)
    done_dirs = {u.run_dir for u in skipped}

    while pending or running:
        # launch ready units up to the jobs cap
        for u in list(pending):
            if len(running) >= jobs:
                break
            if u.ready():
                os.makedirs(u.run_dir, exist_ok=True)
                cmd = u.cmd(iters, rollouts)
                print(f"\n[launch] {u.uid}\n$ {' '.join(cmd)}", flush=True)
                u.proc = subprocess.Popen(cmd, cwd=_PKG_PARENT, env=env)
                running.append(u)
                pending.remove(u)
        if not running:
            # nothing running and nothing ready -> the rest is blocked on missing deps
            blocked = [u.uid for u in pending]
            if blocked:
                print(f"[blocked] no runnable units; waiting deps for: {blocked}", flush=True)
            if not pending:
                break
            time.sleep(5)
            continue
        # wait for any running unit to finish
        time.sleep(3)
        for u in list(running):
            rc = u.proc.poll()
            if rc is None:
                continue
            running.remove(u)
            status = "ok" if rc == 0 else f"rc={rc}"
            print(f"[finish] {u.uid} -> {status}", flush=True)
            if rc == 0 and _done(u.run_dir):
                done_dirs.add(u.run_dir)
            else:
                print(f"[warn] {u.uid} did not produce model.eqx (dependents stay blocked)",
                      flush=True)
    return done_dirs


def _parse_seeds(s):
    """Comma-separated seed list, e.g. '0,1,2' -> [0, 1, 2]."""
    return [int(x) for x in s.split(",") if x.strip() != ""]


def main(argv=None):
    p = argparse.ArgumentParser(
        description="T2 role-picker baseline — flat vs role split on the OPEN density ladder "
                    "(2 arms × 3 worlds = 6 fresh runs per seed)")
    p.add_argument("--out", type=str, default="runs/roles_t2",
                   help="parent dir for per-seed/per-world run-dirs (rel to SharedExploration)")
    p.add_argument("--seeds", type=_parse_seeds, default=[0, 1, 2],
                   help="comma-separated seed list (default 0,1,2)")
    p.add_argument("--iters", type=int, default=1500, help="PPO iters per run")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per iter")
    p.add_argument("--jobs", type=int, default=1,
                   help="max concurrent train_ctde subprocesses (balthar is memory-bound: "
                        "~3 @32/10, ~6 @24/6)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    args = p.parse_args(argv)

    out = args.out if os.path.isabs(args.out) else os.path.join(_PKG_PARENT, args.out)
    seeds = args.seeds
    units = _build_units(out, seeds)

    per_seed = len(_ARM_IDS) * len(_WORLD_IDS)
    print(f"=== T2 role-picker baseline: {len(units)} runs "
          f"(cross {len(_ARM_IDS)} arms × {len(_WORLD_IDS)} worlds = {per_seed}/seed; "
          f"× {len(seeds)} seeds {seeds}), out={out}, iters={args.iters}, jobs={args.jobs} ===",
          flush=True)
    print(f"    fixed spec: {' '.join(_FIXED)}  (cover_r=0 EXPLICIT; comm_r=5 every world)",
          flush=True)

    if args.dry_run:
        for u in units:
            g, n, r = u.rung
            print(f"  {u.uid:20s} grid={g} N={n} comm_r={r} extra={u.extra}", flush=True)
        print(f"\n[dry-run] {len(units)} runs planned; nothing launched.", flush=True)
        return

    t0 = time.time()
    done = _schedule(units, args.iters, args.rollouts, max(1, args.jobs))
    print(f"\n=== batch done: {len(done)}/{len(units)} produced model.eqx "
          f"({round(time.time() - t0, 1)}s) ===", flush=True)
    print(f"compare per-cell history.json under {out}/seed*/ "
          f"(coverage_pct / connectivity_real).", flush=True)


if __name__ == "__main__":
    main()
