"""Overnight OBSTACLE batch launcher — the full-factorial obstacle/exploration grid.

Launches, on the FIXED honest spec — **comm_r = 5 every world**, hard collision-mask on,
**soft/learned** connectivity (no guardrail), frontier-attn explorer, 100-step horizon — a
FULL FACTORIAL over four axes that probe how the SuperBlue arms cope as the world fills with
walls. Every run is graded on the metrics the trainer logs every iter: coverage, the REAL
connectivity bar (λ₂>0.5) alongside the trivial one, behavioural diversity (SND) and
role-distinctness. Unlike the skills ladder, EVERY run here is trained FRESH — no warm-start,
no inter-run deps (needs_dir=None, init_from=None for every unit).

FOUR axes, crossed in full (3 × 2 × 3 × 3 = 54 units per seed):

  ARM (3) — what coordinates the team:
    role  : the hand-coded explorer/relay split (--role-picker expl_relay).
    base  : the homogeneous shared explorer, no extra flags. The REFERENCE.
    sel   : the learned role-selector (--selector on --flock scripted --congestion off).

  BARRIER (2) — the connectivity barrier penalty weight:
    boff  : barrier off  (--barrier-weight 0).
    bon   : barrier on   (--barrier-weight 5).

  EXPLORE (3) — the exploration drive:
    eoff  : no extra exploration shaping.
    ebump : a heavier coverage reward     (--w-coverage 3).
    einfo : an explicit info-gain bonus   (--explore-infogain on).

  WORLD (3) — grid size / agent count / terrain (comm_r FIXED 5 everywhere):
    o16   : 16×16, 4 agents,  open terrain                 (--terrain open).
    r24   : 24×24, 6 agents,  3 rooms                       (--terrain rooms --rooms 3).
    r32   : 32×32, 10 agents, 4 rooms                       (--terrain rooms --rooms 4).

Per seed: 3 arms × 2 barrier × 3 explore × 3 worlds = 54 units. With ``--seeds 1`` (the
DEFAULT) that is 54 runs; ``--seeds 2`` is 108. Each unit's run-dir is
``<out>/seed{S}/{world}/{arm}_{barrier}_{explore}`` (e.g. ``runs/obstacle/seed0/r24/sel_bon_einfo``).

DEPENDENCY-AWARE + RESUMABLE (but every unit here is independent): each run is a unit whose
run-dir is skipped once it holds a ``model.eqx``; since nothing warm-starts anything, all units
are immediately ready. ``--jobs N`` runs up to N train_ctde subprocesses at once — balthar is
MEMORY-BOUND (~3 concurrent @32²/10, ~6 @24²/6), so keep N small if the r32 world is in play.
``--dry-run`` prints the full plan (every unit: id, grid/N, all extra flags) and launches nothing.

    # see the whole plan first (no training):
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python \
        ctde_v0/run_obstacle_overnight.py --dry-run --seeds 1

    # the real overnight batch on balthar (drop JAX_PLATFORMS so it uses the GPU), 2 at a time:
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_obstacle_overnight.py --out runs/obstacle --seeds 1 --iters 1500 \
        --rollouts 16 --jobs 2

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

# The honest fixed spec — IDENTICAL on every run (the comm_r lesson: the agent spec never
# drifts across worlds). frontier-attn explorer + hard collision-mask + soft/learned conn.
_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--mechanism", "soft_lambda", "--horizon", "100"]

# --- the four factorial axes (id -> the extra train_ctde flags that DEFINE that level) ---

# ARM: what coordinates the team (base = no extra; role = hand-coded split; sel = learned selector).
_ARMS = {
    "role": ["--role-picker", "expl_relay"],
    "base": [],
    "sel":  ["--selector", "on", "--flock", "scripted", "--congestion", "off"],
}

# BARRIER: the connectivity-barrier penalty weight (off vs on).
_BARRIERS = {
    "boff": ["--barrier-weight", "0"],
    "bon":  ["--barrier-weight", "5"],
}

# EXPLORE: the exploration drive (none vs heavier coverage reward vs explicit info-gain bonus).
_EXPLORES = {
    "eoff":  [],
    "ebump": ["--w-coverage", "3"],
    "einfo": ["--explore-infogain", "on"],
}

# WORLD: grid / n_agents / terrain flags; comm_r FIXED at 5 everywhere.
#   id -> (grid, n_agents, comm_r, terrain_extra_flags)
_WORLDS = {
    "o16": (16, 4, 5, ["--terrain", "open"]),
    "r24": (24, 6, 5, ["--terrain", "rooms", "--rooms", "3"]),
    "r32": (32, 10, 5, ["--terrain", "rooms", "--rooms", "4"]),
}

# fixed cross order (arm × barrier × explore × world) — every combination, the whole point.
_ARM_IDS = ["role", "base", "sel"]
_BARRIER_IDS = ["boff", "bon"]
_EXPLORE_IDS = ["eoff", "ebump", "einfo"]
_WORLD_IDS = ["o16", "r24", "r32"]


def _tag(g, n, r):
    return f"{g}x{g}x{n}"


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
    """The full factorial of runs across seeds: ARM × BARRIER × EXPLORE × WORLD, every unit
    FRESH (no warm-start, no deps). Run-dir <out>/seed{S}/{world}/{arm}_{barrier}_{explore}."""
    units = []
    for s in seeds:
        for wid in _WORLD_IDS:
            g, n, r, world_extra = _WORLDS[wid]
            for aid in _ARM_IDS:
                for bid in _BARRIER_IDS:
                    for eid in _EXPLORE_IDS:
                        extra = _ARMS[aid] + _BARRIERS[bid] + _EXPLORES[eid] + world_extra
                        combo = f"{aid}_{bid}_{eid}"
                        run_dir = os.path.join(out, f"seed{s}", wid, combo)
                        uid = f"s{s}/{wid}/{combo}"
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


def main(argv=None):
    p = argparse.ArgumentParser(
        description="overnight obstacle batch — full factorial ARM × BARRIER × EXPLORE × WORLD "
                    "(3×2×3×3 = 54 fresh runs per seed)")
    p.add_argument("--out", type=str, default="runs/obstacle",
                   help="parent dir for per-seed/per-world run-dirs (rel to SharedExploration)")
    p.add_argument("--seeds", type=int, default=1, help="number of seeds (0..S-1)")
    p.add_argument("--iters", type=int, default=1500, help="PPO iters per run")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per iter")
    p.add_argument("--jobs", type=int, default=1,
                   help="max concurrent train_ctde subprocesses (balthar is memory-bound: "
                        "~3 @32/10, ~6 @24/6)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    args = p.parse_args(argv)

    out = args.out if os.path.isabs(args.out) else os.path.join(_PKG_PARENT, args.out)
    seeds = list(range(args.seeds))
    units = _build_units(out, seeds)

    per_seed = len(_ARM_IDS) * len(_BARRIER_IDS) * len(_EXPLORE_IDS) * len(_WORLD_IDS)
    print(f"=== overnight obstacle batch: {len(units)} runs "
          f"(full factorial {len(_ARM_IDS)} arms × {len(_BARRIER_IDS)} barrier × "
          f"{len(_EXPLORE_IDS)} explore × {len(_WORLD_IDS)} worlds = {per_seed}/seed; "
          f"× {len(seeds)} seeds), out={out}, iters={args.iters}, jobs={args.jobs} ===",
          flush=True)
    print(f"    fixed spec: {' '.join(_FIXED)}  (comm_r=5 every world)", flush=True)

    if args.dry_run:
        for u in units:
            g, n, r = u.rung
            print(f"  {u.uid:24s} grid={g} N={n} extra={u.extra}", flush=True)
        print(f"\n[dry-run] {len(units)} runs planned; nothing launched.", flush=True)
        return

    t0 = time.time()
    done = _schedule(units, args.iters, args.rollouts, max(1, args.jobs))
    print(f"\n=== batch done: {len(done)}/{len(units)} produced model.eqx "
          f"({round(time.time() - t0, 1)}s) ===", flush=True)
    print(f"compare per-cell history.json under {out}/seed*/ "
          f"(coverage_pct / connectivity_real / snd / role_div).", flush=True)


if __name__ == "__main__":
    main()
