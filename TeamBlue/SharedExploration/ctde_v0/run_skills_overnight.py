"""Overnight SKILLS+SELECTOR batch launcher — the SuperBlue role-selector arms.

Launches, on the FIXED honest spec — **comm_r = 5 every rung**, hard collision-mask on,
**soft/learned** connectivity (no guardrail), frontier-attn explorer, 100-step horizon —
the skill/selector arms across TWO experiment shapes, ``--seeds`` seeds each. Everything is
graded on the metrics the trainer logs every iter: coverage, the REAL connectivity bar
(λ₂>0.5) alongside the trivial one, behavioural diversity (SND) and role-distinctness.

  base         : the homogeneous shared explorer, no extra flags. The REFERENCE, and the
                 single-Actor BOOTSTRAP the ladder warm-starts from at each rung.
  role         : the hardcoded explorer/relay baseline (--role-picker expl_relay) — the
                 "roles by hand" control the learned selector has to beat.
  sel_scr_off  : selector on, scripted flock, congestion off  (--selector on
                 --flock scripted --congestion off).
  sel_scr_on   : selector on, scripted flock, congestion on.
  sel_lrn_off  : selector on, learned flock, congestion off.
  sel_lrn_on   : selector on, learned flock, congestion on.

The 2×2 selector sweep crosses {scripted, learned} flock × {off, on} congestion, asking
whether a LEARNED role-selector (and a learned vs scripted flock, with/without a congestion
signal) divides labour better than the hand-coded role split.

TWO experiment shapes, BOTH at horizon 100 / comm_r 5:
  1. density-pinned LADDER  16x4 -> 24x6 -> 32x10: for base/role + the 4 selector arms, rung i
     warm-starts rung i-1 (--init-from prev model.eqx + a `needs` dep on it). Run-dirs
     <out>/seed{S}/ladder/{arm}/{g}x{g}x{n}.
  2. fixed-world N-SWEEP  grid 24, comm_r 5, N in {4,6,9,12}: for base + the 4 selector arms
     (role SKIPPED here), each trained FRESH (no warm-start, no deps) — "do more agents divide
     labour or just flood?" at a fixed world + fixed 100-step budget. Run-dirs
     <out>/seed{S}/nsweep/{arm}/24x24x{n}.

DEPENDENCY-AWARE + RESUMABLE: each run is a unit whose run-dir is skipped once it holds a
``model.eqx``; a ladder unit launches only once its ``needs`` predecessor's ckpt exists (rung
i-1 -> rung i within an arm's ladder); N-sweep units are fresh and depend on nothing.
``--jobs N`` runs up to N train_ctde subprocesses at once — balthar is MEMORY-BOUND (~3
concurrent @32²/10, ~6 @24²/6), so keep N small at the top rung. ``--dry-run`` prints the
full plan + deps and launches nothing.

    # see the whole plan first (no training):
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python \
        ctde_v0/run_skills_overnight.py --dry-run

    # the real overnight batch on balthar (drop JAX_PLATFORMS so it uses the GPU), 2 at a time:
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_skills_overnight.py --out runs/skills --seeds 3 --iters 1500 \
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
# drifts across rungs). frontier-attn explorer + hard collision-mask + soft/learned conn.
_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--mechanism", "soft_lambda", "--horizon", "100"]

# density-pinned ladder rungs (grid, n_agents, comm_r); comm_r FIXED at 5 everywhere.
_DEFAULT_RUNGS = [(16, 4, 5), (24, 6, 5), (32, 10, 5)]

# fixed-world N-sweep: grid 24 / comm_r 5, vary the agent count only.
_NSWEEP_GRID = 24
_NSWEEP_R = 5
_NSWEEP_NS = [4, 6, 9, 12]

# The arms: id -> the extra train_ctde flags that DEFINE the arm (on top of _FIXED).
#   base       : homogeneous reference, no extra flags (also the ladder warm-start bootstrap).
#   role       : hand-coded explorer/relay split.
#   sel_*      : selector on; cross {scripted,learned} flock × {off,on} congestion.
_ARMS = {
    "base":        [],
    "role":        ["--role-picker", "expl_relay"],
    "sel_scr_off": ["--selector", "on", "--flock", "scripted", "--congestion", "off"],
    "sel_scr_on":  ["--selector", "on", "--flock", "scripted", "--congestion", "on"],
    "sel_lrn_off": ["--selector", "on", "--flock", "learned",  "--congestion", "off"],
    "sel_lrn_on":  ["--selector", "on", "--flock", "learned",  "--congestion", "on"],
}
# the ladder runs base + role + the 4 selector arms; the N-sweep drops role (it is a per-rung
# warm-start control, not a fixed-world-density probe).
_LADDER_ARMS = ["base", "role", "sel_scr_off", "sel_scr_on", "sel_lrn_off", "sel_lrn_on"]
_NSWEEP_ARMS = ["base", "sel_scr_off", "sel_scr_on", "sel_lrn_off", "sel_lrn_on"]


def _tag(g, n, r):
    return f"{g}x{g}x{n}"


def _done(run_dir):
    p = os.path.join(run_dir, "model.eqx")
    return os.path.exists(p) and os.path.getsize(p) > 0


class Unit:
    """One train_ctde run: an id, its run-dir, the extra flags, the seed/rung, and the
    run-dir of a predecessor whose model.eqx must exist before it can start (or None)."""
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


def _build_units(out, rungs, seeds):
    """The full DAG of runs across seeds. Two shapes per seed:
      ladder  : for each ladder arm, a warm-start ladder (rung i needs + inits-from rung i-1);
      nsweep  : for each n-sweep arm, one FRESH run per N at fixed grid 24 / comm_r 5 (no dep).
    """
    units = []
    for s in seeds:
        def rd(shape, arm, g, n, r):
            return os.path.join(out, f"seed{s}", shape, arm, _tag(g, n, r))

        # shape 1 — density-pinned warm-start ladder.
        for arm in _LADDER_ARMS:
            prev = None
            for (g, n, r) in rungs:
                run_dir = rd("ladder", arm, g, n, r)
                units.append(Unit(f"s{s}/ladder/{arm}/{_tag(g,n,r)}", run_dir, (g, n, r),
                                  s, _ARMS[arm], prev, prev))
                prev = run_dir

        # shape 2 — fixed-world N-sweep; every run fresh (no warm-start, no deps).
        for arm in _NSWEEP_ARMS:
            for n in _NSWEEP_NS:
                g, r = _NSWEEP_GRID, _NSWEEP_R
                units.append(Unit(f"s{s}/nsweep/{arm}/{_tag(g,n,r)}",
                                  rd("nsweep", arm, g, n, r), (g, n, r),
                                  s, _ARMS[arm], None, None))
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
    predecessor ckpt exists, up to ``jobs`` at once; poll; stop when all are done/blocked."""
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
        description="overnight skills+selector batch (base/role + 2×2 selector sweep, "
                    "density ladder + fixed-world N-sweep)")
    p.add_argument("--out", type=str, default="runs/skills",
                   help="parent dir for per-seed/per-shape/per-arm run-dirs (rel to "
                        "SharedExploration)")
    p.add_argument("--rungs", type=str, default=None,
                   help="ladder 'GxNxR,...' (default 16x4x5,24x6x5,32x10x5; comm_r FIXED 5)")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..S-1)")
    p.add_argument("--iters", type=int, default=1500, help="PPO iters per run")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per iter")
    p.add_argument("--jobs", type=int, default=1,
                   help="max concurrent train_ctde subprocesses (balthar is memory-bound: "
                        "~3 @32/10, ~6 @24/6)")
    p.add_argument("--dry-run", action="store_true", help="print the plan + deps, launch nothing")
    args = p.parse_args(argv)

    rungs = _DEFAULT_RUNGS if not args.rungs else [
        tuple(int(x) for x in cell.split("x")) for cell in args.rungs.split(",") if cell.strip()]
    out = args.out if os.path.isabs(args.out) else os.path.join(_PKG_PARENT, args.out)
    seeds = list(range(args.seeds))
    units = _build_units(out, rungs, seeds)

    n_ladder = len(seeds) * len(_LADDER_ARMS) * len(rungs)
    n_nsweep = len(seeds) * len(_NSWEEP_ARMS) * len(_NSWEEP_NS)
    print(f"=== overnight skills batch: {len(units)} runs "
          f"(ladder {len(_LADDER_ARMS)} arms × {len(rungs)} rungs = {n_ladder}; "
          f"nsweep {len(_NSWEEP_ARMS)} arms × {len(_NSWEEP_NS)} N = {n_nsweep}; "
          f"× {len(seeds)} seeds), out={out}, iters={args.iters}, jobs={args.jobs} ===",
          flush=True)
    print(f"    fixed spec: {' '.join(_FIXED)}  (comm_r=5 every rung)", flush=True)

    if args.dry_run:
        for u in units:
            dep = f"  needs<-{os.path.relpath(u.needs_dir, out)}" if u.needs_dir else ""
            ws = f"  init<-{os.path.relpath(u.init_from_dir, out)}" if u.init_from_dir else ""
            print(f"  {u.uid:34s} rung={_tag(*u.rung)} extra={u.extra}{ws}{dep}", flush=True)
        print(f"\n[dry-run] {len(units)} runs planned; nothing launched.", flush=True)
        return

    t0 = time.time()
    done = _schedule(units, args.iters, args.rollouts, max(1, args.jobs))
    print(f"\n=== batch done: {len(done)}/{len(units)} produced model.eqx "
          f"({round(time.time() - t0, 1)}s) ===", flush=True)
    print(f"compare per-arm history.json under {out}/seed*/ "
          f"(coverage_pct / connectivity_real / snd / role_div).", flush=True)


if __name__ == "__main__":
    main()
