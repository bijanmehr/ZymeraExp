"""Overnight A/B batch launcher — the SuperBlue connectivity-coverage arms.

Launches, on the FIXED honest spec — **comm_r = 5 every rung**, hard collision-mask on,
**soft/learned** connectivity (no guardrail), frontier-attn explorer, 100-step horizon —
the four arms across the density-pinned ladder, ``--seeds`` seeds each. Everything is
graded on the metrics the trainer now logs every iter: coverage, the REAL connectivity bar
(λ₂>0.5) alongside the trivial one, behavioural diversity (SND) and role-distinctness.

  base   : the shared frontier-attn explorer, warm-start ladder 16->24->32. The REFERENCE,
           and the single-Actor BOOTSTRAP the fork replicates from.
  armA   : curriculum + randomness — the same ladder with --warmstart-noise at each rung —
           then a DTE tail (--critic-mode decentral) warm-started from the top rung.
  bfork  : warm-start from base's single-Actor ckpt AT EACH RUNG, replicate into 2 groups
           (--fork-groups 2); explorer/relay specialization emerges from per-group gradients.
  bdico  : the shared policy + a per-agent mean-zero diversity residual (--diversity-residual
           on), warm-start ladder — controlled diversity without forking.

DEPENDENCY-AWARE + RESUMABLE: each run is a unit whose run-dir is skipped once it holds a
``model.eqx``; a unit launches only once its ``needs`` predecessor's ckpt exists (base-rung
-> bfork-rung; rung i-1 -> rung i within a ladder; armA-top -> armA-DTE). ``--jobs N`` runs
up to N train_ctde subprocesses at once — balthar is MEMORY-BOUND (~3 concurrent @32²/10,
~6 @24²/6), so keep N small at the top rung. ``--dry-run`` prints the full plan + deps and
launches nothing.

    # see the whole plan first (no training):
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python \
        ctde_v0/run_ab_overnight.py --dry-run

    # the real overnight batch on balthar (drop JAX_PLATFORMS so it uses the GPU), 2 at a time:
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        ctde_v0/run_ab_overnight.py --out runs/ab --seeds 3 --iters 1500 --rollouts 16 --jobs 2

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


def _build_units(out, rungs, seeds, noise):
    """The full DAG of runs across seeds. Within a seed: base/armA/bdico are warm-start
    LADDERS (rung i needs rung i-1); bfork warm-starts base AT EACH RUNG (needs base-rung);
    armA-DTE warm-starts armA's top rung with --critic-mode decentral."""
    units = []
    for s in seeds:
        def rd(arm, g, n, r):
            return os.path.join(out, f"seed{s}", arm, _tag(g, n, r))

        prev = {"base": None, "armA": None, "bdico": None}
        for i, (g, n, r) in enumerate(rungs):
            # base: plain shared frontier-attn ladder (the reference + fork bootstrap).
            units.append(Unit(f"s{s}/base/{_tag(g,n,r)}", rd("base", g, n, r), (g, n, r),
                              s, [], prev["base"], prev["base"]))
            # armA: ladder + symmetry-break noise at each warm-started rung.
            a_extra = [] if i == 0 else ["--warmstart-noise", str(noise)]
            units.append(Unit(f"s{s}/armA/{_tag(g,n,r)}", rd("armA", g, n, r), (g, n, r),
                              s, a_extra, prev["armA"], prev["armA"]))
            # bdico: ladder with the per-agent diversity residual on.
            units.append(Unit(f"s{s}/bdico/{_tag(g,n,r)}", rd("bdico", g, n, r), (g, n, r),
                              s, ["--diversity-residual", "on"], prev["bdico"], prev["bdico"]))
            # bfork: warm-start the BASE ckpt at THIS rung, replicate into 2 groups.
            units.append(Unit(f"s{s}/bfork/{_tag(g,n,r)}", rd("bfork", g, n, r), (g, n, r),
                              s, ["--fork-groups", "2"], rd("base", g, n, r),
                              rd("base", g, n, r)))
            prev = {"base": rd("base", g, n, r), "armA": rd("armA", g, n, r),
                    "bdico": rd("bdico", g, n, r)}

        # armA DTE tail: warm-start armA's TOP rung, switch the critic decentralized.
        g, n, r = rungs[-1]
        top = rd("armA", g, n, r)
        units.append(Unit(f"s{s}/armA-DTE/{_tag(g,n,r)}",
                          os.path.join(out, f"seed{s}", "armA_DTE", _tag(g, n, r)),
                          (g, n, r), s, ["--critic-mode", "decentral",
                                         "--warmstart-noise", str(noise)], top, top))
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
    p = argparse.ArgumentParser(description="overnight A/B batch (base/armA/bfork/bdico)")
    p.add_argument("--out", type=str, default="runs/ab",
                   help="parent dir for per-seed/per-arm run-dirs (rel to SharedExploration)")
    p.add_argument("--rungs", type=str, default=None,
                   help="ladder 'GxNxR,...' (default 16x4x5,24x6x5,32x10x5; comm_r FIXED 5)")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..S-1)")
    p.add_argument("--iters", type=int, default=1500, help="PPO iters per run")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per iter")
    p.add_argument("--noise", type=float, default=0.05, help="armA warm-start σ")
    p.add_argument("--jobs", type=int, default=1,
                   help="max concurrent train_ctde subprocesses (balthar is memory-bound: "
                        "~3 @32/10, ~6 @24/6)")
    p.add_argument("--dry-run", action="store_true", help="print the plan + deps, launch nothing")
    args = p.parse_args(argv)

    rungs = _DEFAULT_RUNGS if not args.rungs else [
        tuple(int(x) for x in cell.split("x")) for cell in args.rungs.split(",") if cell.strip()]
    out = args.out if os.path.isabs(args.out) else os.path.join(_PKG_PARENT, args.out)
    seeds = list(range(args.seeds))
    units = _build_units(out, rungs, seeds, args.noise)

    print(f"=== overnight A/B batch: {len(units)} runs "
          f"(arms base/armA/bfork/bdico + armA-DTE × {len(seeds)} seeds × "
          f"{len(rungs)} rungs), out={out}, iters={args.iters}, jobs={args.jobs} ===",
          flush=True)
    print(f"    fixed spec: {' '.join(_FIXED)}  (comm_r=5 every rung)", flush=True)

    if args.dry_run:
        for u in units:
            dep = f"  needs<-{os.path.relpath(u.needs_dir, out)}" if u.needs_dir else ""
            ws = f"  init<-{os.path.relpath(u.init_from_dir, out)}" if u.init_from_dir else ""
            print(f"  {u.uid:28s} rung={_tag(*u.rung)} extra={u.extra}{ws}{dep}", flush=True)
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
