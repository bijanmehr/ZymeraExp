"""Scale-strategy experiment: single-point vs warm-start-ladder (vs multi-scale-joint).

Does CLIMBING the scale ladder with a warm start (train small -> transplant the
weights -> keep training bigger) reach the top rung better/cheaper than training the
top rung from scratch? The LPAC backbone is scale-invariant by construction (param
shapes depend on channels / width / depth / mp_rounds / goal-K / n_roles, NOT grid
size or agent count), so a model trained @16²/4 loads UNCHANGED into a @32²/10 run —
which is exactly what makes the ladder possible.

Three strategies (this launcher runs (a) + (b); (c) is documented, not built — see
MULTI-SCALE-JOINT below):

  (a) single-point     : train from scratch @32²/10 (the top rung only).
  (b) warm-start-ladder: train @16²/4 -> save -> --init-from train @24²/6 -> save ->
                         --init-from train @32²/10. Each rung warm-starts the POLICY
                         from the previous rung's saved (actor, critic); the optimizer
                         + dual are re-initialised fresh per rung (see
                         ppo.init_state_from_checkpoint).
  (c) multi-scale-joint: train on the rungs MIXED per-episode (each episode samples a
                         rung). NOT BUILT — see the limitation note below.

It just orchestrates `python -m ctde_v0.train_ctde` SUBPROCESS calls with the right
--grid/--n-agents/--comm-r/--run-dir/--ckpt/--init-from. Each rung writes its own
run-dir (config.json + history.json + model.eqx), so results are inspectable and the
ladder's hand-off (rung N's model.eqx -> rung N+1's --init-from) is explicit on disk.

RESUMABLE / shard-friendly (mirrors run_ctde_sweep.py): a rung whose run-dir already
holds a finished `model.eqx` is SKIPPED, so a re-launch never redoes a completed rung
(and the ladder's dependency is respected — a later rung waits for its predecessor's
model.eqx). `--only single|ladder` runs one strategy; `--shard i --nshards n` splits
the INDEPENDENT units of work across workers (the single-point run and the ladder are
two units; within the ladder the rungs are sequential by data dependency, so the whole
ladder is one unit).

    # one worker, both strategies, the default 16->24->32 ladder:
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -u \
        ctde_v0/run_scale_ladder.py --out runs/scale_ladder

    # just the warm-start ladder, custom rungs + iters:
    ... run_scale_ladder.py --only ladder --rungs 16x4x5,24x6x5,32x10x5 --iters 400

This experiment is CPU-only by construction (do NOT run on the GPU server); it only
SHELLS OUT to train_ctde — keep the per-rung --iters/--rollouts in the train_ctde
budget. ===>>> This launcher LAUNCHES REAL TRAINING; run it deliberately. <<<===

MULTI-SCALE-JOINT (c) — known limitation, NOT built here.
    The current trainer compiles ONE env at a fixed (grid, n_agents, comm_r): the grid
    H×W and the agent count N are static array dims baked into the jitted rollout (the
    env is a trace-time constant; see env_utils.build_env). Mixing rungs PER EPISODE
    would need either (i) one compiled rollout per rung shape (different N / H×W can't
    share a scan, so they cannot be vmapped together), or (ii) a single padded/masked
    MAX-shape env (pad to the largest grid + agent count and mask the slack) so every
    episode shares one shape and a per-episode key selects the active sub-rung. Both are
    a trainer change, out of scope for this launcher; the scale-invariant backbone makes
    either viable as a follow-up. Until then, the ladder (b) is the supported way to
    train one model across multiple scales.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(_HERE)              # .../SharedExploration
# the Fiedler dir train_ctde reaches for; harmless to also pin it on PYTHONPATH.
_FIEDLER = os.path.abspath(os.path.join(_HERE, "..", "..", "..", "FiedlerValueEstimation"))

# default ladder rungs as (grid, n_agents, comm_r) — the 16->24->32 climb.
_DEFAULT_RUNGS = [(16, 4, 5), (24, 6, 5), (32, 10, 5)]


def _parse_rungs(spec: str | None) -> list[tuple[int, int, int]]:
    """`"16x4x5,24x6x5,32x10x5"` -> [(16,4,5),(24,6,5),(32,10,5)]. None -> default."""
    if not spec:
        return list(_DEFAULT_RUNGS)
    rungs = []
    for cell in spec.split(","):
        cell = cell.strip()
        if not cell:
            continue
        g, n, r = (int(x) for x in cell.split("x"))
        rungs.append((g, n, r))
    if not rungs:
        raise ValueError(f"no rungs parsed from {spec!r}")
    return rungs


def _rung_tag(g: int, n: int, r: int) -> str:
    return f"{g}x{g}x{n}"


def _done(run_dir: str) -> bool:
    """A rung is finished iff its run-dir holds a non-empty model.eqx (resume key)."""
    p = os.path.join(run_dir, "model.eqx")
    return os.path.exists(p) and os.path.getsize(p) > 0


def _train_cmd(*, grid: int, n_agents: int, comm_r: int, run_dir: str,
               iters: int, rollouts: int, seed: int, extra: list[str],
               init_from: str | None) -> list[str]:
    cmd = [
        sys.executable, "-u", "-m", "ctde_v0.train_ctde",
        "--grid", str(grid), "--n-agents", str(n_agents), "--comm-r", str(comm_r),
        "--iters", str(iters), "--rollouts", str(rollouts), "--seed", str(seed),
        "--run-dir", run_dir, "--ckpt",
    ]
    if init_from is not None:
        cmd += ["--init-from", init_from]
    cmd += extra
    return cmd


def _run(cmd: list[str]) -> int:
    """Run a train_ctde subprocess from SharedExploration with PYTHONPATH wired."""
    env = dict(os.environ)
    env.setdefault("JAX_PLATFORMS", "cpu")
    pp = [_PKG_PARENT, _FIEDLER]
    if env.get("PYTHONPATH"):
        pp.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(pp)
    print(f"\n$ {' '.join(cmd)}", flush=True)
    return subprocess.run(cmd, cwd=_PKG_PARENT, env=env).returncode


def run_single_point(out: str, top: tuple[int, int, int], *, iters: int,
                     rollouts: int, seed: int, extra: list[str]) -> dict:
    """(a) train the TOP rung from scratch (no --init-from)."""
    g, n, r = top
    run_dir = os.path.join(out, f"single_{_rung_tag(g, n, r)}")
    rec = {"strategy": "single-point", "rung": _rung_tag(g, n, r), "run_dir": run_dir}
    if _done(run_dir):
        print(f"[single-point {_rung_tag(g, n, r)}] model.eqx present -- skip",
              flush=True)
        rec["status"] = "skipped"
        return rec
    rc = _run(_train_cmd(grid=g, n_agents=n, comm_r=r, run_dir=run_dir, iters=iters,
                         rollouts=rollouts, seed=seed, extra=extra, init_from=None))
    rec["status"] = "ok" if rc == 0 else f"rc={rc}"
    return rec


def run_ladder(out: str, rungs: list[tuple[int, int, int]], *, iters: int,
               rollouts: int, seed: int, extra: list[str]) -> dict:
    """(b) warm-start ladder: rung[0] from scratch, each later rung --init-from the
    previous rung's model.eqx. Sequential by data dependency; resumable per rung."""
    rec: dict = {"strategy": "warm-start-ladder",
                 "rungs": [_rung_tag(*x) for x in rungs], "steps": []}
    prev_ckpt: str | None = None
    for i, (g, n, r) in enumerate(rungs):
        run_dir = os.path.join(out, f"ladder_{i}_{_rung_tag(g, n, r)}")
        ckpt = os.path.join(run_dir, "model.eqx")
        step = {"rung": _rung_tag(g, n, r), "run_dir": run_dir,
                "init_from": prev_ckpt}
        if _done(run_dir):
            print(f"[ladder rung {i} {_rung_tag(g, n, r)}] model.eqx present -- skip",
                  flush=True)
            step["status"] = "skipped"
        else:
            # a later rung needs its predecessor's ckpt; bail clearly if absent.
            if prev_ckpt is not None and not (os.path.exists(prev_ckpt)
                                              and os.path.getsize(prev_ckpt) > 0):
                step["status"] = "blocked: predecessor model.eqx missing"
                rec["steps"].append(step)
                print(f"[ladder rung {i}] BLOCKED: {prev_ckpt} missing "
                      f"(run the earlier rung first)", flush=True)
                break
            rc = _run(_train_cmd(grid=g, n_agents=n, comm_r=r, run_dir=run_dir,
                                 iters=iters, rollouts=rollouts, seed=seed,
                                 extra=extra, init_from=prev_ckpt))
            step["status"] = "ok" if rc == 0 else f"rc={rc}"
            if rc != 0:
                rec["steps"].append(step)
                break
        rec["steps"].append(step)
        prev_ckpt = ckpt
    return rec


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description="scale-strategy: single-point vs warm-start-ladder")
    p.add_argument("--out", type=str, default="runs/scale_ladder",
                   help="parent dir for the per-rung run-dirs (relative to SharedExploration)")
    p.add_argument("--rungs", type=str, default=None,
                   help="ladder rungs 'GxNxR,...' (default 16x4x5,24x6x5,32x10x5); the "
                        "LAST rung is also the single-point top rung")
    p.add_argument("--only", choices=["single", "ladder"], default=None,
                   help="run just one strategy (default: both)")
    p.add_argument("--iters", type=int, default=400, help="PPO iters per rung")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per iter")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--shard", type=int, default=0,
                   help="this worker's index (splits the independent units of work)")
    p.add_argument("--nshards", type=int, default=1)
    # anything after `--` is forwarded verbatim to every train_ctde call (e.g.
    # `-- --role-picker expl_relay --mechanism action_mask`).
    args, extra = p.parse_known_args(argv)
    if extra and extra[0] == "--":
        extra = extra[1:]

    rungs = _parse_rungs(args.rungs)
    top = rungs[-1]
    out = args.out if os.path.isabs(args.out) else os.path.join(_PKG_PARENT, args.out)
    os.makedirs(out, exist_ok=True)

    # the independent units of work: 0 = single-point, 1 = the whole ladder.
    units = []
    if args.only in (None, "single"):
        units.append(("single", 0))
    if args.only in (None, "ladder"):
        units.append(("ladder", 1))
    mine = [u for u in units if u[1] % args.nshards == args.shard]

    print(f"=== scale-ladder: shard {args.shard}/{args.nshards} -> "
          f"{[u[0] for u in mine]}  (rungs {[_rung_tag(*x) for x in rungs]}, "
          f"iters={args.iters}, rollouts={args.rollouts}, out={out}) ===", flush=True)

    results = []
    for name, _idx in mine:
        t0 = time.time()
        if name == "single":
            rec = run_single_point(out, top, iters=args.iters, rollouts=args.rollouts,
                                    seed=args.seed, extra=extra)
        else:
            rec = run_ladder(out, rungs, iters=args.iters, rollouts=args.rollouts,
                             seed=args.seed, extra=extra)
        rec["wall_s"] = round(time.time() - t0, 1)
        results.append(rec)
        print(f"[{name}] -> {rec.get('status', rec.get('steps'))}  "
              f"({rec['wall_s']}s)", flush=True)

    print("\n=== scale-ladder summary ===", flush=True)
    for rec in results:
        print(f"  {rec['strategy']}: {rec}", flush=True)
    print(f"\nCompare the strategies' top-rung run-dirs under {out} "
          f"(single_{_rung_tag(*top)}/history.json vs "
          f"ladder_{len(rungs)-1}_{_rung_tag(*top)}/history.json).", flush=True)


if __name__ == "__main__":
    main()
