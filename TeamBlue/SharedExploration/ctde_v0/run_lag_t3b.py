"""T3b — connectivity-control test, crossed over critic × scale × terrain (the finalized T3 plan).

The question (same as T3): which dual-update controller actually HOLDS the global-λ₂ target θ
without collapsing coverage — soft (fixed weight) vs lagrangian (monotone dual) vs PID (responsive
dual)? λ₂ = true team algebraic connectivity, computed centrally by the trainer and folded into the
REWARD only (CTDE; agents never observe it; NEVER a hard action mask). We sweep θ so it BINDS.

WHAT'S DIFFERENT FROM run_lag_t3 (why this file):
  * CRITIC crossed: {setpool, setattn} — the two count-invariant critics T4 found scale-resilient
    (conv, the fixed-N baseline, degrades with N, so it's dropped here).
  * SCALE crossed: 24²/6 (→ gaius, light tier) AND 32²/10 (→ balthar, heavy tier).
  * TERRAIN: rooms = the CHALLENGING env where connectivity binds -> FULL 6-controller sweep;
    open = the free-connectivity BASELINE -> reduced to the lagrangian arms only (on open every
    controller trivially satisfies λ₂≥θ, so a full sweep there is wasted compute).
  * soft_th7 added -> a clean 3 controllers × 2 thresholds grid (the original T3 lacked soft@0.7).

PLAN (96 units): rooms(6 ctrl) + open(2 ctrl) = 8 controller-conditions × {setpool,setattn} × 3 seeds
  = 48 per scale × {24²,32²} = 96 total -> 48 heavy 32² (balthar) + 48 light 24² (gaius).

METRICS (in each run's history.json): dual_lambda_next = λ(t) (does the dual rise to enforce θ?),
  connectivity_real = λ₂(t) (does realised connectivity converge to θ?), coverage_pct. Verdict:
  which controller holds θ (esp. the hard θ=0.7) at least coverage cost, consistent across critic+scale.

    # see the whole plan, launch nothing:
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        -m ctde_v0.run_lag_t3b --tier heavy --dry-run
    # balthar (32²/10, one at a time -> jobs 1):
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        -m ctde_v0.run_lag_t3b --tier heavy --out runs_t3b --seeds 3 --iters 8000 --rollouts 16 --jobs 1
    # gaius  (24²/6, packed -> jobs 3):
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        -m ctde_v0.run_lag_t3b --tier light --out runs_t3b --seeds 3 --iters 8000 --rollouts 16 --jobs 3

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

# IDENTICAL on every unit. terrain + critic-arch + controller are PER-UNIT (added below). ppo-epochs 6
# matches the completed T2/T4 campaign. role-picker ON (relays = the connectivity-maintaining role).
_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on", "--w-coverage", "3",
          "--sense-walls", "--sense-free", "--boundary", "--horizon", "100", "--cover-r", "0",
          "--role-picker", "expl_relay", "--conn-signal", "global_lambda2", "--ppo-epochs", "6"]

# controller id -> the flags that DEFINE it (mechanism × threshold θ). 3 mechanisms × θ∈{0.5,0.7}.
_CTRL = {
    "soft_th5": ["--mechanism", "soft_lambda", "--constraint-threshold", "0.5"],
    "soft_th7": ["--mechanism", "soft_lambda", "--constraint-threshold", "0.7"],
    "lag_th5":  ["--mechanism", "lagrangian", "--constraint-threshold", "0.5", "--lambda-lr", "0.05"],
    "lag_th7":  ["--mechanism", "lagrangian", "--constraint-threshold", "0.7", "--lambda-lr", "0.05"],
    "pid_th5":  ["--mechanism", "pid_lagrangian", "--constraint-threshold", "0.5"],
    "pid_th7":  ["--mechanism", "pid_lagrangian", "--constraint-threshold", "0.7"],
}
_ROOMS_CTRL = ["soft_th5", "soft_th7", "lag_th5", "lag_th7", "pid_th5", "pid_th7"]   # full sweep
_OPEN_CTRL = ["lag_th5", "lag_th7"]                                                  # reduced baseline
_TERRAINS = {"rooms": _ROOMS_CTRL, "open": _OPEN_CTRL}
_CRITICS = ["setpool", "setattn"]
_SCALES = [(24, 6, 5), (32, 10, 5)]                        # (grid, n_agents, comm_r); 24->light, 32->heavy


class Unit:
    """One train_ctde run at a given rung. FRESH (no warm-start, no deps)."""
    def __init__(self, uid, run_dir, rung, seed, extra):
        self.uid = uid
        self.run_dir = run_dir
        self.rung = rung                     # (grid, n_agents, comm_r)
        self.seed = seed
        self.extra = extra
        self.proc = None

    def cmd(self, iters, rollouts):
        g, n, r = self.rung
        return [sys.executable, "-u", "-m", "ctde_v0.train_ctde",
                "--grid", str(g), "--n-agents", str(n), "--comm-r", str(r),
                "--iters", str(iters), "--rollouts", str(rollouts), "--seed", str(self.seed),
                "--run-dir", self.run_dir, "--ckpt"] + _FIXED + self.extra


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


def _build_units(out_base, seeds):
    """The full cross. Run-dir <out>/<terrain>/<critic>/<controller>/seed{S}/g{grid}."""
    units = []
    for terr, ctrls in _TERRAINS.items():
        for crit in _CRITICS:
            for (g, n, r) in _SCALES:
                for ctrl in ctrls:
                    for s in seeds:
                        rd = os.path.join(out_base, terr, crit, ctrl, f"seed{s}", f"g{g}")
                        extra = _CTRL[ctrl] + ["--critic-arch", crit, "--terrain", terr]
                        uid = f"{terr}/{crit}/{ctrl}/s{s}/g{g}"
                        units.append(Unit(uid, rd, (g, n, r), s, extra))
    return units


def _schedule(units, iters, rollouts, jobs, gpu_ids=None):
    """Bounded-parallel pool; skips done run-dirs (resumable); each unit its own subprocess.
    If gpu_ids is given, every job is PINNED to one free GPU via CUDA_VISIBLE_DEVICES (one unit
    per GPU) so parallel JAX processes never fight over the same card's default preallocation."""
    base_env = _run_env()
    done = []
    pending = []
    for u in units:
        if _done(u.run_dir):
            print(f"[skip] {u.uid} (model.eqx present)", flush=True)
            done.append(u)
        else:
            pending.append(u)
    free_gpus = list(gpu_ids) if gpu_ids else None      # None -> no pinning (single-GPU / jobs=1)
    running = []
    while pending or running:
        while pending and len(running) < jobs and (free_gpus is None or free_gpus):
            u = pending.pop(0)
            env = dict(base_env)
            u.gpu = None
            if free_gpus is not None:
                u.gpu = free_gpus.pop(0)
                env["CUDA_VISIBLE_DEVICES"] = str(u.gpu)
            os.makedirs(u.run_dir, exist_ok=True)
            print(f"[launch] {u.uid}  ({u.rung[0]}²/{u.rung[1]})"
                  + (f"  gpu{u.gpu}" if u.gpu is not None else ""), flush=True)
            u.proc = subprocess.Popen(u.cmd(iters, rollouts), cwd=_PKG_PARENT, env=env)
            running.append(u)
        time.sleep(3)
        for u in list(running):
            rc = u.proc.poll()
            if rc is None:
                continue
            running.remove(u)
            if free_gpus is not None and getattr(u, "gpu", None) is not None:
                free_gpus.append(u.gpu)            # return the card to the pool
            ok = _done(u.run_dir)
            print(f"[finish] {u.uid}  rc={rc}  {'ok' if ok else 'NO-CKPT'}", flush=True)
            if ok:
                done.append(u)
    return done


def main(argv=None):
    p = argparse.ArgumentParser(
        description="T3b connectivity-control test crossed over critic×scale×terrain "
                    "(soft/lag/pid × θ∈{0.5,0.7} × {setpool,setattn} × {24²,32²} × {rooms,open})")
    p.add_argument("--tier", choices=["heavy", "light"], required=True,
                   help="heavy = grid==32 units (balthar, jobs 1); light = grid<32 (gaius, jobs 3)")
    p.add_argument("--out", type=str, default="runs_t3b", help="run-dir root, rel to SharedExploration")
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--iters", type=int, default=8000)
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--jobs", type=int, default=None, help="concurrent runs (default heavy=1, light=3)")
    p.add_argument("--gpus", type=str, default=None,
                   help="comma-sep GPU ids to pin ONE unit per card (e.g. '0,1'); caps jobs at #gpus")
    p.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    a = p.parse_args(argv)

    out = a.out if os.path.isabs(a.out) else os.path.join(_PKG_PARENT, a.out)
    seeds = list(range(a.seeds))
    units = _build_units(out, seeds)

    is_heavy = (a.tier == "heavy")
    tier_units = [u for u in units if (u.rung[0] >= 32) == is_heavy]
    jobs = a.jobs if a.jobs is not None else (1 if is_heavy else 3)
    gpu_ids = [int(x) for x in a.gpus.split(",")] if a.gpus else None
    if gpu_ids:
        jobs = min(jobs, len(gpu_ids))                 # one 24²/6 unit per GPU -> no OOM

    n_heavy = sum(1 for u in units if u.rung[0] >= 32)
    print(f"=== T3b [{a.tier}]: {len(tier_units)} units (total {len(units)} = "
          f"{n_heavy} heavy 32² + {len(units)-n_heavy} light 24²), jobs={jobs}"
          f"{f' gpus={gpu_ids}' if gpu_ids else ''}, "
          f"iters={a.iters}, seeds={seeds}, out={a.out} ===", flush=True)
    print(f"    fixed: {' '.join(_FIXED)}", flush=True)

    if a.dry_run:
        for u in tier_units:
            print(f"  {u.uid:26s} {' '.join(_CTRL[u.uid.split('/')[2]])}", flush=True)
        print(f"\n[dry-run] {len(tier_units)} '{a.tier}' units; nothing launched.", flush=True)
        return

    t0 = time.time()
    done = _schedule(tier_units, a.iters, a.rollouts, jobs, gpu_ids)
    print(f"\n=== T3b {a.tier} DONE: {len(done)}/{len(tier_units)} produced model.eqx "
          f"({round(time.time()-t0, 1)}s) ===", flush=True)
    print("watch per-run history.json: dual_lambda_next (λ) + connectivity_real (λ₂) vs θ.", flush=True)


if __name__ == "__main__":
    main()
