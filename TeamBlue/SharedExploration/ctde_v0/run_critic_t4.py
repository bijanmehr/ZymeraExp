"""T4 — count-invariant critic A/B, a FROM-SCRATCH + ZERO-SHOT-TRANSFER test.

Belief: the fixed spec now carries the FULL 7-channel SLAM occupancy KB as the baseline belief
(``--sense-walls --sense-free --boundary``: free/occupied/unknown + occ_frontier + boundary),
so every arm/rung reads occupancy rather than the old 5-channel coverage-trail belief.

Does a FIXED-N conv critic degrade where a count-invariant DeepSets critic holds, when
the SAME team is scaled up? The conv `Critic` reads the global central_obs (grid-invariant
via same-padding conv + global-average-pool), so it LOADS across rungs — but its value
predictions are learned against a fixed agent count; the DeepSets critics pool per-agent
encodings permutation- and count-invariantly, so they *should* transfer better as N grows.

Three ARMS, identical in everything but the critic architecture (the exact Wire-phase flag):
    conv    (control)  --critic-arch conv     conv over the global central_obs (v0 baseline)
    setpool            --critic-arch setpool  DeepSets, mean agent-pool
    setattn            --critic-arch setattn  DeepSets, attention agent-pool

TWO phases:

  (1) TRAIN — every scale FROM SCRATCH (no warm-start): 16²/4, 24²/6, 32²/10 each trained
      independently per arm, per seed. 3 arms × 3 seeds × 3 scales = 27 independent trainings.
      The 16² run is the transfer eval's SOURCE; the 24²/32² runs are the in-distribution
      REFERENCE the zero-shot transfer numbers are read against (trained-at-that-scale ceiling).
      The --critic-arch flag is forwarded to every run so the checkpoint arch always matches.

      Uses the SAME bounded-parallel scheduler as the obstacle launcher
      (run_obstacle_overnight._schedule): a run with a model.eqx is skipped; every unit is
      independent (no --init-from deps); --jobs runs up to N at once.

  (2) ZERO-SHOT TRANSFER EVAL — the transfer test proper. For each arm × seed, take the
      16²-TRAINED checkpoint (ladder rung 0) and evaluate it, WITHOUT any further training,
      at 16²/4 (in-distribution anchor), 24²/6 and 32²/10. Two metrics per target scale:
        - value-prediction error: the critic's v_team vs the Monte-Carlo discounted
          return-to-go (bootstrap-free ground truth). Reported as RMSE and as NRMSE
          (RMSE / std(returns)) — the NRMSE is the cross-rung-comparable number, since the
          raw return magnitude differs between 16²/4 and 32²/10.
        - downstream coverage: the final-step coverage the transplanted (actor, critic)
          achieves at the target scale (does the policy still sweep, zero-shot).
      The hypothesis: conv's NRMSE blows up at 24²/6 and 32²/10 while setpool/setattn hold.

This launcher SHELLS OUT to `python -m ctde_v0.train_ctde` for training and runs the eval
IN-PROCESS (loads each checkpoint via ppo.init_state_from_checkpoint + a fresh rollout).
CPU-friendly; keep --iters / --rollouts in the train_ctde budget.

    # see the whole plan (train + eval), launch nothing:
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m ctde_v0.run_critic_t4 \
        --dry-run

    # the real thing (drop JAX_PLATFORMS on balthar for GPU), 2 trainings at a time:
    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python \
        -m ctde_v0.run_critic_t4 --out runs/critic_t4 --seeds 3 --iters 2000 --jobs 2

===>>> This launcher LAUNCHES REAL TRAINING; run it deliberately. <<<===
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

# Reuse the obstacle launcher's dependency-aware scheduler + path wiring (NOT its _FIXED
# spec — our Unit bakes T4's fixed spec into its own cmd()).
from ctde_v0.run_obstacle_overnight import _PKG_PARENT, _done, _schedule

# --- T4 honest shared spec: IDENTICAL on every training run and every eval build ---------
# frontier-attn explorer · hard collision-mask · learned-Lagrangian SOFT connectivity on the
# local edge-margin signal · hand-coded explorer/relay split · coverage-bump ·
# 100-step horizon · cover_r=0 (real visited coverage). The belief is now the FULL 7-channel
# SLAM occupancy KB: --sense-walls + --sense-free + --boundary give free/occupied/unknown
# occupancy + a Yamauchi occ_frontier + the field-edge boundary (5→7 ch). The ONLY thing an
# arm changes is --critic-arch.
_FIXED = ["--explorer-tool", "frontier_attn", "--collision-mask", "on",
          "--mechanism", "lagrangian", "--conn-signal", "global_lambda2",
          "--constraint-threshold", "0.5",
          "--role-picker", "expl_relay", "--w-coverage", "3",
          "--sense-walls", "--sense-free", "--boundary",
          "--horizon", "100", "--cover-r", "0"]

# the 3 arms — the count-invariant critic A/B. Value = the exact Wire-phase --critic-arch flag.
_ARMS = {
    "conv":    ["--critic-arch", "conv"],       # control: fixed-N conv over central_obs
    "setpool": ["--critic-arch", "setpool"],    # DeepSets-mean pooling (count-invariant)
    "setattn": ["--critic-arch", "setattn"],    # DeepSets-attention pooling (count-invariant)
}
_ARM_IDS = ["conv", "setpool", "setattn"]

# the warm-start ladder rungs as (grid, n_agents, comm_r); rung 0 (16²/4) is the transfer
# eval's source checkpoint, the last rung (32²/10) the top of the climb.
_RUNGS = [(16, 4, 5), (24, 6, 5), (32, 10, 5)]
# zero-shot transfer eval target scales (rung 0 = in-distribution anchor; 24²/6 & 32²/10 = the
# transfer test — does the fixed-N conv critic degrade where DeepSets holds).
_EVAL_TARGETS = [(16, 4, 5), (24, 6, 5), (32, 10, 5)]


def _tag(g: int, n: int, r: int) -> str:
    return f"{g}x{g}x{n}"


class Unit:
    """One warm-start-ladder rung (a train_ctde subprocess). Mirrors the scheduler's Unit
    interface (run_dir/uid/ready()/cmd()/proc/init_from) but bakes T4's _FIXED + the arm's
    --critic-arch into cmd() so the obstacle launcher's _FIXED never leaks in."""

    def __init__(self, uid, run_dir, rung, seed, extra, init_from_dir, needs_dir):
        self.uid = uid
        self.run_dir = run_dir
        self.rung = rung                     # (g, n, r)
        self.seed = seed
        self.extra = extra                   # _FIXED + arm flags
        self.init_from_dir = init_from_dir   # predecessor rung's run-dir (warm-start), or None
        self.needs_dir = needs_dir           # run-dir that must finish first, or None
        self.proc = None

    @property
    def init_from(self):
        return (os.path.join(self.init_from_dir, "model.eqx")
                if self.init_from_dir else None)

    def ready(self):
        return self.needs_dir is None or _done(self.needs_dir)

    def cmd(self, iters, rollouts):
        g, n, r = self.rung
        c = [sys.executable, "-u", "-m", "ctde_v0.train_ctde",
             "--grid", str(g), "--n-agents", str(n), "--comm-r", str(r),
             "--iters", str(iters), "--rollouts", str(rollouts), "--seed", str(self.seed),
             "--run-dir", self.run_dir, "--ckpt"] + self.extra
        if self.init_from:
            c += ["--init-from", self.init_from]
        return c


def _rung0_dir(out: str, arm: str, seed: int) -> str:
    """The 16²/4 rung-0 run-dir — the transfer eval's source checkpoint for (arm, seed)."""
    g, n, r = _RUNGS[0]
    return os.path.join(out, arm, f"seed{seed}", f"ladder_0_{_tag(g, n, r)}")


def _build_units(out, seeds):
    """3 arms × |seeds| × 3 scales, EVERY rung trained FROM SCRATCH (no warm-start, no deps) —
    each scale is an independent training; the transfer test lives in the zero-shot EVAL phase,
    which transplants the from-scratch 16² checkpoint up to 24²/32². The 24²/32² from-scratch
    runs stand as the in-distribution reference the zero-shot numbers are read against.
    Run-dir <out>/<arm>/seed{S}/ladder_{i}_{tag}."""
    units = []
    for arm in _ARM_IDS:
        arm_extra = _FIXED + _ARMS[arm]
        for s in seeds:
            for i, (g, n, r) in enumerate(_RUNGS):
                rd = os.path.join(out, arm, f"seed{s}", f"ladder_{i}_{_tag(g, n, r)}")
                uid = f"{arm}/s{s}/scale{i}_{_tag(g, n, r)}"
                units.append(Unit(uid, rd, (g, n, r), s, arm_extra,
                                  init_from_dir=None, needs_dir=None))
    return units


# =============================================================================
# Zero-shot transfer eval (in-process)
# =============================================================================


def _returns_to_go(rew_bt, gamma):
    """Monte-Carlo discounted return-to-go G_t = Σ_{k≥t} γ^{k-t} r_k, per episode.
    Bootstrap-free ground truth for the value-prediction error. rew_bt (B,T) -> (B,T)."""
    import jax
    import jax.numpy as jnp

    def per_ep(r):                                        # (T,)
        def step(carry, x):
            carry = x + gamma * carry
            return carry, carry
        _, g = jax.lax.scan(step, jnp.zeros(()), r, reverse=True)
        return g                                          # (T,)

    return jax.vmap(per_ep)(rew_bt)


def _eval_one(arm, ckpt16, target, seed, eval_rollouts):
    """Load the 16²-trained (actor, critic) for `arm` into a `target`-scale skeleton and roll
    out `eval_rollouts` fresh episodes — no training. Returns the value-prediction error
    (RMSE / NRMSE / bias vs the MC return) + downstream final-step coverage."""
    import jax
    import jax.numpy as jnp

    from ctde_v0 import ppo, train_ctde
    from ctde_v0 import env_utils

    g, n, r = target
    # rebuild the EXACT training config at the target scale (parity via train_ctde's own
    # parser), only --grid/--n-agents/--comm-r differ; --rollouts sets collect's batch.
    argv = (["--grid", str(g), "--n-agents", str(n), "--comm-r", str(r),
             "--rollouts", str(eval_rollouts)] + _FIXED + _ARMS[arm])
    cfg, _rd, _ck, _init = train_ctde._parse_args(argv)

    env = env_utils.build_env(cfg)
    key = jax.random.PRNGKey(1234 + seed)
    kload, kroll = jax.random.split(key)
    # the transfer load — scale-invariant params from 16²/4 into the target skeleton (raises
    # loudly on any shape mismatch, which for these arches never happens: conv is grid-
    # invariant, DeepSets is count-invariant).
    state = ppo.init_state_from_checkpoint(env, cfg, ckpt16, kload)
    stencil = ppo.make_stencil(cfg)

    traj = ppo.collect(env, state.actor, state.critic, cfg, stencil, kroll, state.dual.lam)
    v = traj["v_team"]                                   # (B,T) critic prediction
    G = _returns_to_go(traj["rew_team"], cfg.trainer.gamma)   # (B,T) MC return-to-go
    err = v - G
    rmse = float(jnp.sqrt(jnp.mean(err ** 2)))
    ret_std = float(jnp.std(G))
    nrmse = rmse / (ret_std + 1e-8)
    bias = float(jnp.mean(v) - jnp.mean(G))
    coverage = float(traj["coverage"][:, -1].mean())
    return {
        "arm": arm, "seed": seed, "target": _tag(g, n, r),
        "vpe_rmse": round(rmse, 4), "vpe_nrmse": round(nrmse, 4),
        "vpe_bias": round(bias, 4), "ret_std": round(ret_std, 4),
        "mean_return": round(float(jnp.mean(G)), 4),
        "mean_pred": round(float(jnp.mean(v)), 4),
        "coverage_pct": round(coverage, 4),
    }


def _run_eval(out, seeds, eval_rollouts):
    """For every arm × seed whose 16² checkpoint exists, eval it zero-shot at each target
    scale. Writes <out>/transfer_eval.json and prints a per-(arm,target) summary."""
    results = []
    for arm in _ARM_IDS:
        for s in seeds:
            ck = os.path.join(_rung0_dir(out, arm, s), "model.eqx")
            if not (os.path.exists(ck) and os.path.getsize(ck) > 0):
                print(f"[eval skip] {arm}/s{s}: 16² checkpoint missing ({ck})", flush=True)
                continue
            for target in _EVAL_TARGETS:
                rec = _eval_one(arm, ck, target, s, eval_rollouts)
                results.append(rec)
                print(f"[eval] {arm}/s{s} @{rec['target']:9s}  "
                      f"vpe_nrmse={rec['vpe_nrmse']:.3f}  vpe_rmse={rec['vpe_rmse']:.3f}  "
                      f"cov={rec['coverage_pct']*100:5.1f}%", flush=True)

    if results:
        path = os.path.join(out, "transfer_eval.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nsaved transfer eval -> {path}", flush=True)
        _print_transfer_table(results)
    else:
        print("\n[eval] no 16² checkpoints found -- run the training phase first.", flush=True)
    return results


def _print_transfer_table(results):
    """Mean±std across seeds per (arm, target): NRMSE (the transfer number) + coverage."""
    print("\n=== zero-shot transfer: value-pred NRMSE (lower=better) | coverage% ===",
          flush=True)
    hdr = f"  {'arm':9s}" + "".join(f"{_tag(*t):>18s}" for t in _EVAL_TARGETS)
    print(hdr, flush=True)
    for arm in _ARM_IDS:
        cells = []
        for t in _EVAL_TARGETS:
            tag = _tag(*t)
            rows = [r for r in results if r["arm"] == arm and r["target"] == tag]
            if not rows:
                cells.append(f"{'-':>18s}")
                continue
            nr = [r["vpe_nrmse"] for r in rows]
            cv = [r["coverage_pct"] * 100 for r in rows]
            cells.append(f"{_mean(nr):5.2f}±{_std(nr):.2f}|{_mean(cv):4.1f}".rjust(18))
        print(f"  {arm:9s}" + "".join(cells), flush=True)
    print("  (NRMSE = RMSE(v - MC-return)/std(return); cross-scale comparable)", flush=True)


def _mean(xs):
    return sum(xs) / len(xs) if xs else float("nan")


def _std(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


# =============================================================================
# Main
# =============================================================================


def main(argv=None):
    p = argparse.ArgumentParser(
        description="T4 count-invariant critic A/B — warm-start ladder + zero-shot transfer eval")
    p.add_argument("--out", type=str, default="runs/critic_t4",
                   help="parent dir for per-arm/per-seed ladder run-dirs (rel to SharedExploration)")
    p.add_argument("--seeds", type=int, default=3, help="number of seeds (0..S-1)")
    p.add_argument("--iters", type=int, default=2000, help="PPO iters per rung")
    p.add_argument("--rollouts", type=int, default=16, help="episodes per training iter")
    p.add_argument("--eval-rollouts", type=int, default=32,
                   help="episodes per zero-shot transfer eval point")
    p.add_argument("--jobs", type=int, default=1,
                   help="max concurrent train_ctde subprocesses (memory-bound: ~3 @32²/10)")
    p.add_argument("--only", choices=["train", "eval", "both"], default="both",
                   help="run just the training phase, just the transfer eval, or both (default)")
    p.add_argument("--dry-run", action="store_true", help="print the plan, launch nothing")
    args = p.parse_args(argv)

    out = args.out if os.path.isabs(args.out) else os.path.join(_PKG_PARENT, args.out)
    seeds = list(range(args.seeds))
    units = _build_units(out, seeds)

    print(f"=== T4 critic A/B: {len(_ARM_IDS)} arms × {len(seeds)} seeds × {len(_RUNGS)} scales "
          f"= {len(units)} FROM-SCRATCH trainings; "
          f"+ zero-shot transfer eval @ {[_tag(*t) for t in _EVAL_TARGETS]} ===", flush=True)
    print(f"    arms: {_ARM_IDS}   rungs: {[_tag(*x) for x in _RUNGS]}", flush=True)
    print(f"    fixed spec: {' '.join(_FIXED)}", flush=True)
    print(f"    out={out}  iters={args.iters}  jobs={args.jobs}  only={args.only}", flush=True)

    if args.dry_run:
        print("\n-- TRAIN plan (every scale from scratch; no --init-from) --", flush=True)
        for u in units:
            g, n, r = u.rung
            print(f"  {u.uid:26s} grid={g:2d} N={n:2d}  init-from=scratch", flush=True)
        print("\n-- EVAL plan (zero-shot transfer of each 16² checkpoint) --", flush=True)
        for arm in _ARM_IDS:
            for s in seeds:
                src = os.path.relpath(_rung0_dir(out, arm, s), out)
                tgts = ", ".join(_tag(*t) for t in _EVAL_TARGETS)
                print(f"  {arm}/s{s}: {src}/model.eqx  ->  eval @ {tgts}", flush=True)
        print(f"\n[dry-run] {len(units)} trainings + "
              f"{len(_ARM_IDS) * len(seeds) * len(_EVAL_TARGETS)} eval points planned; "
              f"nothing launched.", flush=True)
        return

    if args.only in ("train", "both"):
        done = _schedule(units, args.iters, args.rollouts, max(1, args.jobs))
        print(f"\n=== T4 training done: {len(done)}/{len(units)} produced model.eqx ===",
              flush=True)

    if args.only in ("eval", "both"):
        _run_eval(out, seeds, args.eval_rollouts)

    print(f"\nCompare per-rung history.json under {out}/<arm>/seed*/ and the zero-shot "
          f"transfer table above / {os.path.join(out, 'transfer_eval.json')}.", flush=True)


if __name__ == "__main__":
    main()
