"""MERL coexistence runner — ES evolves the selector, CTDE-gradient trains the executor.

Wires the injectable ES trainer (``es.py``) to the real selector + the PPO/CTDE executor.
Each outer round: (a) ``grad_steps`` of ``ppo.train_step`` train the whole agent (the dense
per-step CTDE signal — incl. the selector); (b) the gradient-current selector is taken as
the ES mean (MERL "inject the learner"); (c) ``es.es_step`` refines ONLY the selector head
against TEAM FITNESS (mean episode return via ``ppo.collect``) on the CURRENT executor,
using common random numbers (one fixed eval key per round) for a fair population compare;
(d) the evolved selector is written back. This is the MERL / feudal-evolutionary loop
(Khadka & Tumer 2019): ES + gradient share CTDE's centralized-training signal (team return /
central critic) and touch disjoint-ish params, so they compose rather than fight.

Requires ``--selector on`` (ES evolves ``actor.selector_head``). CPU smoke:
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m ctde_v0.run_es \
        --grid 10 --n-agents 4 --outer 3 --grad-steps 3 --pop 8 \
        --flock scripted --rollouts 4
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys

import equinox as eqx
import jax
import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ctde_v0 import env_utils, es, ppo
    from ctde_v0.config import Backbone, CTDEConfig, Trainer, World
else:
    from . import env_utils, es, ppo
    from .config import Backbone, CTDEConfig, Trainer, World


def merl_train(env, cfg: CTDEConfig, *, key, n_outer: int, grad_steps: int,
               es_cfg: es.ESConfig, log_fn=None, init_from=None):
    """Run the MERL coexistence loop on ``env``/``cfg`` (which must have selector on).
    ``init_from`` (path to a prior model.eqx) warm-starts the (actor, critic) — the
    scale-ladder entry point (16²→24²→32²), carrying the ES-evolved selector up.
    Returns ``(final_state, history)`` where history is a list of per-round records."""
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    gstate = (ppo.init_state_from_checkpoint(env, cfg, init_from, key) if init_from
              else ppo.init_state(env, cfg, key))

    @eqx.filter_jit
    def eval_return(actor, critic, dual_lam, ekey):
        # team fitness for a given selector = mean episode return on the CURRENT executor.
        traj = ppo.collect(env, actor, critic, cfg, stencil, ekey, dual_lam)
        return traj["rew_team"].sum(axis=1).mean()

    history = []
    k = key
    for outer in range(n_outer):
        # (a) gradient on the executor (dense CTDE signal; trains the whole agent).
        glog = None
        for _ in range(grad_steps):
            k, sk = jax.random.split(k)
            gstate, glog = ppo.train_step(env, gstate, cfg, sk, opt, stencil)
        # (b) take the gradient-current selector as the ES mean (inject the learner).
        theta = es.module_theta(gstate.actor.selector_head)
        # (c) ES step on the selector — common random numbers (one eval key for the round).
        k, ek, pk = jax.random.split(k, 3)

        def eval_fn(th, _ek=ek, _g=gstate):
            actor = eqx.tree_at(
                lambda a: a.selector_head, _g.actor,
                es.set_module_theta(_g.actor.selector_head, th))
            return float(eval_return(actor, _g.critic, _g.dual.lam, _ek))

        theta, info = es.es_step(theta, eval_fn, es_cfg, pk)
        # (d) write the evolved selector back into the executor's actor.
        gstate = eqx.tree_at(
            lambda s: s.actor.selector_head, gstate,
            es.set_module_theta(gstate.actor.selector_head, theta))

        rec = {"outer": outer,
               "best": round(float(info.get("best_fitness", 0.0)), 3),
               "mean": round(float(info.get("mean_fitness", 0.0)), 3),
               "grad_cov": round(float(glog["coverage_pct"]) * 100, 1) if glog else 0.0,
               "grad_conn5": round(float(glog.get("connectivity_real", 0.0)) * 100, 1)
               if glog else 0.0}
        history.append(rec)
        if log_fn is not None:
            log_fn(outer, rec)
    return gstate, history


def _build_cfg(a) -> CTDEConfig:
    """A selector-on config on the locked honest spec (frontier-attn explorer, collision
    mask, soft connectivity), built minimally for the ES runner."""
    base = CTDEConfig()
    return dataclasses.replace(
        base,
        world=World(grid=a.grid, n_agents=a.n_agents, comm_r=a.comm_r, horizon=a.horizon),
        backbone=dataclasses.replace(base.backbone),
        action_head=dataclasses.replace(base.action_head, explorer_tool="frontier_attn"),
        mission_safety=dataclasses.replace(base.mission_safety, mechanism="soft_lambda"),
        trainer=dataclasses.replace(base.trainer),
        collision_mask="on",
        selector="on", flock=a.flock, congestion=a.congestion,
        scale=f"{a.grid}x{a.grid}/{a.n_agents}",
        rollouts_per_iter=a.rollouts, seed=a.seed,
    )


def main(argv=None):
    p = argparse.ArgumentParser(description="MERL coexistence (ES selector + CTDE executor)")
    p.add_argument("--grid", type=int, default=16)
    p.add_argument("--n-agents", type=int, default=4)
    p.add_argument("--comm-r", type=int, default=5)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--flock", choices=["scripted", "learned"], default="scripted")
    p.add_argument("--congestion", choices=["off", "on"], default="off")
    p.add_argument("--outer", type=int, default=50, help="MERL outer rounds")
    p.add_argument("--grad-steps", type=int, default=5, help="PPO steps per outer round")
    p.add_argument("--rollouts", type=int, default=16)
    p.add_argument("--pop", type=int, default=16, help="ES population size")
    p.add_argument("--sigma", type=float, default=0.05)
    p.add_argument("--es-lr", type=float, default=0.05)
    p.add_argument("--es-kind", choices=["nes", "cem"], default="nes")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--init-from", type=str, default=None,
                   help="warm-start (actor, critic) from a prior model.eqx — the ES ladder rung-to-rung")
    a = p.parse_args(argv)

    cfg = _build_cfg(a)
    es_cfg = es.ESConfig(pop_size=a.pop, sigma=a.sigma, lr=a.es_lr, kind=a.es_kind)
    env = env_utils.build_env(cfg)
    print(f"MERL: {cfg.scale} selector=on flock={a.flock} congestion={a.congestion} "
          f"| outer={a.outer} grad_steps={a.grad_steps} pop={a.pop} kind={a.es_kind}",
          flush=True)

    def _log(outer, rec):
        print(f"[round {outer:3d}] es_best={rec['best']:8.2f} es_mean={rec['mean']:8.2f} "
              f"grad_cov={rec['grad_cov']:5.1f}% grad_conn.5={rec['grad_conn5']:5.1f}%",
              flush=True)

    gstate, history = merl_train(env, cfg, key=jax.random.PRNGKey(a.seed),
                                 n_outer=a.outer, grad_steps=a.grad_steps,
                                 es_cfg=es_cfg, log_fn=_log, init_from=a.init_from)
    if a.run_dir:
        os.makedirs(a.run_dir, exist_ok=True)
        with open(os.path.join(a.run_dir, "es_history.json"), "w") as f:
            json.dump(history, f, indent=2)
        # Save the deployable (actor, critic) snapshot — the ES-evolved selector + the
        # CTDE-trained executor — so the learned policy can be loaded
        # (init_state_from_checkpoint), rendered into the gallery, deployed, warm-started.
        # config.json is what make_report/render read to rebuild the skeleton.
        if __package__ in (None, ""):
            from ctde_v0 import checkpoint as ckpt
        else:
            from . import checkpoint as ckpt
        with open(os.path.join(a.run_dir, "config.json"), "w") as f:
            json.dump(cfg.to_dict(), f, indent=2)
        ckpt.save_model(os.path.join(a.run_dir, "model.eqx"),
                        (gstate.actor, gstate.critic), meta=cfg.to_dict())
        print(f"saved es_history.json + config.json + model.eqx -> {a.run_dir}", flush=True)
    return history


if __name__ == "__main__":
    main()
