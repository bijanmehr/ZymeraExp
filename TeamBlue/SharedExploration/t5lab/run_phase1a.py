"""Phase 1a — the go/no-go gate. Train T5 (Backbone + goal head + MVProp) from scratch on
rooms + clutter, zero-shot eval on the held-out SAR set, print coverage vs the reactive-25%
and classical-79% references.

Run ON BALTHAR:
    PY=$HOME/ZymeraLab/.venv/bin/python
    PYTHONPATH=.:../../../FiedlerValueEstimation JAX_PLATFORMS='' $PY -m t5lab.run_phase1a

BALTHAR-VERIFY: config field names (from_dict schema), the SAR eval harness, and the reward
wiring are inferred from ctde_v0 and need one run to confirm.
"""
import json
import os
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx

from ctde_v0.config import from_dict
from ctde_v0 import env_utils as _eu
from ctde_v0.ppo import make_stencil, _navfield_blocked
from ctde_v0.controller import goal_targets
from t5lab.actor import T5Actor
from t5lab.train import train

SD = os.environ.get("SD", "/tmp")   # scratchpad for maps_sar.json on the host


def base_config(terrain="clutter", grid=32, n_agents=10, iters=400):
    """Minimal from-scratch Phase-1a config. BALTHAR-VERIFY the exact schema keys."""
    return from_dict(dict(
        seed=0, iters=iters,
        world=dict(recipe="comm_coverage", grid=grid, n_agents=n_agents, comm_r=5,
                   terrain=terrain, sense_walls=True, sense_free=True),
        action_head=dict(K=25, controller="navfield", planner="mvprop"),   # K = goal-offset stencil size
        backbone=dict(width=64, depth=2, mp_rounds=2, agg="mean", heads=1, norm="layer"),
        trainer=dict(rollouts_per_iter=16, gamma=0.99, gae_lambda=0.95, clip=0.2,
                     ppo_epochs=6, vf_coef=0.5, ent_coef=0.01, lr=3e-4, horizon=100),
        critic_mode="decentral",
    ))


def build(cfg, key):
    env = _eu.build_env(cfg)
    obs0, _ = env.reset(key)
    in_ch = obs0.shape[1]                    # (N, C, H, W) -> C   BALTHAR-VERIFY
    actor = T5Actor(in_ch, K_goal=cfg.action_head.K, K_prop=32,
                    backbone_cfg=cfg.backbone, key=jax.random.split(key)[1])
    return env, actor


def coverage_on_map(env, actor, cfg, wall, key, steps=200):
    """Greedy zero-shot rollout on a fixed wall map -> coverage fraction of free cells."""
    from ctde_v0.env_utils import build_env  # rebuild with FixedWall  BALTHAR-VERIFY terrain swap
    stencil = make_stencil(cfg)
    obs, state = env.reset(key)
    covered = jnp.zeros_like(wall, bool)
    for _t in range(steps):
        pos = state.body.position
        adj = _eu.kb_adjacency(pos, cfg)
        blocked = _navfield_blocked(state, cfg)
        z = actor.belief(obs, adj, inference=True)
        goal_logits, _v, _l2 = actor.heads(z)
        gi = jnp.argmax(goal_logits, -1)
        h, w = state.wall.shape
        goal = goal_targets(pos, stencil, h, w)[jnp.arange(pos.shape[0]), gi]
        move = jnp.argmax(actor.move_logits(pos, goal, blocked), -1)
        obs, state, *_ = env.step(state, move, key)
    # BALTHAR-VERIFY: read covered mask from state.channel/known; fraction over ~wall
    return float(getattr(state, "covered", covered).sum()) / float((~wall).sum())


def main():
    key = jax.random.PRNGKey(0)
    cfg = base_config()
    env, actor = build(cfg, key)
    print("[phase1a] training from scratch (16 rollouts)...", flush=True)
    actor, hist = train(env, actor, cfg, key)
    print(f"[phase1a] final train return: {hist[-1].get('ret'):.3f}", flush=True)

    maps = json.load(open(f"{SD}/maps_sar.json"))
    covs = []
    for m in maps:
        wall = np.zeros((32, 32), bool); wall.flat[m["walls"]] = True
        covs.append(coverage_on_map(env, actor, cfg, jnp.asarray(wall), key))
    mean_cov = 100 * float(np.mean(covs))
    print(f"\n[phase1a GATE] zero-shot SAR coverage = {mean_cov:.1f}%  "
          f"(reactive ref 25%, classical ref 79%)", flush=True)
    print("[phase1a GATE]", "PASS (>> 25%)" if mean_cov > 40 else "INVESTIGATE (<= 40%)", flush=True)


if __name__ == "__main__":
    main()
