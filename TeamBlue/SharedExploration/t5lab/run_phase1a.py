"""Phase 1a — the go/no-go gate. Train T5 (Backbone + goal head + MVProp) from scratch on a
walled terrain, zero-shot eval on the held-out SAR set, print coverage vs the reactive-25% and
classical-79% references.

Run ON BALTHAR from TeamBlue/SharedExploration:
    PY=$HOME/ZymeraLab/.venv/bin/python
    SD=<dir with maps_sar.json> PYTHONPATH=.:../../../FiedlerValueEstimation $PY -m t5lab.run_phase1a

Config schema + env contracts verified against ctde_v0. Remaining runtime unknowns are the
Backbone obs channel count (read from obs.shape[1]) and the GridEnv FixedWall swap (mirrors
the existing scratchpad eval_driver / render_classical).
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
from zymera.env import GridEnv
from t5lab.actor import T5Actor
from t5lab.train import train

SD = os.environ.get("SD", ".")


def base_config(terrain="clutter", grid=32, n_agents=10, iters=400):
    return from_dict(dict(
        seed=0, iters=iters, rollouts_per_iter=16, critic_mode="decentral",
        collision_mask="on",   # hard collision-mask ALWAYS on (like the previous architecture)
        world=dict(recipe="comm-coverage", grid=grid, n_agents=n_agents, comm_r=5,
                   cover_r=1, terrain=terrain, sense_walls=True, sense_free=True, horizon=100),
        backbone=dict(width=64, depth=2, mp_rounds=2, agg="max", heads=4, norm="layer"),
        action_head=dict(K=9, stride=3),
        trainer=dict(lr=3e-4, gamma=0.99, gae_lambda=0.95, clip=0.2,
                     ppo_epochs=6, minibatches=4, max_grad_norm=0.5),
        loss=dict(vf_coef=0.5),
        regularization=dict(entropy_coef=0.01, weight_decay=1e-4, dropout=0.0),
        reward=dict(w_coverage=3.0, w_connectivity=2.0, w_collision=-4.0),
    ))


def build_actor(cfg, env, key):
    obs0, _ = env.reset(key)
    in_ch = obs0.shape[1]                                   # (N, C, H, W) -> C
    return T5Actor(in_ch, K_goal=cfg.action_head.K, K_prop=32,
                   backbone_cfg=cfg.backbone, key=key)


class _FW(eqx.Module):
    wall: jax.Array
    def walls(self, k, h, w):
        return self.wall


def _env_with_wall(cfg, wall):
    """Rebuild the comm-coverage GridEnv swapping only the terrain to a fixed SAR wall
    (mirrors scratchpad/render_classical.env_with_wall)."""
    b = _eu.build_env(cfg)
    return GridEnv(grid_h=b.grid_h, grid_w=b.grid_w, n_agents=b.n_agents, cover_r=b.cover_r,
                   wall_sense_r=b.wall_sense_r, sense_free=b.sense_free,
                   terrain=_FW(jnp.asarray(wall, bool)), spawn=b.spawn, dynamics=b.dynamics,
                   channel=b.channel, obs=b.obs, mission=b.mission)


def coverage_on_map(cfg, actor, wall, key, steps=200):
    """Greedy zero-shot rollout on a fixed SAR wall -> coverage fraction of free cells."""
    env = _env_with_wall(cfg, wall)
    stencil = make_stencil(cfg)
    obs, state = env.reset(key)
    k = key
    for _t in range(steps):
        k, sk = jax.random.split(k)
        pos = state.body.position
        adj = _eu.kb_adjacency(pos, cfg)
        blocked = _navfield_blocked(state, cfg)
        z = actor.belief(obs, adj, inference=True)
        goal_logits, _v, _l2 = actor.heads(z)
        gi = jnp.argmax(goal_logits, -1)
        h, w = state.wall.shape
        goal = goal_targets(pos, stencil, h, w)[jnp.arange(pos.shape[0]), gi]
        move = jnp.argmax(actor.move_logits(pos, goal, blocked), -1)
        obs, state, *_ = env.step(state, move, sk)
    covered = np.asarray(state.covered).astype(bool)
    free = ~np.asarray(wall).astype(bool)
    return float((covered & free).sum()) / float(free.sum())


def main():
    key = jax.random.PRNGKey(0)
    ka, kt, ke = jax.random.split(key, 3)
    cfg = base_config()
    env = _eu.build_env(cfg)
    actor = build_actor(cfg, env, ka)
    print("[phase1a] training from scratch (16 rollouts)...", flush=True)
    actor, hist = train(env, actor, cfg, kt)
    print(f"[phase1a] final train return: {hist[-1].get('ret'):.3f}", flush=True)
    ckpt = f"{SD}/t5_phase1a_actor.eqx"
    eqx.tree_serialise_leaves(ckpt, actor)
    print(f"[phase1a] saved checkpoint -> {ckpt}", flush=True)

    maps = json.load(open(f"{SD}/maps_sar.json"))
    covs = []
    for m in maps:
        wall = np.zeros((32, 32), bool)
        wall.flat[m["walls"]] = True
        covs.append(coverage_on_map(cfg, actor, jnp.asarray(wall), ke))
    mean_cov = 100 * float(np.mean(covs))
    print(f"\n[phase1a GATE] zero-shot SAR coverage = {mean_cov:.1f}%  "
          f"(reactive ref 25%, classical ref 79%)", flush=True)
    print("[phase1a GATE]", "PASS (>> 25%)" if mean_cov > 40 else "INVESTIGATE (<= 40%)", flush=True)


if __name__ == "__main__":
    main()
