"""Smoke tests for the grounded CTDE v0 agent (LPAC + goal head + aux-λ₂).

CPU-only by construction. Run:
    JAX_PLATFORMS=cpu \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m pytest \
        ctde_v0/tests/test_smoke.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))  # .../SharedExploration
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from ctde_v0 import controller as ctrl  # noqa: E402
from ctde_v0 import env_utils, ppo  # noqa: E402
from ctde_v0.config import (  # noqa: E402
    Backbone, CTDEConfig, MissionSafety, Trainer, World, from_dict,
)
from ctde_v0.nets import Actor, Critic  # noqa: E402


def _tiny_cfg(**over) -> CTDEConfig:
    """A tiny, fast config: 8x8 / 3 agents / 6-step horizon."""
    base = CTDEConfig(
        world=World(grid=8, n_agents=3, comm_r=3, horizon=6),
        backbone=Backbone(width=16, depth=2, mp_rounds=2),
        trainer=Trainer(minibatches=2, ppo_epochs=2),
        iters=1, rollouts_per_iter=2, seed=0,
    )
    if over:
        import dataclasses
        base = dataclasses.replace(base, **over)
    return base


# ---- config schema ----------------------------------------------------------

def test_config_roundtrip():
    cfg = _tiny_cfg()
    d = cfg.to_dict()
    assert d["backbone"]["agg"] == "max"
    assert d["action_head"]["kind"] == "goal_pointer"
    assert d["mission_safety"]["mechanism"] == "action_mask"
    cfg2 = from_dict(d)
    assert cfg2.to_dict() == d


# ---- env / shapes -----------------------------------------------------------

def test_env_shapes():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    assert env.n_agents == cfg.world.n_agents
    assert env.n_actions == 5
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (cfg.world.n_agents, env.obs.obs_channels, 8, 8)
    assert env.central_obs(state).shape == (env.obs.central_channels, 8, 8)
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    assert adj.shape == (cfg.world.n_agents, cfg.world.n_agents)
    assert not bool(jnp.diag(adj).any())  # diagonal cleared


# ---- backbone + heads -------------------------------------------------------

def test_actor_backbone_forward():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K,
                  backbone_cfg=cfg.backbone, dropout=0.0, key=jax.random.PRNGKey(2))
    goal_logits, value, l2_hat, z = actor(obs, adj, inference=True)
    assert goal_logits.shape == (cfg.world.n_agents, cfg.action_head.K)
    assert value.shape == (cfg.world.n_agents,)
    assert l2_hat.shape == (cfg.world.n_agents,)
    assert z.shape == (cfg.world.n_agents, cfg.backbone.width)


def test_aggregators_all_run():
    """mean / max / multihead must all produce finite beliefs."""
    cfg0 = _tiny_cfg()
    env = env_utils.build_env(cfg0)
    obs, state = env.reset(jax.random.PRNGKey(7))
    adj = env_utils.kb_adjacency(state.body.position, cfg0)
    import dataclasses
    for agg in ("mean", "max", "multihead"):
        bb = dataclasses.replace(cfg0.backbone, agg=agg)
        actor = Actor(env.obs.obs_channels, cfg0.action_head.K,
                      backbone_cfg=bb, dropout=0.0, key=jax.random.PRNGKey(3))
        _, _, _, z = actor(obs, adj, inference=True)
        assert jnp.isfinite(z).all(), agg


# ---- controller: only valid moves ------------------------------------------

def test_controller_emits_only_valid_moves():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(4))
    stencil = ppo.make_stencil(cfg)
    pos = state.body.position
    h, w = state.wall.shape
    goal_cells = ctrl.goal_targets(pos, stencil, h, w)
    valid_targets = env.dynamics.targets(state)
    action_valid = env.action_mask(state)
    # try every candidate as the chosen goal for every agent
    n, K = pos.shape
    for k in range(K):
        goal = goal_cells[:, k]
        move = ctrl.greedy_move(pos, goal, valid_targets, action_valid)
        ok = jnp.take_along_axis(action_valid, move[:, None], axis=1)[:, 0]
        assert bool(ok.all()), (k, move)


def test_safe_goal_mask_keeps_a_candidate():
    cfg = _tiny_cfg(mission_safety=MissionSafety(mechanism="action_mask",
                                                 min_lambda2=0.5))
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(8))
    stencil = ppo.make_stencil(cfg)
    pos = state.body.position
    h, w = state.wall.shape
    goal_cells = ctrl.goal_targets(pos, stencil, h, w)
    safe = ctrl.safe_goal_mask(
        pos, goal_cells, env.dynamics.targets(state), env.action_mask(state),
        cfg.world.comm_r, cfg.connectivity.lambda2_sharp, cfg.mission_safety.min_lambda2,
    )
    assert safe.shape == (pos.shape[0], stencil.shape[0])
    assert bool(safe.any(-1).all())          # every agent has >=1 safe candidate
    assert bool(safe[:, 0].all())            # 'here' is always safe


# ---- reward + λ₂ ------------------------------------------------------------

def test_reward_and_lambda2():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(3))
    move = jnp.zeros((cfg.world.n_agents,), jnp.int32)
    _, state2, _, _, info = env.step(state, move, jax.random.PRNGKey(4))
    rew = env_utils.compose_reward(info["reward_terms"], state2, cfg)
    assert rew.shape == (cfg.world.n_agents,)
    l2 = env_utils.true_lambda2(state2.body.position, cfg)
    assert l2.shape == () and float(l2) >= 0.0


# ---- one full PPO iteration -------------------------------------------------

def test_one_train_step_runs():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    new_state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    for k in ("ep_reward", "coverage_pct", "connectivity_pct", "mean_lambda2",
              "aux_loss", "aux_acc", "ctrl_valid_frac", "degree_reg"):
        assert k in logs and jnp.isfinite(logs[k]), (k, logs.get(k))
    # the controller guarantee: 100% valid moves in the rollout.
    assert float(logs["ctrl_valid_frac"]) == 1.0
    import equinox as eqx
    p0 = jax.tree_util.tree_leaves(eqx.filter(state.actor, eqx.is_array))
    p1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(p0, p1)), "actor unchanged"


def test_soft_lambda_mechanism_runs():
    cfg = _tiny_cfg(mission_safety=MissionSafety(mechanism="soft_lambda"))
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    _, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"])
    assert float(logs["ctrl_valid_frac"]) == 1.0


def test_short_train_loop():
    cfg = _tiny_cfg(iters=2)
    env = env_utils.build_env(cfg)
    _, history = ppo.train(env, cfg)
    assert len(history) == 2
    assert all(jnp.isfinite(h["aux_loss"]) for h in history)


if __name__ == "__main__":
    test_config_roundtrip()
    test_env_shapes()
    test_actor_backbone_forward()
    test_aggregators_all_run()
    test_controller_emits_only_valid_moves()
    test_safe_goal_mask_keeps_a_candidate()
    test_reward_and_lambda2()
    test_one_train_step_runs()
    test_soft_lambda_mechanism_runs()
    test_short_train_loop()
    print("all smoke tests passed")
