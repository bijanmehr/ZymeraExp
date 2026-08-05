"""Per-agent contribution (COMA on the attention critic) — the #70 engine end to end.

CPU-only. Run:
    JAX_PLATFORMS=cpu \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m pytest \
        ctde_v0/tests/test_contribution.py -q
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
import numpy as np  # noqa: E402

from ctde_v0 import env_utils  # noqa: E402
from ctde_v0.config import Backbone, CTDEConfig, Trainer, World  # noqa: E402
from ctde_v0.contribution import (  # noqa: E402
    episode_contributions, per_agent_summary, step_contributions)
from ctde_v0.nets import Actor, AttnCritic  # noqa: E402

_T, _N, _D, _K, _H = 8, 5, 16, 9, 32


def _critic(seed=0):
    return AttnCritic(feat_dim=_D, n_actions=_K, hid=_H, heads=4, key=jax.random.PRNGKey(seed))


class TestContributionShapesAndSummary:
    def test_episode_shape(self):
        critic = _critic()
        k = jax.random.PRNGKey(1)
        feats = jax.random.normal(k, (_T, _N, _D))
        logits = jax.random.normal(jax.random.PRNGKey(2), (_T, _N, _K))
        goals = jax.random.randint(jax.random.PRNGKey(3), (_T, _N), 0, _K)
        contrib = episode_contributions(critic, feats, logits, goals, _K)
        assert contrib.shape == (_T, _N)
        assert jnp.all(jnp.isfinite(contrib))

    def test_summary_share_normalized(self):
        contrib = jax.random.normal(jax.random.PRNGKey(4), (_T, _N))
        s = per_agent_summary(contrib)
        assert s["mean"].shape == (_N,) and s["share"].shape == (_N,)
        assert np.isclose(float(s["share"].sum()), 1.0, atol=1e-5)
        assert 0.0 <= float(s["gini"]) < 1.0

    def test_even_team_zero_gini(self):
        """Identical contribution across agents -> even share, Gini 0 (a team that divides)."""
        contrib = jnp.ones((_T, _N))
        s = per_agent_summary(contrib)
        assert np.allclose(np.asarray(s["share"]), 1.0 / _N, atol=1e-6)
        assert float(s["gini"]) < 1e-5

    def test_one_carrier_high_gini(self):
        """One agent does everything -> spiky share, Gini near (N-1)/N (a brittle team)."""
        contrib = jnp.zeros((_T, _N)).at[:, 0].set(3.0)
        s = per_agent_summary(contrib)
        assert float(s["share"][0]) > 0.99
        assert float(s["gini"]) > 0.5


class TestComposesWithActor:
    def test_real_actor_features(self):
        """End to end in the CTDE stack: real Actor beliefs + goal-logits -> contributions."""
        cfg = CTDEConfig(world=World(grid=8, n_agents=4, comm_r=3, horizon=6),
                         backbone=Backbone(width=16, depth=2, mp_rounds=2),
                         trainer=Trainer(minibatches=2, ppo_epochs=2),
                         iters=1, rollouts_per_iter=2, seed=0)
        env = env_utils.build_env(cfg)
        obs, state = env.reset(jax.random.PRNGKey(0))
        adj = env_utils.kb_adjacency(state.body.position, cfg)
        actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                      dropout=0.0, key=jax.random.PRNGKey(1))
        goal_logits, _role, _v, _l2, feat, _h = actor(obs, adj, inference=True)
        n = cfg.world.n_agents
        assert feat.shape == (n, cfg.backbone.width)
        assert goal_logits.shape == (n, cfg.action_head.K)
        critic = AttnCritic(feat_dim=cfg.backbone.width, n_actions=cfg.action_head.K,
                            hid=cfg.backbone.width, heads=4, key=jax.random.PRNGKey(2))
        goals = jnp.argmax(goal_logits, axis=-1)
        adv = step_contributions(critic, feat, goal_logits, goals, cfg.action_head.K)
        assert adv.shape == (n,) and jnp.all(jnp.isfinite(adv))
        # a tiny stacked "episode" to exercise the time sweep + summary on real features
        feats_T = jnp.stack([feat] * 4)
        logits_T = jnp.stack([goal_logits] * 4)
        goals_T = jnp.stack([goals] * 4)
        s = per_agent_summary(episode_contributions(critic, feats_T, logits_T, goals_T,
                                                    cfg.action_head.K))
        assert np.isclose(float(s["share"].sum()), 1.0, atol=1e-5)
