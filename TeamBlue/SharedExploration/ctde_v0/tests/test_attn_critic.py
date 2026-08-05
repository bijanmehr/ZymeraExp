"""Attention critic (MAAC) + COMA counterfactual — the per-agent credit engine (#68 → #70).

CPU-only. Run:
    JAX_PLATFORMS=cpu \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m pytest \
        ctde_v0/tests/test_attn_critic.py -q
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

from ctde_v0.nets import AttnCritic, coma_counterfactual  # noqa: E402

_N, _D, _K, _H, _HEADS = 6, 16, 5, 32, 4


def _critic(seed=0):
    return AttnCritic(feat_dim=_D, n_actions=_K, hid=_H, heads=_HEADS,
                      key=jax.random.PRNGKey(seed))


def _inputs(seed=1, n=_N):
    k1, k2 = jax.random.split(jax.random.PRNGKey(seed))
    feats = jax.random.normal(k1, (n, _D))
    acts = jax.random.randint(k2, (n,), 0, _K)
    onehot = jax.nn.one_hot(acts, _K)
    return feats, acts, onehot


class TestAttnCritic:
    def test_per_agent_shape(self):
        """One Q per agent — the whole point vs the scalar team Critic."""
        feats, _a, onehot = _inputs()
        q = _critic()(feats, onehot)
        assert q.shape == (_N,)
        assert jnp.all(jnp.isfinite(q))

    def test_permutation_equivariance(self):
        """Symmetric over agents (no identity): permuting agents permutes Q the same way."""
        critic = _critic()
        feats, _a, onehot = _inputs()
        perm = jnp.array([3, 0, 5, 1, 4, 2])
        q = critic(feats, onehot)
        q_perm = critic(feats[perm], onehot[perm])
        assert np.allclose(np.asarray(q_perm), np.asarray(q)[np.asarray(perm)], atol=1e-5)

    def test_single_agent_no_nan(self):
        """n==1: the self-masked attention row is all -inf — must not NaN out."""
        feats, _a, onehot = _inputs(n=1)
        q = _critic()(feats, onehot)
        assert q.shape == (1,) and jnp.all(jnp.isfinite(q))
        adv, _qt = coma_counterfactual(_critic(), feats, onehot,
                                       jnp.zeros((1, _K)))
        assert jnp.all(jnp.isfinite(adv))


class TestCOMA:
    def test_deterministic_policy_zero_advantage(self):
        """If π_i is (near) certain on the taken action, the counterfactual baseline equals
        Q_i(taken) -> advantage ≈ 0 (you cannot out-perform your own sure action)."""
        critic = _critic()
        feats, acts, onehot = _inputs()
        logits = jnp.where(onehot > 0, 30.0, -30.0)        # ≈ one-hot policy on the taken action
        adv, q_taken = coma_counterfactual(critic, feats, onehot, logits)
        assert adv.shape == (_N,) and q_taken.shape == (_N,)
        assert np.allclose(np.asarray(adv), 0.0, atol=1e-3)

    def test_uniform_policy_gives_live_credit(self):
        """Under a uniform policy the baseline is the mean Q over own actions, so the taken
        action's advantage is generally non-zero — the engine assigns real per-agent credit."""
        critic = _critic()
        feats, _a, onehot = _inputs()
        adv, _q = coma_counterfactual(critic, feats, onehot, jnp.zeros((_N, _K)))
        assert jnp.all(jnp.isfinite(adv))
        assert float(jnp.max(jnp.abs(adv))) > 1e-4          # not trivially zero

    def test_q_depends_on_own_action(self):
        """Q_i must actually vary with agent i's own action (else COMA is degenerate)."""
        critic = _critic()
        feats, _a, onehot = _inputs()
        # agent 0: swap its action to two different one-hots, hold the rest fixed
        a_alt0 = onehot.at[0].set(jax.nn.one_hot(0, _K))
        a_alt1 = onehot.at[0].set(jax.nn.one_hot(_K - 1, _K))
        q0 = critic(feats, a_alt0)[0]
        q1 = critic(feats, a_alt1)[0]
        assert abs(float(q0 - q1)) > 1e-5

    def test_baseline_matches_expectation(self):
        """A_i = Q_taken − Σ_a π_i(a) Q_i(a'): recompute the baseline by hand and match."""
        critic = _critic()
        feats, acts, onehot = _inputs()
        logits = jax.random.normal(jax.random.PRNGKey(7), (_N, _K))
        adv, q_taken = coma_counterfactual(critic, feats, onehot, logits)
        pi = jax.nn.softmax(logits, axis=-1)
        # hand-rolled per-agent marginal over own action
        cand = jnp.eye(_K)
        man = []
        for i in range(_N):
            qi = jnp.array([critic(feats, onehot.at[i].set(c))[i] for c in cand])  # (K,)
            man.append(float(q_taken[i] - jnp.sum(pi[i] * qi)))
        assert np.allclose(np.asarray(adv), np.array(man), atol=1e-4)
