"""Per-agent CONTRIBUTION via the MAAC attention critic + COMA counterfactual (#70).

This is what the attention critic (#68) powers: given a (trained) :class:`~ctde_v0.nets.AttnCritic`
and a rollout's per-step ``(agent beliefs feat_i, policy goal-logits, sampled goals)``, score each
agent's COMA counterfactual advantage at every step — *how much its chosen action beat its own
policy-average, holding the rest of the team fixed* — and aggregate to a per-agent contribution:
who is load-bearing vs interchangeable. That distribution is the resilience / difference-reward
signal (Wolpert–Tumer ``D_i = G − G_{−i}``, here the learned-critic approximation) and the bridge
to the minimum-m-of-n question.

The critic is the learned Q; :func:`ctde_v0.nets.coma_counterfactual` is the engine. This module
adds only the rollout sweep + the team-level aggregation — no new learning.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp

from ctde_v0.nets import coma_counterfactual

_NEG = -1e9


def step_contributions(critic, feats, goal_logits, goals, n_actions, gmask=None):
    """One step. ``feats (N,D)`` agent beliefs, ``goal_logits (N,K)`` policy logits,
    ``goals (N,)`` the sampled goal indices. ``gmask (N,K)`` bool (optional) = the
    valid-action mask used at sample time (invalid goals are excluded from the policy
    marginal, exactly as the agent saw them). Returns ``adv (N,)`` per-agent COMA
    counterfactual advantage = the per-step contribution."""
    onehot = jax.nn.one_hot(goals, n_actions, dtype=feats.dtype)
    logits = goal_logits if gmask is None else jnp.where(gmask, goal_logits, _NEG)
    adv, _q = coma_counterfactual(critic, feats, onehot, logits)
    return adv


def episode_contributions(critic, feats_T, goal_logits_T, goals_T, n_actions, gmask_T=None):
    """Sweep a whole episode (vmap over time). ``feats_T (T,N,D)``, ``goal_logits_T (T,N,K)``,
    ``goals_T (T,N)`` (+ optional ``gmask_T (T,N,K)``) -> contributions ``(T,N)``."""
    def one(ft, gl, g, gm):
        return step_contributions(critic, ft, gl, g, n_actions, gm)
    if gmask_T is None:
        return jax.vmap(lambda ft, gl, g: one(ft, gl, g, None))(feats_T, goal_logits_T, goals_T)
    return jax.vmap(one)(feats_T, goal_logits_T, goals_T, gmask_T)


def per_agent_summary(contributions):
    """Aggregate ``contributions (T,N)`` -> a per-agent readout:

    * ``mean``      (N,) — average signed contribution (positive = pulled its weight).
    * ``magnitude`` (N,) — mean |contribution| (how load-bearing, regardless of sign).
    * ``share``     (N,) — magnitude normalized to sum 1: each agent's slice of the total
      credit. A flat share = the team divides labour evenly; a spiky share = a few agents
      carry the mission (the minimum-m-of-n picture, the flood-vs-divide diagnostic).
    * ``gini``      ()   — inequality of the share in [0,1): 0 = perfectly even, →1 = one
      agent carries everything. The single-number "is this team resilient or brittle?"."""
    mean = jnp.mean(contributions, axis=0)                           # (N,)
    mag = jnp.mean(jnp.abs(contributions), axis=0)                   # (N,)
    share = mag / (mag.sum() + 1e-8)                                 # (N,)
    n = share.shape[0]
    diffs = jnp.abs(share[:, None] - share[None, :]).sum()          # Σ_ij |s_i − s_j|
    gini = diffs / (2.0 * n * (share.sum() + 1e-8))                 # standard Gini on the shares
    return {"mean": mean, "magnitude": mag, "share": share, "gini": gini}
