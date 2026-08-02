"""Isolated rollout: one batched episode under (T5Actor, critic). Mirrors the structure of
ctde_v0's ``_single_rollout`` but the L1 move comes from MVProp, not ``_goal_to_move``.

Reuses ctde_v0 read-only: kb_adjacency, make_stencil, goal_targets, _navfield_blocked,
compose_reward. Zero ctde_v0 edits.

Phase-1a scope: single goal-offset PG action + single move PG action; no selector / roles /
difference-credit / dual (those are ctde_v0 knobs we don't need for the free-space gate).
"""
import jax
import jax.numpy as jnp

from ctde_v0 import env_utils as _eu
from ctde_v0.ppo import make_stencil, _navfield_blocked          # READ-ONLY
from ctde_v0.controller import goal_targets                       # READ-ONLY

_NEG = -1e9


def _team_reward(info, env, state, move, next_state, cfg):
    """Phase-1a team reward. BALTHAR-VERIFY: confirm ctde_v0's reward-term keys/weights.
    Ledger weights: r = 3*new_coverage + 2*reach_fraction - 4*collision. We read the env's
    populated ``info['reward_terms']`` and re-weight, matching ctde_v0.compose_reward's intent.
    """
    try:
        from ctde_v0.env_utils import compose_reward   # BALTHAR-VERIFY import path + signature
        return compose_reward(info, cfg)                # expected () or (N,) team reward
    except Exception:
        # Fallback: the env's own scalar reward (recipe-composed). Align weights on balthar.
        return info.get("reward", jnp.zeros(()))


def rollout(env, actor, cfg, key, stencil=None):
    """One episode (fixed-length scan of ``cfg.horizon`` steps). Returns a trajectory pytree."""
    if stencil is None:
        stencil = make_stencil(cfg)
    reset_key, scan_key = jax.random.split(key)
    obs0, state0 = env.reset(reset_key)
    H = int(getattr(cfg.trainer, "horizon", getattr(cfg, "horizon", 100)))   # BALTHAR-VERIFY field

    def body(carry, _):
        state, obs, k = carry
        k, ak, mk, sk = jax.random.split(k, 4)
        pos = state.body.position                                  # (N,2)
        adj = _eu.kb_adjacency(pos, cfg)                           # (N,N)
        blocked = _navfield_blocked(state, cfg)                    # (N,H,W) known walls (optimistic)

        z = actor.belief(obs, adj, inference=True)                 # (N,W)
        goal_logits, value, l2 = actor.heads(z)                    # (N,K),(N,),(N,)

        # L3: sample a goal offset (no mask in free-space Phase 1a; all offsets valid)
        goal_idx = jax.random.categorical(ak, goal_logits, axis=-1)            # (N,)
        goal_logp = jnp.take_along_axis(
            jax.nn.log_softmax(goal_logits, -1), goal_idx[:, None], -1)[:, 0]  # (N,)
        h, w = state.wall.shape
        goal_cells = goal_targets(pos, stencil, h, w)                          # (N,K,2)
        goal = goal_cells[jnp.arange(pos.shape[0]), goal_idx]                  # (N,2)

        # L1: MVProp move over the belief occupancy toward the chosen goal
        move_logits = actor.move_logits(pos, goal, blocked)                    # (N,5)
        move = jax.random.categorical(mk, move_logits, axis=-1)               # (N,)
        move_logp = jnp.take_along_axis(
            jax.nn.log_softmax(move_logits, -1), move[:, None], -1)[:, 0]      # (N,)

        obs_n, state_n, rew, done, info = env.step(state, move, sk)            # BALTHAR-VERIFY return
        rew_team = _team_reward(info, env, state, move, state_n, cfg)
        v_team = value.mean()                                                  # DTE team value

        out = dict(rew_team=rew_team, v_team=v_team, value=value, l2=l2,
                   goal_logp=goal_logp, move_logp=move_logp,
                   goal_logits=goal_logits, move_logits=move_logits,
                   obs=obs, adj=adj, blocked=blocked, pos=pos, goal_idx=goal_idx,
                   move=move, done=done)
        return (state_n, obs_n, k), out

    (state_last, obs_last, _), traj = jax.lax.scan(
        body, (state0, obs0, scan_key), None, length=H)
    # bootstrap value for GAE
    adj_last = _eu.kb_adjacency(state_last.body.position, cfg)
    z_last = actor.belief(obs_last, adj_last, inference=True)
    _, v_last, _ = actor.heads(z_last)
    traj["v_last"] = v_last.mean()
    return traj
