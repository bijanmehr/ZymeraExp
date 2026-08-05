"""Isolated rollout: one episode under (T5Actor). Mirrors ctde_v0's ``_single_rollout``
structure but the L1 move comes from MVProp, not ``_goal_to_move``.

Reuses ctde_v0 read-only: kb_adjacency, make_stencil, _navfield_blocked, goal_targets,
compose_reward. Zero ctde_v0 edits.

Phase-1a scope: goal-offset PG action + MVProp move PG action; no selector / roles /
difference-credit / dual. Contracts verified against ctde_v0 (comm-coverage env):
  env.reset(key)               -> (obs (N,C,H,W), state)
  env.step(state, move, key)   -> (obs, state, rew, done, info); info["reward_terms"] dict
  compose_reward(terms, world, cfg) -> (N,) per-agent reward; team = mean
  horizon = cfg.world.horizon (=100)
"""
import jax
import jax.numpy as jnp

from ctde_v0 import env_utils as _eu
from ctde_v0.ppo import make_stencil, _navfield_blocked          # READ-ONLY
from ctde_v0.controller import goal_targets                       # READ-ONLY


def resolve_conflicts(moves, targets):
    """Priority-based collision resolver (self-contained; no ctde_v0 dependency, since the two
    run-boxes' ctde_v0 clones differ). moves (N,), targets (N,A,2) = the cell each action lands
    on. Process agents in index order: an agent keeps its move unless its target cell was already
    claimed by a lower-index agent this step, in which case it STAYs (action 0). Guarantees no two
    agents share a cell after the step (STAY lands on the agent's own, distinct, cell)."""
    N = moves.shape[0]
    committed = targets[jnp.arange(N), moves]                     # (N,2)
    stay_cell = targets[:, 0]                                     # (N,2) STAY = own cell
    def body(claimed, i):
        cell = committed[i]
        taken = jnp.all(claimed == cell[None, :], axis=-1).any()  # claimed by an earlier agent?
        mv = jnp.where(taken, 0, moves[i])
        newcell = jnp.where(taken, stay_cell[i], cell)
        return claimed.at[i].set(newcell), mv
    claimed0 = jnp.full((N, 2), -999, dtype=committed.dtype)
    _, out = jax.lax.scan(body, claimed0, jnp.arange(N))
    return out


_DELTAS5 = jnp.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])   # ACTION_DELTAS


def greedy_toward(pos, goal):
    """(N,) deterministic action that most reduces Chebyshev distance to the goal — the
    NO-PLANNER control (drops MVProp). On open terrain this is optimal navigation."""
    cand = pos[:, None, :] + _DELTAS5[None, :, :]                # (N,5,2)
    d = jnp.max(jnp.abs(cand - goal[:, None, :]), -1)            # (N,5)
    return jnp.argmin(d, -1).astype(jnp.int32)                   # (N,)


def rollout(env, actor, cfg, key, stencil=None, controller="mvprop"):
    """One fixed-length episode (cfg.world.horizon steps). Returns a trajectory pytree.
    controller: "mvprop" (learned planner, move is a PG action) or "greedy" (deterministic
    step-toward-goal, goal is the only PG action) — the no-planner control."""
    if stencil is None:
        stencil = make_stencil(cfg)
    reset_key, scan_key = jax.random.split(key)
    obs0, state0 = env.reset(reset_key)
    horizon = int(cfg.world.horizon)

    def body(carry, _):
        state, obs, k = carry
        k, ak, mk, sk = jax.random.split(k, 4)
        pos = state.body.position                                  # (N,2)
        n = pos.shape[0]
        adj = _eu.kb_adjacency(pos, cfg)                           # (N,N)
        blocked = _navfield_blocked(state, cfg)                    # (N,H,W) known walls (optimistic)

        z = actor.belief(obs, adj, inference=True)                 # (N,W)
        goal_logits, value, l2 = actor.heads(z)                    # (N,K),(N,),(N,)

        goal_idx = jax.random.categorical(ak, goal_logits, axis=-1)            # (N,)
        goal_logp = jnp.take_along_axis(
            jax.nn.log_softmax(goal_logits, -1), goal_idx[:, None], -1)[:, 0]  # (N,)
        h, w = state.wall.shape
        goal_cells = goal_targets(pos, stencil, h, w)                          # (N,K,2)
        goal = goal_cells[jnp.arange(n), goal_idx]                             # (N,2)

        if controller == "greedy":
            move = greedy_toward(pos, goal)                                    # deterministic
            move_logp = jnp.zeros(n)                                           # goal = only PG action
        else:
            move_logits = actor.move_logits(pos, goal, blocked)                # (N,5)
            move = jax.random.categorical(mk, move_logits, axis=-1)           # (N,) SAMPLED (for PG)
            move_logp = jnp.take_along_axis(
                jax.nn.log_softmax(move_logits, -1), move[:, None], -1)[:, 0]  # (N,)
        # hard COLLISION resolution: the env permits cell overlap, so break simultaneous
        # same-cell convergence (allowed for collisions; NEVER for connectivity).
        exec_move = resolve_conflicts(move, env.dynamics.targets(state))
        obs_n, state_n, _rew, done, info = env.step(state, exec_move, sk)
        rew_agent = _eu.compose_reward(info["reward_terms"], state_n, cfg)     # (N,)
        rew_team = rew_agent.mean()                                            # () DTE target
        v_team = value.mean()                                                  # () DTE value

        out = dict(rew_team=rew_team, v_team=v_team, l2=l2,
                   goal_logp=goal_logp, move_logp=move_logp,
                   obs=obs, adj=adj, blocked=blocked, pos=pos, goal_idx=goal_idx, move=move)
        return (state_n, obs_n, k), out

    (state_last, obs_last, _), traj = jax.lax.scan(
        body, (state0, obs0, scan_key), None, length=horizon)
    adj_last = _eu.kb_adjacency(state_last.body.position, cfg)
    z_last = actor.belief(obs_last, adj_last, inference=True)
    _, v_last, _ = actor.heads(z_last)
    traj["v_last"] = v_last.mean()                                            # () GAE bootstrap
    return traj
