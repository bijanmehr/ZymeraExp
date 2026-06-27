"""L1 controller + goal-candidate stencil + the mission-safety mechanism.

The multi-level (L3->L1) action stack lives here. L3 (the goal head in
``nets.Actor``) picks one of K candidate relative waypoints; this module turns
that choice into the actual env move, emitting ONLY valid 1-step moves (STAY
fallback). The simulator therefore still sees movement-only actions and the
100-step budget is unchanged (agent_architecture.md L2/L1; EXPERIMENT_PLAN 1a').

Pieces
------
* :func:`goal_stencil` — the fixed K relative offsets (center + 8 compass dirs
  at ``stride``, in ABSOLUTE cells so the goal geometry is scale-invariant).
* :func:`goal_targets` — absolute goal cell for every agent × candidate.
* :func:`greedy_move` — the L1 controller: of the env-valid moves
  (``dynamics.targets`` / ``action_mask``), take the one that most reduces
  Chebyshev distance to the chosen goal; STAY if none helps.
* :func:`team_lambda2_after_action` — true λ₂ the team would have after a
  proposed joint move (used by the action-mask mechanism to score candidates).
* :func:`candidate_first_moves` — for each agent × candidate, the env move the
  greedy controller would take FIRST (so the mechanism can mask a candidate by
  the connectivity it would cause).

Everything is pure JAX (vmap/scan-safe). The env (its ``dynamics`` /
``action_mask`` tables) is the single source of truth for wall-awareness.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
from zymera.env import ACTION_DELTAS, ActionId, N_ACTIONS
from zymera.missions_terms import _lambda2

# Compass order for the candidate stencil. Index 0 is "here" (a STAY goal).
# Offsets are unit directions; scaled by ``stride`` in :func:`goal_stencil`.
_COMPASS = jnp.array(
    [
        (0, 0),    # 0 here
        (-1, 0),   # 1 N
        (0, 1),    # 2 E
        (1, 0),    # 3 S
        (0, -1),   # 4 W
        (-1, 1),   # 5 NE
        (1, 1),    # 6 SE
        (1, -1),   # 7 SW
        (-1, -1),  # 8 NW
    ],
    dtype=jnp.int32,
)


def goal_stencil(K: int, stride: int) -> jax.Array:
    """(K, 2) int32 relative offsets — the first ``K`` of the compass stencil,
    each scaled by ``stride`` (cell 0 = here, unscaled)."""
    if K < 1 or K > _COMPASS.shape[0]:
        raise ValueError(f"K must be in 1..{_COMPASS.shape[0]}, got {K}")
    base = _COMPASS[:K]
    scale = jnp.where((jnp.arange(K) == 0)[:, None], 1, stride)
    return (base * scale).astype(jnp.int32)


def goal_targets(pos: jax.Array, stencil: jax.Array, h: int, w: int) -> jax.Array:
    """(N, K, 2) int32 absolute goal cells (clipped in-bounds) for each agent ×
    candidate. ``pos`` (N,2), ``stencil`` (K,2)."""
    g = pos[:, None, :] + stencil[None, :, :]                  # (N,K,2)
    r = jnp.clip(g[..., 0], 0, h - 1)
    c = jnp.clip(g[..., 1], 0, w - 1)
    return jnp.stack([r, c], axis=-1).astype(jnp.int32)


def _cheby(a: jax.Array, b: jax.Array) -> jax.Array:
    """Chebyshev distance between cell arrays broadcasting on the last axis."""
    return jnp.max(jnp.abs(a - b), axis=-1)


def greedy_move(pos: jax.Array, goal: jax.Array, valid_targets: jax.Array,
                action_valid: jax.Array) -> jax.Array:
    """(N,) int32 — L1 greedy controller move toward ``goal``.

    For each agent: among env-VALID actions (``action_valid`` (N,A) bool, from
    ``env.action_mask``), pick the one whose committed cell (``valid_targets``
    (N,A,2), from ``dynamics.targets``) minimizes Chebyshev distance to the goal.
    Ties and "no move helps" fall back to STAY (STAY is always valid). The result
    is always a valid move, so :class:`SequentialClaim`/the env never reverts it.

    ``goal`` (N,2) int32 — the chosen absolute goal cell per agent.
    """
    d = _cheby(valid_targets, goal[:, None, :]).astype(jnp.float32)   # (N,A) dist if taken
    # forbid invalid actions by a large distance so argmin never selects them.
    d = jnp.where(action_valid, d, jnp.inf)
    # current distance (= STAY distance, STAY target == own cell).
    stay = int(ActionId.STAY)
    best = jnp.argmin(d, axis=-1).astype(jnp.int32)                   # (N,)
    # if the best valid move doesn't strictly improve on STAY, STAY.
    d_best = jnp.take_along_axis(d, best[:, None], axis=-1)[:, 0]
    d_stay = d[:, stay]
    move = jnp.where(d_best < d_stay, best, jnp.int32(stay))
    return move


def candidate_first_moves(pos: jax.Array, goal_cells: jax.Array,
                          valid_targets: jax.Array, action_valid: jax.Array) -> jax.Array:
    """(N, K) int32 — the FIRST greedy move each agent would take for every
    candidate goal. Used by the action-mask mechanism to evaluate the
    connectivity each candidate would cause.

    ``goal_cells`` (N,K,2). Vmaps :func:`greedy_move` over the K candidates.
    """
    def for_candidate(goal_k):                                       # goal_k (N,2)
        return greedy_move(pos, goal_k, valid_targets, action_valid)
    # vmap over K (axis 1 of goal_cells) -> (K, N) -> transpose
    moves = jax.vmap(for_candidate, in_axes=1, out_axes=0)(goal_cells)  # (K,N)
    return moves.T                                                    # (N,K)


def positions_after(pos: jax.Array, actions: jax.Array,
                    valid_targets: jax.Array) -> jax.Array:
    """(N,2) int32 — committed positions if every agent took ``actions`` (N,).

    Reads the env's ``valid_targets`` (N,A,2) table (already wall/boundary
    resolved). NoCollision semantics (agents may share a cell) — matches the
    comm-coverage recipe default; the mechanism only needs the comm graph, which
    is collision-agnostic.
    """
    n = actions.shape[0]
    return valid_targets[jnp.arange(n), actions]                     # (N,2)


def team_lambda2_after(pos_next: jax.Array, comm_r: int, sharp: float) -> jax.Array:
    """Scalar true λ₂ of the soft comm-graph at ``pos_next`` (N,2)."""
    return _lambda2(pos_next, comm_r, sharp)


def safe_goal_mask(pos: jax.Array, goal_cells: jax.Array, valid_targets: jax.Array,
                   action_valid: jax.Array, comm_r: int, sharp: float,
                   min_lambda2: float) -> jax.Array:
    """(N, K) bool — action-mask mechanism: a candidate is SAFE iff, when its
    agent alone takes the greedy first move toward it (others STAY), the team's
    true λ₂ stays >= ``min_lambda2``.

    This is a LOCAL, per-agent guardrail (each agent screens its own candidates
    independently against the connectivity floor) — the "forbid goals that would
    disconnect" mechanism. If a row would mask ALL candidates, the "here"
    candidate (index 0, a STAY goal) is force-unmasked so a valid goal always
    exists.
    """
    n = pos.shape[0]
    first_moves = candidate_first_moves(pos, goal_cells, valid_targets, action_valid)  # (N,K)
    K = goal_cells.shape[1]
    stay = jnp.full((n,), int(ActionId.STAY), dtype=jnp.int32)

    def lambda2_for_agent_candidate(i, kk):
        actions = stay.at[i].set(first_moves[i, kk])                 # only agent i moves
        pos_next = positions_after(pos, actions, valid_targets)
        return team_lambda2_after(pos_next, comm_r, sharp)

    # vectorize over (N agents) x (K candidates) -> (N, K) true-λ₂-if-taken.
    l2 = jax.vmap(jax.vmap(lambda2_for_agent_candidate, in_axes=(None, 0)),
                  in_axes=(0, None))(jnp.arange(n), jnp.arange(K))   # (N,K)
    safe = l2 >= min_lambda2
    # guarantee >=1 safe candidate per agent: force "here" (index 0) on.
    safe = safe.at[:, 0].set(True)
    return safe
