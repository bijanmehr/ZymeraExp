"""Privileged data-generation policies for the Fiedler study.

`guardrail_disperse_actions` is the **hard-connectivity guardrail** that drives the
realistic, always-connected, dispersed training regime the deployed swarm runs in:
every agent spreads out (anti-crowding) but a move is only taken if it keeps the
*global* comm graph a single connected component. This is privileged (it reads all
agent positions) and runs offline, so it is plain numpy and need not be fast.

Move encoding (matches `zymera.ACTION_DELTAS`):
    0=STAY  1=NORTH(row-1)  2=EAST(col+1)  3=SOUTH(row+1)  4=WEST(col-1)
"""
import jax
import numpy as np
from scipy.sparse.csgraph import connected_components

import zymera.metrics as zmetrics

# (5,2) row/col deltas, index = action id.
ACTION_DELTAS = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int32)


def _is_connected(positions, comm_r):
    """True iff the Chebyshev<=comm_r comm graph over `positions` (N,2) is one component.

    Uses scipy connected_components (a single graph traversal) rather than an
    eigendecomposition. Trivially True for N<=1.
    """
    n = positions.shape[0]
    if n <= 1:
        return True
    adj = np.asarray(zmetrics.adjacency(positions, radius=comm_r))
    n_comp, _ = connected_components(adj, directed=False)
    return n_comp == 1


def _min_dist_to_others(cand, others):
    """Min Chebyshev distance from a single point `cand` (2,) to `others` (M,2).

    Returns +inf when there are no other agents (M==0), so a lone agent is free to
    move anywhere (its dispersion score is unbounded -> ties broken at random).
    """
    if others.shape[0] == 0:
        return np.inf
    return float(np.abs(cand[None, :] - others).max(axis=1).min())


def guardrail_disperse_actions(positions, grid, comm_r, key):
    """Hard-connectivity dispersion actions for every agent.

    Args:
        positions: (N,2) int array of current (row,col) cells.
        grid: world side length; candidate cells are clamped to [0, grid).
        comm_r: Chebyshev comm radius for the connectivity mask.
        key: a JAX PRNGKey, used only for uniform random tie-breaking.

    Returns:
        (N,) int32 numpy array of actions in [0,5).

    Agents are processed one at a time in a key-shuffled order. Each agent's chosen
    move is *committed* into a working copy of the positions BEFORE the next agent is
    considered, so every agent validates its move against the configuration the env
    will actually realize this step. For the current agent we:
      1. enumerate the 5 candidate next-cells, dropping any that leave the grid;
      2. keep a candidate only if, with this agent moved there (others at their
         already-committed positions), the GLOBAL comm graph is a single connected
         component -- except STAY, which is always kept (it leaves the working set
         unchanged), so the allowed set is never empty;
      3. pick the allowed candidate maximizing this agent's min distance to all
         other agents (anti-crowding spread), breaking ties uniformly at random.

    Because STAY preserves the working set and a move is only committed when it keeps
    the working set connected, the working set stays connected by induction -- so the
    full joint action, applied simultaneously by the env, keeps the graph connected
    (given a connected start). This is the simultaneous-move-safe form of the guard.
    """
    positions = np.asarray(positions, dtype=np.int32)
    n = positions.shape[0]

    # per-agent numpy seeds for tie-breaking + a key-shuffled processing order so the
    # spread is not biased toward low-index agents.
    seeds = np.asarray(jax.random.randint(key, (n,), 0, np.iinfo(np.int32).max))
    order = np.asarray(jax.random.permutation(jax.random.fold_in(key, 1), n))

    working = positions.copy()                              # committed positions so far
    actions = np.zeros(n, dtype=np.int32)
    for i in order:
        i = int(i)
        cur = working[i]
        others = np.delete(working, i, axis=0)              # (N-1,2) committed
        rng = np.random.default_rng(int(seeds[i]))

        best_action = 0                                     # STAY fallback
        best_score = _min_dist_to_others(cur, others)       # STAY's dispersion score
        best_tie = rng.random()                             # random key for STAY

        for act in range(1, 5):                             # the four real moves
            cand = cur + ACTION_DELTAS[act]
            if np.any(cand < 0) or np.any(cand >= grid):    # out of bounds
                continue
            moved = working.copy()
            moved[i] = cand
            if not _is_connected(moved, comm_r):            # hard connectivity mask
                continue
            score = _min_dist_to_others(cand, others)
            tie = rng.random()
            # strictly-better score wins; equal score -> uniform random tie-break.
            if score > best_score + 1e-9 or (
                abs(score - best_score) <= 1e-9 and tie > best_tie
            ):
                best_action, best_score, best_tie = act, score, tie

        actions[i] = best_action
        working[i] = cur + ACTION_DELTAS[best_action]       # commit before next agent
    return actions
