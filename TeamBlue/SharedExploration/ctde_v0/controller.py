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
* :func:`occupied_cell_mask` — (N,A) bool flagging actions whose committed cell
  is the CURRENT cell of another agent (the hard collision-mask signal); both
  controllers can ``forbid_collision`` to remove those actions before argmin/max.
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


def occupied_cell_mask(pos: jax.Array, valid_targets: jax.Array) -> jax.Array:
    """(N, A) bool — True where action ``a``'s committed target ``valid_targets[i,a]``
    lands on the CURRENT cell of ANY OTHER agent ``j != i`` (the hard collision
    signal). Exact integer-cell match, broadcast over agents.

    STAY (``valid_targets[i, STAY] == pos[i]``, the agent's own cell) is NEVER
    flagged: an agent's own current cell does not count as "occupied by another",
    and we force STAY's column off regardless so it always remains selectable.

    CAVEAT: this forbids moving onto a cell another agent occupies RIGHT NOW; two
    agents simultaneously claiming the same EMPTY cell is still possible under the
    NoCollision env semantics (see :func:`positions_after`). A deterministic
    index-priority resolver for that case is a future refinement — not built here.
    """
    n = pos.shape[0]
    # tgt[i,a] == pos[j] for some j != i  (compare every target to every cur cell).
    same = jnp.all(valid_targets[:, :, None, :] == pos[None, None, :, :], axis=-1)  # (N,A,N)
    other = ~jnp.eye(n, dtype=bool)                                  # (N,N) j != i
    occ = jnp.any(same & other[:, None, :], axis=-1)                 # (N,A) any other on it
    stay = int(ActionId.STAY)
    return occ.at[:, stay].set(False)                               # STAY never blocked


def greedy_move(pos: jax.Array, goal: jax.Array, valid_targets: jax.Array,
                action_valid: jax.Array, forbid_collision: bool = False) -> jax.Array:
    """(N,) int32 — L1 greedy controller move toward ``goal``.

    For each agent: among env-VALID actions (``action_valid`` (N,A) bool, from
    ``env.action_mask``), pick the one whose committed cell (``valid_targets``
    (N,A,2), from ``dynamics.targets``) minimizes Chebyshev distance to the goal.
    Ties and "no move helps" fall back to STAY (STAY is always valid). The result
    is always a valid move, so :class:`SequentialClaim`/the env never reverts it.

    ``goal`` (N,2) int32 — the chosen absolute goal cell per agent. When
    ``forbid_collision`` is True the hard collision-mask (:func:`occupied_cell_mask`)
    removes actions whose target sits on another agent's CURRENT cell (their
    distance is set to +inf before the argmin); STAY always stays selectable.
    """
    d = _cheby(valid_targets, goal[:, None, :]).astype(jnp.float32)   # (N,A) dist if taken
    # forbid invalid actions by a large distance so argmin never selects them.
    d = jnp.where(action_valid, d, jnp.inf)
    if forbid_collision:
        # hard collision mask: same +inf-distance trick removes occupied-cell moves.
        d = jnp.where(occupied_cell_mask(pos, valid_targets), jnp.inf, d)
    # current distance (= STAY distance, STAY target == own cell).
    stay = int(ActionId.STAY)
    best = jnp.argmin(d, axis=-1).astype(jnp.int32)                   # (N,)
    # if the best valid move doesn't strictly improve on STAY, STAY.
    d_best = jnp.take_along_axis(d, best[:, None], axis=-1)[:, 0]
    d_stay = d[:, stay]
    move = jnp.where(d_best < d_stay, best, jnp.int32(stay))
    return move


# =============================================================================
# Nav-field planner (L2) + reactive controller (L1)  — ADDITIVE, gated behind
# ``action_head.controller == 'navfield'`` (default 'greedy' -> none of this runs).
# =============================================================================
#
# A distance-field ("nav-field") planner computes a distance-to-GOAL field over the
# FREE cells of the KB occupancy grid and the reactive L1 controller descends its
# gradient one env-valid step at a time (with a hard collision veto + STAY fallback).
# On OPEN terrain the field is the plain Chebyshev (king-move) distance, so the emitted
# move is identical to :func:`greedy_move`; around KNOWN walls the field routes around
# them (something the pure Chebyshev-descent greedy controller cannot do).
#
# Everything is pure JAX (fixed-iteration relaxation over the small grid -> vmap/scan/
# jit-safe). ``_FAR`` is a large FINITE sentinel (not ``inf``) so ``+1`` relaxations and
# the argmin never produce inf/nan.

_FAR = jnp.float32(1e9)   # "unreached / blocked" distance sentinel (finite, jit-safe)


def _king_neighbor_min(D: jax.Array) -> jax.Array:
    """(H,W) — for every cell, the MIN distance over its 8 king-move neighbours (out-of-
    bounds neighbours read ``_FAR``). One Bellman-Ford relaxation ring."""
    f = _FAR
    n = jnp.pad(D[:-1, :], ((1, 0), (0, 0)), constant_values=f)     # from (r-1, c)
    s = jnp.pad(D[1:, :],  ((0, 1), (0, 0)), constant_values=f)     # from (r+1, c)
    w = jnp.pad(D[:, :-1], ((0, 0), (1, 0)), constant_values=f)     # from (r, c-1)
    e = jnp.pad(D[:, 1:],  ((0, 0), (0, 1)), constant_values=f)     # from (r, c+1)
    nw = jnp.pad(D[:-1, :-1], ((1, 0), (1, 0)), constant_values=f)  # (r-1, c-1)
    ne = jnp.pad(D[:-1, 1:],  ((1, 0), (0, 1)), constant_values=f)  # (r-1, c+1)
    sw = jnp.pad(D[1:, :-1],  ((0, 1), (1, 0)), constant_values=f)  # (r+1, c-1)
    se = jnp.pad(D[1:, 1:],   ((0, 1), (0, 1)), constant_values=f)  # (r+1, c+1)
    return jnp.min(jnp.stack([n, s, w, e, nw, ne, sw, se], axis=0), axis=0)


def _bfs_distance_field(goal: jax.Array, blocked: jax.Array, n_iters: int) -> jax.Array:
    """(H,W) float32 — king-move BFS distance transform to ``goal`` over the FREE cells of
    ``blocked`` (True = obstacle). Jacobi min-plus relaxation for ``n_iters`` sweeps: the
    goal cell is pinned to 0, blocked cells stay ``_FAR``, every other cell relaxes to
    ``min(self, min_king_neighbour + 1)``. ``n_iters`` fixed (grid is small) -> vmap/scan-
    safe. Unreached cells (field not yet propagated that far) stay ``_FAR`` — the reactive
    controller then STAYs / moves only as far as the field reached (graceful)."""
    h, w = blocked.shape
    gr, gc = goal[0], goal[1]
    seed = jnp.full((h, w), _FAR, jnp.float32).at[gr, gc].set(0.0)

    def body(D, _):
        D2 = jnp.minimum(D, _king_neighbor_min(D) + 1.0)            # relax one ring
        D2 = jnp.where(blocked, _FAR, D2)                          # obstacles impassable
        D2 = D2.at[gr, gc].set(0.0)                                # re-pin the goal source
        return D2, None

    D, _ = jax.lax.scan(body, seed, xs=None, length=n_iters)
    return D


def _fmm_sweep(D: jax.Array, blocked: jax.Array, row_rev: bool, col_rev: bool) -> jax.Array:
    """One fast-sweeping (Gauss-Seidel) corner pass over ``D``: rows are processed in order
    (reversed when ``row_rev``) carrying the already-updated previous row's 3 king
    neighbours (+1), and within each row cells relax left->right (reversed when ``col_rev``)
    off the running cumulative minimum (+1). The four (row_rev, col_rev) corners together
    propagate king-move distances along every characteristic direction — the fast-sweeping
    Eikonal method. Obstacles are forced to ``_FAR``. Pure JAX (nested ``lax.scan``)."""
    Dr = D[::-1] if row_rev else D
    Br = blocked[::-1] if row_rev else blocked
    w = D.shape[1]

    def row_step(prev_row, cur):
        d_row, b_row = cur
        up = prev_row + 1.0
        upl = jnp.pad(prev_row[:-1], (1, 0), constant_values=_FAR) + 1.0
        upr = jnp.pad(prev_row[1:], (0, 1), constant_values=_FAR) + 1.0
        base = jnp.minimum(d_row, jnp.minimum(up, jnp.minimum(upl, upr)))   # (W,) from above
        row = base[::-1] if col_rev else base
        brow = b_row[::-1] if col_rev else b_row

        def col_step(prev_cell, x):
            d_cell, b_cell = x
            v = jnp.minimum(d_cell, prev_cell + 1.0)               # relax off left neighbour
            v = jnp.where(b_cell, _FAR, v)
            return v, v

        _, out = jax.lax.scan(col_step, _FAR, (row, brow))         # left->right cumulative
        out = out[::-1] if col_rev else out
        out = jnp.where(b_row, _FAR, out)
        return out, out

    _, Dout = jax.lax.scan(row_step, jnp.full((w,), _FAR), (Dr, Br))
    return Dout[::-1] if row_rev else Dout


def _fmm_distance_field(goal: jax.Array, blocked: jax.Array, n_rounds: int) -> jax.Array:
    """(H,W) float32 — fast-sweeping Eikonal approximation of the king-move distance to
    ``goal``. Each round runs the four corner sweeps (:func:`_fmm_sweep`); a single round
    already resolves distances on obstacle-free regions (so nav-field == greedy on open
    terrain), extra rounds route around walls. Goal pinned to 0 each round."""
    h, w = blocked.shape
    gr, gc = goal[0], goal[1]
    D = jnp.full((h, w), _FAR, jnp.float32).at[gr, gc].set(0.0)
    D = jnp.where(blocked, _FAR, D).at[gr, gc].set(0.0)

    def rnd(D, _):
        for row_rev in (False, True):
            for col_rev in (False, True):
                D = _fmm_sweep(D, blocked, row_rev, col_rev)
        D = D.at[gr, gc].set(0.0)                                  # re-pin the goal source
        return D, None

    D, _ = jax.lax.scan(rnd, D, xs=None, length=n_rounds)
    return D


def nav_distance_field(goal: jax.Array, blocked: jax.Array, planner: str = "wavefront") -> jax.Array:
    """(H,W) float32 distance-to-``goal`` field over the FREE cells of ``blocked`` (True =
    obstacle), the L2 nav-field. ``planner`` (Python-static) selects the solver:

    * ``"wavefront"`` / ``"bfs"`` — BFS distance transform (Jacobi min-plus relaxation).
    * ``"astar"``                 — reuses the BFS field. Descending a shortest-path field
      reconstructs the FIRST move of an A* shortest path; a real priority-queue A* is not
      vmap-safe, and the emitted 1-step move is identical, so the field is shared.
    * ``"fmm"``                   — fast-sweeping Eikonal approximation (:func:`_fmm_...`).

    Iterations are fixed from the grid size (``2*(H+W)`` BFS sweeps / 4 fast-sweep rounds) —
    enough to fully propagate an open grid and route moderate wall detours; the grid is
    small (agent_architecture.md L2). Pure JAX (vmap/scan/jit-safe)."""
    h, w = blocked.shape
    if planner == "fmm":
        return _fmm_distance_field(goal, blocked, n_rounds=4)
    # wavefront / bfs / astar -> BFS distance transform (down-gradient == shortest-path move)
    return _bfs_distance_field(goal, blocked, n_iters=2 * (h + w))


def navfield_move(pos: jax.Array, goal: jax.Array, blocked: jax.Array,
                  valid_targets: jax.Array, action_valid: jax.Array,
                  planner: str = "wavefront", forbid_collision: bool = True) -> jax.Array:
    """(N,) int32 — the L2 nav-field + L1 reactive controller move toward ``goal``.

    For each agent: plan a distance field to its goal over its OWN KB occupancy
    ``blocked[i]`` (:func:`nav_distance_field`), then among the env-VALID actions take the
    one whose committed cell (``valid_targets``) most reduces the field — descending the
    nav-field gradient one step. The reactive collision veto (the hard
    :func:`occupied_cell_mask`, on by default) removes moves onto a cell another agent
    occupies NOW; STAY is always selectable and is the fallback when no valid move strictly
    improves on staying (so the emitted move is always env-valid, exactly like
    :func:`greedy_move`). On open terrain the field is Chebyshev distance -> this reproduces
    the greedy move; around known walls it routes around them.

    ``blocked`` (N,H,W) bool — per-agent obstacle map (known walls; unknown/known-free are
    traversable, optimistic). ``planner`` (Python-static) picks the field solver."""
    n = pos.shape[0]
    fields = jax.vmap(lambda g, b: nav_distance_field(g, b, planner))(goal, blocked)  # (N,H,W)

    def gather_agent(i):
        D = fields[i]                                              # (H,W) agent i's field
        return D[valid_targets[i, :, 0], valid_targets[i, :, 1]]  # (A,) field if action taken

    d = jax.vmap(gather_agent)(jnp.arange(n))                     # (N,A) down-gradient score
    d = jnp.where(action_valid, d, _FAR)                         # forbid invalid actions
    if forbid_collision:
        d = jnp.where(occupied_cell_mask(pos, valid_targets), _FAR, d)  # reactive veto
    stay = int(ActionId.STAY)
    best = jnp.argmin(d, axis=-1).astype(jnp.int32)              # (N,) steepest descent
    d_best = jnp.take_along_axis(d, best[:, None], axis=-1)[:, 0]
    d_stay = d[:, stay]
    return jnp.where(d_best < d_stay, best, jnp.int32(stay))      # STAY unless a move helps


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


def _local_conn_score(pos: jax.Array, comm_r: int, sharp: float) -> jax.Array:
    """(N,) per-agent LOCAL connectivity proxy from positions ``pos`` (N,2).

    The relay anchor uses the soft per-node degree = sum of smooth edge weights to
    in-range teammates (≈ a soft count of neighbours, the local λ̂₂-contribution an
    agent can read off its own neighbourhood). Higher = better anchored. This is a
    LOCAL signal (each agent's own incident edge mass), not the global λ₂ — it is
    what a partial-observability relay can actually compute to "hold the bridge".
    """
    d = jnp.max(jnp.abs(pos[:, None, :] - pos[None, :, :]), axis=-1).astype(jnp.float32)
    w = jax.nn.sigmoid(sharp * (comm_r - d))
    w = w * (1.0 - jnp.eye(pos.shape[0]))                            # zero self
    return w.sum(-1)                                                 # (N,) soft degree


def relay_move(pos: jax.Array, valid_targets: jax.Array, action_valid: jax.Array,
               comm_r: int, sharp: float, forbid_collision: bool = False) -> jax.Array:
    """(N,) int32 — relay L1 controller: each relay agent takes the env-VALID move
    that MAXIMIZES its own local connectivity proxy (soft incident-edge mass),
    others held at their current cell while it is scored. STAY is the safe default
    (it is always valid and is scored at the agent's current anchoring).

    For agent i and candidate action a, we move ONLY i to ``valid_targets[i,a]``
    (everyone else stays) and read i's soft degree at the resulting layout; the
    valid action with the largest value wins. This makes the relay hold/strengthen
    the bridge from purely local information (agent_architecture.md: relay = the
    λ̂₂-anchor tool). Result is always a valid move, so the env never reverts it.

    When ``forbid_collision`` is True the hard collision-mask
    (:func:`occupied_cell_mask`) removes actions whose target sits on another
    agent's CURRENT cell (their score is set to -inf before the argmax); STAY
    always stays selectable.
    """
    n, A = action_valid.shape
    blocked = occupied_cell_mask(pos, valid_targets) if forbid_collision else None

    def score_agent(i):
        # for each action a: i -> its committed cell, others stay at pos.
        def for_action(a):
            tgt = valid_targets[i, a]                                # (2,)
            pos_next = pos.at[i].set(tgt)                            # only i moves
            return _local_conn_score(pos_next, comm_r, sharp)[i]     # scalar
        s = jax.vmap(for_action)(jnp.arange(A))                      # (A,)
        s = jnp.where(action_valid[i], s, -jnp.inf)                  # forbid invalid
        if blocked is not None:
            s = jnp.where(blocked[i], -jnp.inf, s)                   # forbid collisions
        return jnp.argmax(s).astype(jnp.int32)                      # best valid action

    return jax.vmap(score_agent)(jnp.arange(n))                      # (N,)


def relay_hold_move(pos: jax.Array, valid_targets: jax.Array, action_valid: jax.Array,
                    comm_r: int, sharp: float, hold_target: float = 0.5,
                    forbid_collision: bool = False) -> jax.Array:
    """(N,) int32 — the "hold" relay L1 controller: a STATIC BEACON. Each relay
    agent simply STAYS where it is — a low-energy "don't wander, keep the bridge
    from where you stand" tool — UNLESS staying would leave it isolated, in which
    case it takes the SINGLE env-valid move that best restores a neighbour (minimal
    movement to re-anchor). STAY is the safe default and is always valid.

    This is the complement of :func:`relay_move` (the ``lambda2_anchor`` tool, which
    ACTIVELY climbs local connectivity every step): ``relay_hold_move`` moves ONLY
    when its anchoring drops below the floor, so a well-connected relay never wanders
    off its post (agent_architecture.md: relay = "stop & hold the connection";
    Relay-tool axis = hold-connection *heuristic*).

    Decision (per relay i, from purely LOCAL information):
      * soft_deg_i(STAY) = i's soft incident-edge mass at its CURRENT cell
        (:func:`_local_conn_score`, the same proxy ``relay_move`` / the edge-margin
        signal use, so "anchored enough" agrees across the agent).
      * if soft_deg_i(STAY) >= ``hold_target`` -> STAY (it is comfortably anchored).
      * else (about to isolate) -> among the env-VALID moves, take the one that
        MAXIMIZES soft_deg_i at the resulting layout (others held), i.e. the minimal
        step that best re-establishes a neighbour. STAY is included in the argmax, so
        if no move improves anchoring the agent still STAYs (never an invalid move).

    ``hold_target`` is the soft-degree floor that defines "isolated" (default 0.5 — a
    single in-range neighbour at the comm edge already carries ~0.5 soft mass at the
    default sharpness, so the relay holds as long as it has roughly one live link).
    When ``forbid_collision`` is True the hard collision-mask
    (:func:`occupied_cell_mask`) removes actions whose target sits on another agent's
    CURRENT cell (their score is set to -inf before the argmax); STAY always stays
    selectable. Result is always a valid move, so the env never reverts it. Pure JAX
    (vmap/scan/jit-safe) — same signature/contract as :func:`relay_move`.
    """
    n, A = action_valid.shape
    stay = int(ActionId.STAY)
    blocked = occupied_cell_mask(pos, valid_targets) if forbid_collision else None
    # current soft degree at the agent's present cell (the STAY anchoring).
    deg_now = _local_conn_score(pos, comm_r, sharp)                  # (N,)

    def score_agent(i):
        # for each action a: move ONLY i to its committed cell, read i's soft degree.
        def for_action(a):
            tgt = valid_targets[i, a]                                # (2,)
            pos_next = pos.at[i].set(tgt)                            # only i moves
            return _local_conn_score(pos_next, comm_r, sharp)[i]     # scalar
        s = jax.vmap(for_action)(jnp.arange(A))                      # (A,)
        s = jnp.where(action_valid[i], s, -jnp.inf)                  # forbid invalid
        if blocked is not None:
            s = jnp.where(blocked[i], -jnp.inf, s)                   # forbid collisions
        best = jnp.argmax(s).astype(jnp.int32)                      # best valid re-anchor
        # HOLD unless isolated: stay put while comfortably anchored, only move to
        # re-establish a neighbour when the current cell falls below the floor.
        isolated = deg_now[i] < hold_target
        return jnp.where(isolated, best, jnp.int32(stay))           # (,) int32

    return jax.vmap(score_agent)(jnp.arange(n))                      # (N,)


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
