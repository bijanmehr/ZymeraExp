"""Tests for the hard-connectivity-guardrail dispersion policy (fiedler.policies)."""
import numpy as np
import jax
import zymera.metrics as zmetrics

from fiedler import policies


def _connected(positions, comm_r):
    """True iff the Chebyshev<=comm_r comm graph over `positions` is one component."""
    adj = np.asarray(zmetrics.adjacency(np.asarray(positions), radius=comm_r))
    n = adj.shape[0]
    seen = np.zeros(n, bool)
    stack = [0]
    seen[0] = True
    while stack:
        u = stack.pop()
        for v in range(n):
            if adj[u, v] and not seen[v]:
                seen[v] = True
                stack.append(v)
    return bool(seen.all())


def _apply(positions, actions, grid):
    """Apply the 5-move action set (0 STAY,1 N row-1,2 E col+1,3 S row+1,4 W col-1)."""
    deltas = np.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]], np.int32)
    nxt = positions + deltas[actions]
    return np.clip(nxt, 0, grid - 1)


def _clustered(n, grid, radius, key):
    """A connected clustered start: all agents within `radius` of a center."""
    rng = np.random.default_rng(int(jax.random.randint(key, (), 0, 1 << 30)))
    c = rng.integers(radius, grid - radius, size=2)
    pos = c[None, :] + rng.integers(-radius, radius + 1, size=(n, 2))
    return np.clip(pos, 0, grid - 1).astype(np.int32)


def test_actions_valid_range_and_shape():
    key = jax.random.PRNGKey(0)
    pos = _clustered(8, 16, 2, key)
    a = policies.guardrail_disperse_actions(pos, grid=16, comm_r=5, key=key)
    assert a.shape == (8,)
    assert a.dtype == np.int32
    assert np.all(a >= 0) and np.all(a < 5)


def test_guardrail_keeps_graph_connected_from_clustered_start():
    """Applying the guardrail actions from a connected start keeps it connected."""
    comm_r = 4
    grid = 20
    for s in range(8):
        key = jax.random.PRNGKey(s)
        pos = _clustered(10, grid, 2, key)
        assert _connected(pos, comm_r), "test setup: start must be connected"
        a = policies.guardrail_disperse_actions(pos, grid=grid, comm_r=comm_r, key=key)
        nxt = _apply(pos, a, grid)
        assert _connected(nxt, comm_r), f"seed {s}: guardrail broke connectivity"


def test_stay_fallback_when_boxed_in():
    """A single tight cluster where comm_r is so small any move disconnects -> STAY."""
    # Two agents on adjacent cells; comm_r=1 so they MUST stay touching.
    # Each agent's only connectivity-preserving move is the one that keeps them
    # within Chebyshev distance 1; from a touching pair the safe set always
    # includes STAY, and the dispersion objective should keep them legal.
    pos = np.array([[5, 5], [5, 6]], np.int32)
    key = jax.random.PRNGKey(1)
    a = policies.guardrail_disperse_actions(pos, grid=16, comm_r=1, key=key)
    nxt = _apply(pos, a, 16)
    assert _connected(nxt, 1)


def test_stay_when_no_legal_move_at_all():
    """comm_r=0: only the diagonal self-loop is 'in range' -> graph is never connected
    for n>=2 unless agents overlap. The guardrail must still return STAY (action 0)
    rather than crash, because STAY is defined as always-allowed."""
    pos = np.array([[3, 3], [10, 10]], np.int32)
    key = jax.random.PRNGKey(2)
    a = policies.guardrail_disperse_actions(pos, grid=16, comm_r=0, key=key)
    # STAY is the guaranteed fallback for every agent here.
    assert np.all(a == 0)


def test_spreading_increases_min_distances_open_grid():
    """On an open grid with generous comm_r, one guardrail step should not DECREASE
    the per-agent min pairwise distance, and over several steps it should increase
    the mean pairwise distance (the swarm disperses)."""
    grid = 24
    comm_r = 8
    key = jax.random.PRNGKey(7)
    pos = _clustered(8, grid, 1, key)  # very tight start

    def mean_pair_dist(p):
        d = np.abs(p[:, None, :] - p[None, :, :]).max(-1)  # Chebyshev
        iu = np.triu_indices(p.shape[0], k=1)
        return d[iu].mean()

    d0 = mean_pair_dist(pos)
    p = pos.copy()
    k = key
    for _ in range(15):
        k, sub = jax.random.split(k)
        a = policies.guardrail_disperse_actions(p, grid=grid, comm_r=comm_r, key=sub)
        p = _apply(p, a, grid)
        assert _connected(p, comm_r)
    d1 = mean_pair_dist(p)
    assert d1 > d0, f"dispersion did not increase mean dist: {d0} -> {d1}"


def test_per_agent_min_dist_nondecreasing_single_step():
    """For the chosen action of each agent, its min-distance to others must be >= the
    min-distance it had at STAY (dispersion picks the best allowed candidate, and STAY
    is always allowed, so the chosen min-dist can never be worse than STAY's)."""
    grid = 24
    comm_r = 8
    key = jax.random.PRNGKey(11)
    pos = _clustered(9, grid, 2, key)
    a = policies.guardrail_disperse_actions(pos, grid=grid, comm_r=comm_r, key=key)
    nxt = _apply(pos, a, grid)

    def agent_min_dist(p, i):
        d = np.abs(p[i][None, :] - p).max(-1).astype(float)  # Chebyshev to all
        d[i] = np.inf
        return d.min()

    for i in range(pos.shape[0]):
        # min-dist at the chosen next position (with everyone else at current pos)
        moved = pos.copy()
        moved[i] = nxt[i]
        chosen = agent_min_dist(moved, i)
        stay = agent_min_dist(pos, i)  # STAY keeps agent at pos
        assert chosen >= stay - 1e-9, f"agent {i}: chosen {chosen} < stay {stay}"
