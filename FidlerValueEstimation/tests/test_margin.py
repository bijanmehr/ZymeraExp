"""Tests for the connectivity-margin node-feature augmentation (fidler.margin).

`augment_with_margin(X_node, X_adj, X_pos, comm_r)` appends ONE per-agent feature to the
node-feature window tensor X_node (S,H,N,F): the agent's MAX in-range-neighbor distance
divided by comm_r -- its closest-to-breaking link (~1 means about to disconnect). The
feature is computed from the window's LAST step and is 0 for isolated / padded agents.
Returns (X_node_aug (S,H,N,F+1), in_size = F+1).
"""
import numpy as np

from fidler import margin


def _toy(S=3, H=4, N=5, F=6, comm_r=5.0, seed=0):
    """Random base node features + adjacency/positions windows for a tiny batch."""
    rng = np.random.default_rng(seed)
    X_node = rng.standard_normal((S, H, N, F)).astype(np.float32)
    X_pos = rng.uniform(0.0, comm_r, size=(S, H, N, 2)).astype(np.float32)
    # symmetric adjacency per (S,H) with self-loops, derived loosely from positions
    X_adj = np.zeros((S, H, N, N), dtype=bool)
    for s in range(S):
        for h in range(H):
            d = np.linalg.norm(X_pos[s, h][:, None, :] - X_pos[s, h][None, :, :], axis=-1)
            a = d <= comm_r
            X_adj[s, h] = a | a.T | np.eye(N, dtype=bool)
    return X_node, X_adj, X_pos


# --------------------------------------------------------------------------------------
# shapes / in_size
# --------------------------------------------------------------------------------------
def test_shapes_and_in_size():
    X_node, X_adj, X_pos = _toy(F=6)
    out, in_size = margin.augment_with_margin(X_node, X_adj, X_pos, comm_r=5.0)
    assert in_size == 7
    assert out.shape == (X_node.shape[0], X_node.shape[1], X_node.shape[2], 7)
    # original 6 channels preserved exactly
    np.testing.assert_array_equal(out[..., :6], X_node)
    assert out.dtype == np.float32


def test_in_size_tracks_input_width():
    """The appended feature is always exactly one, whatever the input width."""
    X_node, X_adj, X_pos = _toy(F=10)
    out, in_size = margin.augment_with_margin(X_node, X_adj, X_pos, comm_r=5.0)
    assert in_size == 11
    assert out.shape[-1] == 11


def test_constant_across_H():
    """The appended margin is computed from the LAST step -> constant across the H window."""
    X_node, X_adj, X_pos = _toy(S=2, H=4, N=5)
    out, _ = margin.augment_with_margin(X_node, X_adj, X_pos, comm_r=5.0)
    feat = out[..., -1]                                   # (S,H,N)
    for h in range(1, feat.shape[1]):
        np.testing.assert_array_equal(feat[:, h], feat[:, 0])


# --------------------------------------------------------------------------------------
# semantics: near-comm_r neighbor -> ~1 ; isolated -> 0
# --------------------------------------------------------------------------------------
def test_near_comm_r_neighbor_gives_value_near_one():
    """Two agents almost exactly comm_r apart -> margin ~ 1 (about to break)."""
    comm_r = 5.0
    S, H, N, F = 1, 2, 2, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    # last step (h=1): the two agents are 0.99*comm_r apart on the x-axis
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 1, 0] = [0.0, 0.0]
    X_pos[0, 1, 1] = [0.99 * comm_r, 0.0]
    X_adj = np.zeros((S, H, N, N), bool)
    X_adj[0, 1] = np.array([[True, True], [True, True]])  # both connected (with self-loop)
    out, _ = margin.augment_with_margin(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, -1, :, -1]                              # both agents' margin at last step
    np.testing.assert_allclose(feat, [0.99, 0.99], rtol=0, atol=1e-5)


def test_isolated_node_gets_zero():
    """An agent with no in-range neighbor (only a self-loop) gets margin 0."""
    comm_r = 5.0
    S, H, N, F = 1, 1, 3, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 0] = [[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]]  # agent 2 far away
    X_adj = np.zeros((S, H, N, N), bool)
    # agents 0,1 connected; agent 2 only self-loop (isolated)
    X_adj[0, 0] = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
    out, _ = margin.augment_with_margin(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, 0, :, -1]
    assert feat[2] == 0.0                                  # isolated agent -> 0
    # agents 0 and 1 are distance 1 apart -> margin 1/comm_r = 0.2
    np.testing.assert_allclose(feat[:2], [0.2, 0.2], rtol=0, atol=1e-6)


def test_max_over_neighbors_is_the_weakest_link():
    """With several neighbors, the feature is the LARGEST neighbor distance / comm_r."""
    comm_r = 10.0
    S, H, N, F = 1, 1, 3, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    # agent 0 has neighbors at distance 3 (agent 1) and 7 (agent 2): max = 7
    X_pos[0, 0] = [[0.0, 0.0], [3.0, 0.0], [7.0, 0.0]]
    X_adj = np.ones((S, H, N, N), bool)                   # fully connected
    out, _ = margin.augment_with_margin(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, 0, :, -1]
    # agent 0: max(3,7)/10 = 0.7 ; agent 1: max(3,4)/10=0.4 ; agent 2: max(7,4)/10=0.7
    np.testing.assert_allclose(feat, [0.7, 0.4, 0.7], rtol=0, atol=1e-6)


def test_self_loops_do_not_count():
    """A self-loop on the diagonal must not register as a (distance-0) neighbor link."""
    comm_r = 5.0
    S, H, N, F = 1, 1, 2, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 0] = [[0.0, 0.0], [2.0, 0.0]]
    X_adj = np.array([[[True, True], [True, True]]], dtype=bool)[:, None]  # (1,1,2,2) w/ self
    out, _ = margin.augment_with_margin(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, 0, :, -1]
    # both have one real neighbor at distance 2 -> 2/5 = 0.4 (NOT 0 from the self-loop)
    np.testing.assert_allclose(feat, [0.4, 0.4], rtol=0, atol=1e-6)


def test_padded_node_via_isolation_is_zero():
    """A padded agent (no edges in adj, zero position) reads as isolated -> 0."""
    comm_r = 5.0
    S, H, N, F = 1, 1, 3, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 0] = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]    # agent 2 padded (zeros)
    # padded agent 2 has NO edges (all-zero row/col, no self-loop either after padding)
    X_adj = np.zeros((S, H, N, N), bool)
    X_adj[0, 0] = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    out, _ = margin.augment_with_margin(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, 0, :, -1]
    assert feat[2] == 0.0
