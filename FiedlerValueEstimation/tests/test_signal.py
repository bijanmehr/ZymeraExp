"""Tests for the signal-strength (path-loss) module (fiedler.signal).

Three pieces:
  * `signal_strength(dist_over_r)`  -> smooth path-loss falloff s = exp(-3*x^2) in [0,1],
    ~1 at x=0 (close), ~0.05 at x=1 (=comm_r), monotone-decreasing.
  * `signal_weighted_adj(X_adj_bool, X_pos, comm_r)` -> FLOAT adjacency: each existing edge's
    1.0 replaced by signal_strength(dist/comm_r); non-edges stay 0; binary support preserved.
  * `augment_with_signal(X_node, X_adj_bool, X_pos, comm_r)` -> append ONE per-agent node
    feature = mean in-range-neighbor signal strength at the window's LAST step (0 if
    isolated / padded). Returns (X_node_aug, in_size=F+1).
"""
import numpy as np
import jax.numpy as jnp

from fiedler import signal


# --------------------------------------------------------------------------------------
# signal_strength: shape of the falloff
# --------------------------------------------------------------------------------------
def test_signal_strength_endpoints():
    """s(0) ~ 1 (close neighbor) and s(1) ~ 0.05 (at comm range)."""
    s0 = float(signal.signal_strength(jnp.asarray(0.0)))
    s1 = float(signal.signal_strength(jnp.asarray(1.0)))
    assert abs(s0 - 1.0) < 1e-6
    # exp(-3) = 0.049787...
    np.testing.assert_allclose(s1, np.exp(-3.0), rtol=0, atol=1e-6)
    assert 0.04 < s1 < 0.06


def test_signal_strength_in_unit_interval():
    """s stays in [0,1] across a wide sweep of normalized distances (incl. beyond comm_r)."""
    x = jnp.linspace(0.0, 3.0, 64)
    s = np.asarray(signal.signal_strength(x))
    assert np.all(s >= 0.0)
    assert np.all(s <= 1.0)


def test_signal_strength_monotone_decreasing():
    """s is strictly monotone-decreasing in distance."""
    x = jnp.linspace(0.0, 2.0, 50)
    s = np.asarray(signal.signal_strength(x))
    assert np.all(np.diff(s) < 0.0)


def test_signal_strength_vectorized_matches_elementwise():
    """Vectorized call equals the elementwise formula exp(-3 x^2)."""
    x = jnp.asarray([0.0, 0.25, 0.5, 0.75, 1.0, 1.5])
    s = np.asarray(signal.signal_strength(x))
    expected = np.exp(-3.0 * np.asarray(x) ** 2)
    np.testing.assert_allclose(s, expected, rtol=1e-6, atol=1e-7)


# --------------------------------------------------------------------------------------
# signal_weighted_adj: soft weights on the binary support
# --------------------------------------------------------------------------------------
def _line_graph(comm_r=5.0):
    """3 agents on a line at x=0, comm_r/2, comm_r; fully (bool) connected with self-loops."""
    X_pos = np.array([[0.0, 0.0], [comm_r / 2.0, 0.0], [comm_r, 0.0]], np.float32)[None]  # (1,3,2)
    X_adj = np.ones((1, 3, 3), bool)
    return X_adj, X_pos


def test_weighted_adj_preserves_support_and_is_float():
    """The weighted adjacency is float, with exactly the same nonzero pattern as the bool one."""
    comm_r = 5.0
    rng = np.random.default_rng(0)
    X_pos = rng.uniform(0.0, comm_r, size=(2, 6, 2)).astype(np.float32)
    d = np.linalg.norm(X_pos[:, :, None, :] - X_pos[:, None, :, :], axis=-1)
    X_adj = (d <= comm_r)
    X_adj = X_adj | np.transpose(X_adj, (0, 2, 1)) | np.eye(6, dtype=bool)[None]
    W = signal.signal_weighted_adj(X_adj, X_pos, comm_r)
    W = np.asarray(W)
    assert W.dtype == np.float32
    # same support (ignoring the diagonal, which is handled like elsewhere)
    off = ~np.eye(6, dtype=bool)[None]
    assert np.array_equal((W != 0) & off, X_adj.astype(bool) & off)
    # all weights are valid signal strengths in [0,1]
    assert np.all(W >= 0.0) and np.all(W <= 1.0)


def test_weighted_adj_values_match_signal_strength():
    """Edge (i,j) weight equals signal_strength(dist_ij/comm_r) on a known line graph."""
    comm_r = 5.0
    X_adj, X_pos = _line_graph(comm_r)
    W = np.asarray(signal.signal_weighted_adj(X_adj, X_pos, comm_r))[0]   # (3,3)
    pos = X_pos[0]
    for i in range(3):
        for j in range(3):
            if i == j:
                continue
            dist = np.linalg.norm(pos[i] - pos[j])
            expected = float(np.exp(-3.0 * (dist / comm_r) ** 2))
            np.testing.assert_allclose(W[i, j], expected, rtol=1e-6, atol=1e-7)
    # the close pair (dist=comm_r/2 -> x=0.5) is strong; the far pair (x=1.0) is weak (~0.05)
    assert W[0, 1] > W[0, 2]
    np.testing.assert_allclose(W[0, 2], np.exp(-3.0), rtol=0, atol=1e-6)


def test_weighted_adj_non_edges_stay_zero():
    """A non-edge (bool False) stays exactly 0 even if the two agents are close."""
    comm_r = 5.0
    X_pos = np.array([[0.0, 0.0], [1.0, 0.0]], np.float32)[None]          # close together
    X_adj = np.array([[[1, 0], [0, 1]]], bool)                           # but NOT connected
    W = np.asarray(signal.signal_weighted_adj(X_adj, X_pos, comm_r))[0]
    assert W[0, 1] == 0.0
    assert W[1, 0] == 0.0


def test_weighted_adj_shape_preserved_with_window_axes():
    """Works on (...,N,N) with leading window axes (S,H,N,N)."""
    comm_r = 5.0
    S, H, N = 2, 4, 5
    rng = np.random.default_rng(1)
    X_pos = rng.uniform(0.0, comm_r, size=(S, H, N, 2)).astype(np.float32)
    d = np.linalg.norm(X_pos[..., :, None, :] - X_pos[..., None, :, :], axis=-1)
    X_adj = (d <= comm_r) | np.eye(N, dtype=bool)
    W = signal.signal_weighted_adj(X_adj, X_pos, comm_r)
    assert np.asarray(W).shape == (S, H, N, N)
    assert np.asarray(W).dtype == np.float32


# --------------------------------------------------------------------------------------
# augment_with_signal: the per-agent link-quality node feature
# --------------------------------------------------------------------------------------
def _toy(S=3, H=4, N=5, F=6, comm_r=5.0, seed=0):
    rng = np.random.default_rng(seed)
    X_node = rng.standard_normal((S, H, N, F)).astype(np.float32)
    X_pos = rng.uniform(0.0, comm_r, size=(S, H, N, 2)).astype(np.float32)
    X_adj = np.zeros((S, H, N, N), dtype=bool)
    for s in range(S):
        for h in range(H):
            d = np.linalg.norm(X_pos[s, h][:, None, :] - X_pos[s, h][None, :, :], axis=-1)
            a = d <= comm_r
            X_adj[s, h] = a | a.T | np.eye(N, dtype=bool)
    return X_node, X_adj, X_pos


def test_signal_feature_shapes_and_in_size():
    X_node, X_adj, X_pos = _toy(F=6)
    out, in_size = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=5.0)
    assert in_size == 7
    assert out.shape == (X_node.shape[0], X_node.shape[1], X_node.shape[2], 7)
    np.testing.assert_array_equal(out[..., :6], X_node)   # originals preserved
    assert out.dtype == np.float32


def test_signal_feature_in_size_tracks_width():
    X_node, X_adj, X_pos = _toy(F=10)
    out, in_size = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=5.0)
    assert in_size == 11
    assert out.shape[-1] == 11


def test_signal_feature_in_unit_interval():
    X_node, X_adj, X_pos = _toy()
    out, _ = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=5.0)
    feat = out[..., -1]
    assert np.all(feat >= 0.0) and np.all(feat <= 1.0)


def test_signal_feature_constant_across_H():
    """The feature is computed from the LAST step -> constant across the H window."""
    X_node, X_adj, X_pos = _toy(S=2, H=4, N=5)
    out, _ = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=5.0)
    feat = out[..., -1]                                   # (S,H,N)
    for h in range(1, feat.shape[1]):
        np.testing.assert_array_equal(feat[:, h], feat[:, 0])


def test_signal_feature_very_close_neighbor_near_one():
    """A single very-close neighbor -> mean signal strength ~ 1."""
    comm_r = 5.0
    S, H, N, F = 1, 2, 2, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 1, 0] = [0.0, 0.0]
    X_pos[0, 1, 1] = [0.01 * comm_r, 0.0]                 # essentially on top of each other
    X_adj = np.zeros((S, H, N, N), bool)
    X_adj[0, 1] = np.array([[True, True], [True, True]])
    out, _ = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, -1, :, -1]
    np.testing.assert_allclose(feat, [1.0, 1.0], rtol=0, atol=1e-3)


def test_signal_feature_near_comm_r_is_small():
    """A single neighbor near comm_r -> mean signal strength ~ exp(-3) ~ 0.05 (weak link)."""
    comm_r = 5.0
    S, H, N, F = 1, 2, 2, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 1, 0] = [0.0, 0.0]
    X_pos[0, 1, 1] = [0.999 * comm_r, 0.0]
    X_adj = np.zeros((S, H, N, N), bool)
    X_adj[0, 1] = np.array([[True, True], [True, True]])
    out, _ = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, -1, :, -1]
    np.testing.assert_allclose(feat, [np.exp(-3.0), np.exp(-3.0)], rtol=0, atol=2e-3)


def test_signal_feature_isolated_node_is_zero():
    """An agent with no in-range neighbor (only a self-loop) gets feature 0."""
    comm_r = 5.0
    S, H, N, F = 1, 1, 3, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 0] = [[0.0, 0.0], [1.0, 0.0], [100.0, 100.0]]
    X_adj = np.zeros((S, H, N, N), bool)
    X_adj[0, 0] = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=bool)
    out, _ = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, 0, :, -1]
    assert feat[2] == 0.0                                 # isolated agent -> 0


def test_signal_feature_self_loops_do_not_count():
    """A self-loop (distance 0, would give signal 1) must not enter the mean."""
    comm_r = 5.0
    S, H, N, F = 1, 1, 2, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 0] = [[0.0, 0.0], [0.999 * comm_r, 0.0]]
    X_adj = np.array([[[True, True], [True, True]]], dtype=bool)[:, None]   # (1,1,2,2) w/ self
    out, _ = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, 0, :, -1]
    # the ONLY real neighbor is near comm_r -> ~exp(-3); a counted self-loop would push it to ~0.5
    np.testing.assert_allclose(feat, [np.exp(-3.0), np.exp(-3.0)], rtol=0, atol=2e-3)


def test_signal_feature_mean_over_neighbors():
    """With several neighbors, the feature is the MEAN of their signal strengths."""
    comm_r = 10.0
    S, H, N, F = 1, 1, 3, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    # agent 0 has neighbors at distance 3 and 7
    X_pos[0, 0] = [[0.0, 0.0], [3.0, 0.0], [7.0, 0.0]]
    X_adj = np.ones((S, H, N, N), bool)
    out, _ = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, 0, :, -1]
    s3 = np.exp(-3.0 * (3.0 / comm_r) ** 2)
    s7 = np.exp(-3.0 * (7.0 / comm_r) ** 2)
    s4 = np.exp(-3.0 * (4.0 / comm_r) ** 2)
    expected = [0.5 * (s3 + s7), 0.5 * (s3 + s4), 0.5 * (s7 + s4)]
    np.testing.assert_allclose(feat, expected, rtol=1e-5, atol=1e-6)


def test_signal_feature_padded_node_is_zero():
    """A padded agent (no edges, zero position) reads as isolated -> 0."""
    comm_r = 5.0
    S, H, N, F = 1, 1, 3, 6
    X_node = np.zeros((S, H, N, F), np.float32)
    X_pos = np.zeros((S, H, N, 2), np.float32)
    X_pos[0, 0] = [[0.0, 0.0], [1.0, 0.0], [0.0, 0.0]]
    X_adj = np.zeros((S, H, N, N), bool)
    X_adj[0, 0] = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 0]], dtype=bool)
    out, _ = signal.augment_with_signal(X_node, X_adj, X_pos, comm_r=comm_r)
    feat = out[0, 0, :, -1]
    assert feat[2] == 0.0
