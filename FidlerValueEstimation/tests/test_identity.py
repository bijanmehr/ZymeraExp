"""Tests for the agent-identity augmentation (fidler.identity).

`augment_with_id(X_node, node_mask, *, id_mode, id_dim, seed)` appends per-agent ID
features to the node-feature window tensor X_node (S,H,N,6) and returns the augmented
tensor plus the resulting `in_size`. Three modes:
  * "none"   : no-op, in_size stays 6.
  * "random" : id_dim extra features per agent, CONSTANT across the H window for a given
               (sample,agent) but random per agent and per sample; padded agents -> 0.
  * "index"  : ONE extra feature = node-axis index / N_max in [0,1]; padded agents -> 0.
"""
import numpy as np

from fidler import identity


def _toy(S=3, H=4, N=5, seed=0):
    """Random base node features (S,H,N,6) + a node_mask (S,N) with some padded agents."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((S, H, N, 6)).astype(np.float32)
    mask = np.ones((S, N), dtype=bool)
    # pad out a couple of trailing agents per row (varying count) to mimic pad_batch
    mask[0, 4:] = False           # row 0: last agent padded
    mask[1, 3:] = False           # row 1: last two agents padded
    return X, mask


# --------------------------------------------------------------------------------------
# none
# --------------------------------------------------------------------------------------
def test_none_is_noop():
    X, mask = _toy()
    out, in_size = identity.augment_with_id(X, mask, id_mode="none", id_dim=4, seed=1)
    assert in_size == 6
    assert out.shape == X.shape
    np.testing.assert_array_equal(out, X)
    # must not be a view that aliases / mutates the caller's array unexpectedly
    assert out.dtype == np.float32


# --------------------------------------------------------------------------------------
# random
# --------------------------------------------------------------------------------------
def test_random_shapes_and_in_size():
    X, mask = _toy()
    for id_dim in (1, 4, 7):
        out, in_size = identity.augment_with_id(X, mask, id_mode="random", id_dim=id_dim, seed=2)
        assert in_size == 6 + id_dim
        assert out.shape == (X.shape[0], X.shape[1], X.shape[2], 6 + id_dim)
        # the first 6 channels are exactly the original features
        np.testing.assert_array_equal(out[..., :6], X)
        assert out.dtype == np.float32


def test_random_constant_across_H():
    """For a given (sample, agent) the appended tag is identical at every H step."""
    X, mask = _toy(S=3, H=4, N=5)
    out, _ = identity.augment_with_id(X, mask, id_mode="random", id_dim=4, seed=3)
    tag = out[..., 6:]                                   # (S,H,N,id_dim)
    # every H slice equals the H=0 slice
    for h in range(1, tag.shape[1]):
        np.testing.assert_array_equal(tag[:, h], tag[:, 0])


def test_random_differs_across_agents():
    """Real agents within a sample get DIFFERENT tags (symmetry-breaking)."""
    X, mask = _toy(S=3, H=4, N=5)
    out, _ = identity.augment_with_id(X, mask, id_mode="random", id_dim=4, seed=4)
    tag0 = out[0, 0]                                     # (N, id_dim) for sample 0
    real = np.where(mask[0])[0]
    # pairwise distinct among real agents
    for a in range(len(real)):
        for b in range(a + 1, len(real)):
            assert not np.array_equal(tag0[real[a]], tag0[real[b]]), \
                f"agents {real[a]} and {real[b]} share a random tag"


def test_random_differs_across_samples():
    """The same agent index in two different samples gets different tags."""
    X, mask = _toy(S=3, H=4, N=5)
    out, _ = identity.augment_with_id(X, mask, id_mode="random", id_dim=4, seed=5)
    tag = out[..., 6:]
    # agent 0 (real in every row) differs between sample 0 and sample 2
    assert not np.array_equal(tag[0, 0, 0], tag[2, 0, 0])


def test_random_padded_agents_zero():
    X, mask = _toy(S=3, H=4, N=5)
    out, _ = identity.augment_with_id(X, mask, id_mode="random", id_dim=4, seed=6)
    tag = out[..., 6:]                                   # (S,H,N,id_dim)
    pad = ~mask                                          # (S,N)
    pad_full = np.broadcast_to(pad[:, None, :], tag.shape[:3])
    assert np.all(tag[pad_full] == 0.0)
    # and real agents are (almost surely) not all-zero
    real_full = np.broadcast_to(mask[:, None, :], tag.shape[:3])
    assert np.any(tag[real_full] != 0.0)


def test_random_deterministic_in_seed():
    X, mask = _toy()
    a, _ = identity.augment_with_id(X, mask, id_mode="random", id_dim=4, seed=123)
    b, _ = identity.augment_with_id(X, mask, id_mode="random", id_dim=4, seed=123)
    c, _ = identity.augment_with_id(X, mask, id_mode="random", id_dim=4, seed=999)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)


# --------------------------------------------------------------------------------------
# index
# --------------------------------------------------------------------------------------
def test_index_shapes_and_in_size():
    X, mask = _toy()
    out, in_size = identity.augment_with_id(X, mask, id_mode="index", id_dim=4, seed=7)
    # index appends exactly ONE feature regardless of id_dim
    assert in_size == 7
    assert out.shape == (X.shape[0], X.shape[1], X.shape[2], 7)
    np.testing.assert_array_equal(out[..., :6], X)
    assert out.dtype == np.float32


def test_index_value_is_normalized_position():
    """Real agent k gets feature value k / N_max, constant across H; in [0,1]."""
    X, mask = _toy(S=3, H=4, N=5)
    N = X.shape[2]
    out, _ = identity.augment_with_id(X, mask, id_mode="index", id_dim=4, seed=8)
    idx_feat = out[..., 6]                               # (S,H,N)
    # constant across H
    for h in range(1, idx_feat.shape[1]):
        np.testing.assert_array_equal(idx_feat[:, h], idx_feat[:, 0])
    # value for real agent k == k / N
    expected = np.arange(N, dtype=np.float32) / float(N)
    for s in range(X.shape[0]):
        real = np.where(mask[s])[0]
        np.testing.assert_allclose(idx_feat[s, 0, real], expected[real], rtol=0, atol=1e-6)
    # bounded
    assert idx_feat.min() >= 0.0 and idx_feat.max() <= 1.0


def test_index_padded_agents_zero():
    X, mask = _toy(S=3, H=4, N=5)
    out, _ = identity.augment_with_id(X, mask, id_mode="index", id_dim=4, seed=9)
    idx_feat = out[..., 6]                               # (S,H,N)
    pad_full = np.broadcast_to((~mask)[:, None, :], idx_feat.shape)
    assert np.all(idx_feat[pad_full] == 0.0)


# --------------------------------------------------------------------------------------
# misc
# --------------------------------------------------------------------------------------
def test_unknown_mode_raises():
    X, mask = _toy()
    try:
        identity.augment_with_id(X, mask, id_mode="bogus", id_dim=4, seed=0)
    except ValueError:
        return
    raise AssertionError("expected ValueError for unknown id_mode")
