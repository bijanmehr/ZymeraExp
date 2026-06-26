import numpy as np
from fidler import dataset


def _fake(E=3, T1=7, N=4):
    """Return (features (E,T1,N,6), adjacency (E,T1,N,N) bool, lambda2 (E,T1))."""
    rng = np.random.default_rng(0)
    feats = rng.standard_normal((E, T1, N, 6)).astype(np.float32)
    adj = rng.random((E, T1, N, N)) > 0.5
    lam = rng.random((E, T1)).astype(np.float32)
    return feats, adj, lam


def test_make_windows_shapes_and_dtype():
    E, T1, N, H = 3, 7, 4, 3
    feats, _, lam = _fake(E, T1, N)
    X_node, y = dataset.make_windows(feats, lam, H)
    S = E * (T1 - H + 1)
    assert X_node.shape == (S, H, N, 6)
    assert y.shape == (S,)
    assert X_node.dtype == np.float32
    assert y.dtype == np.float32


def test_make_windows_target_is_last_step_lambda2():
    E, T1, N, H = 3, 7, 4, 3
    feats, _, lam = _fake(E, T1, N)
    _, y = dataset.make_windows(feats, lam, H)
    # target = lambda2 at each window's LAST step == lam[:, H-1:] flattened (row-major over episodes)
    expected = lam[:, H - 1:].reshape(-1)
    assert np.allclose(y, expected)


def test_make_windows_h1_keeps_all_steps():
    E, T1, N, H = 2, 5, 4, 1
    feats, _, lam = _fake(E, T1, N)
    X_node, y = dataset.make_windows(feats, lam, H)
    assert X_node.shape == (E * T1, 1, N, 6)
    assert np.allclose(y, lam.reshape(-1))


def test_make_windows_content_matches_source_slice():
    # the first window of episode 0 should be feats[0, 0:H]
    E, T1, N, H = 2, 6, 4, 3
    feats, _, lam = _fake(E, T1, N)
    X_node, _ = dataset.make_windows(feats, lam, H)
    assert np.allclose(X_node[0], feats[0, 0:H])
    # last window of episode 0 (index T1-H) should be feats[0, T1-H : T1]
    n_per_ep = T1 - H + 1
    assert np.allclose(X_node[n_per_ep - 1], feats[0, T1 - H:T1])
    # first window of episode 1 starts right after episode 0's windows
    assert np.allclose(X_node[n_per_ep], feats[1, 0:H])


def test_make_adj_windows_same_order_and_shape():
    E, T1, N, H = 3, 7, 4, 3
    feats, adj, lam = _fake(E, T1, N)
    X_adj = dataset.make_adj_windows(adj, H)
    X_node, _ = dataset.make_windows(feats, lam, H)
    S = E * (T1 - H + 1)
    assert X_adj.shape == (S, H, N, N)
    assert X_adj.dtype == bool
    # same window order: adj window 0 == adj[0, 0:H]
    assert np.array_equal(X_adj[0], adj[0, 0:H])
    n_per_ep = T1 - H + 1
    assert np.array_equal(X_adj[n_per_ep], adj[1, 0:H])
    # sample count matches make_windows
    assert X_adj.shape[0] == X_node.shape[0]


def test_train_val_split_sizes_and_disjoint():
    tr, va = dataset.train_val_split(100, val_frac=0.2, seed=0)
    assert len(va) == 20
    assert len(tr) == 80
    assert len(set(tr.tolist()) & set(va.tolist())) == 0
    assert sorted(tr.tolist() + va.tolist()) == list(range(100))
    assert tr.dtype.kind == "i" and va.dtype.kind == "i"


def test_train_val_split_deterministic_with_seed():
    tr1, va1 = dataset.train_val_split(50, val_frac=0.2, seed=7)
    tr2, va2 = dataset.train_val_split(50, val_frac=0.2, seed=7)
    assert np.array_equal(tr1, tr2) and np.array_equal(va1, va2)
    tr3, _ = dataset.train_val_split(50, val_frac=0.2, seed=8)
    assert not np.array_equal(tr1, tr3)
