import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fiedler.models_eqx import GRUEstimator, GCRNEstimator, ConfigurableGCRN


def _inputs(H, N, key):
    kx, ka = jax.random.split(key)
    x_node = jax.random.normal(kx, (H, N, 6))
    a = jax.random.bernoulli(ka, 0.5, (H, N, N))
    # symmetric with self-loops on the diagonal (as the real adjacency is)
    a = a | jnp.transpose(a, (0, 2, 1))
    eye = jnp.eye(N, dtype=bool)[None].repeat(H, 0)
    a = a | eye
    return x_node, a


@pytest.mark.parametrize("Model", [GRUEstimator, GCRNEstimator])
def test_returns_per_node_vector(Model):
    H, N = 5, 4
    model = Model(in_size=6, hidden=16, key=jax.random.PRNGKey(0))
    x_node, x_adj = _inputs(H, N, jax.random.PRNGKey(1))
    out = model(x_node, x_adj)
    assert out.shape == (N,)
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize("Model", [GRUEstimator, GCRNEstimator])
def test_size_invariance_same_params_two_N(Model):
    """Instantiate ONCE, call at N=4 and N=8 with the same params -> (N,), finite."""
    H = 5
    model = Model(in_size=6, hidden=16, key=jax.random.PRNGKey(2))
    x4, a4 = _inputs(H, 4, jax.random.PRNGKey(10))
    x8, a8 = _inputs(H, 8, jax.random.PRNGKey(11))
    out4 = model(x4, a4)
    out8 = model(x8, a8)
    assert out4.shape == (4,)
    assert out8.shape == (8,)
    assert np.all(np.isfinite(np.asarray(out4)))
    assert np.all(np.isfinite(np.asarray(out8)))


def test_gru_ignores_adjacency():
    H, N = 4, 5
    model = GRUEstimator(in_size=6, hidden=16, key=jax.random.PRNGKey(3))
    x_node, x_adj = _inputs(H, N, jax.random.PRNGKey(4))
    other_adj = jnp.zeros_like(x_adj)
    out_a = model(x_node, x_adj)
    out_b = model(x_node, other_adj)
    assert np.allclose(np.asarray(out_a), np.asarray(out_b))


def test_gcrn_uses_adjacency():
    """GCRN aggregates over neighbors, so changing the graph should change the output."""
    H, N = 4, 6
    model = GCRNEstimator(in_size=6, hidden=16, key=jax.random.PRNGKey(5))
    x_node, x_adj = _inputs(H, N, jax.random.PRNGKey(6))
    # a different connected graph (full graph with self loops)
    full = jnp.ones_like(x_adj)
    out_a = model(x_node, x_adj)
    out_b = model(x_node, full)
    assert not np.allclose(np.asarray(out_a), np.asarray(out_b))


# --------------------------------------------------------------------------------------
# ConfigurableGCRN
# --------------------------------------------------------------------------------------
def _inputs_pos(H, N, key):
    kx, ka, kp = jax.random.split(key, 3)
    x_node = jax.random.normal(kx, (H, N, 6))
    a = jax.random.bernoulli(ka, 0.5, (H, N, N))
    a = a | jnp.transpose(a, (0, 2, 1))
    eye = jnp.eye(N, dtype=bool)[None].repeat(H, 0)
    a = a | eye
    x_pos = jax.random.uniform(kp, (H, N, 2), minval=0.0, maxval=8.0)
    return x_node, a, x_pos


def test_configurable_gcrn_three_heads_shapes_and_finite():
    H, N = 5, 4
    model = ConfigurableGCRN(in_size=6, hidden=16, n_rounds=2, op="mean", content="value",
                             key=jax.random.PRNGKey(0))
    x_node, x_adj, x_pos = _inputs_pos(H, N, jax.random.PRNGKey(1))
    out = model(x_node, x_adj, x_pos)
    assert set(out.keys()) == {"logl2", "cflag", "logsig"}
    for k in ("logl2", "cflag", "logsig"):
        assert out[k].shape == (N,)
        assert np.all(np.isfinite(np.asarray(out[k])))


def test_configurable_gcrn_size_invariance_same_params():
    """Instantiate ONCE; call at N=4 and N=12 with identical params -> right shapes, finite."""
    H = 4
    model = ConfigurableGCRN(in_size=6, hidden=16, n_rounds=2, op="gcn", content="geom",
                             key=jax.random.PRNGKey(2))
    x4, a4, p4 = _inputs_pos(H, 4, jax.random.PRNGKey(10))
    x12, a12, p12 = _inputs_pos(H, 12, jax.random.PRNGKey(11))
    o4 = model(x4, a4, p4)
    o12 = model(x12, a12, p12)
    assert o4["logl2"].shape == (4,) and o12["logl2"].shape == (12,)
    for o in (o4, o12):
        for k in ("logl2", "cflag", "logsig"):
            assert np.all(np.isfinite(np.asarray(o[k])))


@pytest.mark.parametrize("op,content", [
    ("mean", "value"), ("attention", "learned"),
    ("multihead_attention", "value"), ("gated", "geom"), ("laplacian", "value"),
])
def test_configurable_gcrn_various_configs_run(op, content):
    H, N = 3, 6
    model = ConfigurableGCRN(in_size=6, hidden=16, n_rounds=1, op=op, content=content,
                             heads=4, key=jax.random.PRNGKey(7))
    x_node, x_adj, x_pos = _inputs_pos(H, N, jax.random.PRNGKey(8))
    out = model(x_node, x_adj, x_pos)
    assert out["logl2"].shape == (N,)
    assert np.all(np.isfinite(np.asarray(out["logl2"])))
    assert np.all(np.isfinite(np.asarray(out["cflag"])))
    assert np.all(np.isfinite(np.asarray(out["logsig"])))


def test_configurable_gcrn_uses_graph():
    H, N = 4, 6
    model = ConfigurableGCRN(in_size=6, hidden=16, n_rounds=2, op="mean", content="value",
                             key=jax.random.PRNGKey(5))
    x_node, x_adj, x_pos = _inputs_pos(H, N, jax.random.PRNGKey(6))
    full = jnp.ones_like(x_adj)
    o_a = model(x_node, x_adj, x_pos)
    o_b = model(x_node, full, x_pos)
    assert not np.allclose(np.asarray(o_a["logl2"]), np.asarray(o_b["logl2"]))
