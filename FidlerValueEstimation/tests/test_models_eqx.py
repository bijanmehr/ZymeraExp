import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fidler.models_eqx import GRUEstimator, GCRNEstimator


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
