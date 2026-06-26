"""Tests for the message-design space (fidler.messages).

Rigorous on SIZE-INVARIANCE: instantiate MessageParams ONCE and call aggregate at
N=4 and N=12 with the SAME params for every (op, content) combo -> (N, hidden), finite.
Only `sum` is allowed to break size-transfer (it is the deliberate ablation).
"""
import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fidler import messages

OPS = ["mean", "gcn", "max", "sum", "attention", "multihead_attention", "gated", "laplacian"]
CONTENTS = ["value", "learned", "geom"]
HIDDEN = 8
HEADS = 4
COMM_R = 5.0


def _graph(N, key):
    """Symmetric in-range graph with self-loops on the diagonal + random positions."""
    ka, kz, kp = jax.random.split(key, 3)
    a = jax.random.bernoulli(ka, 0.55, (N, N))
    a = a | a.T
    a = a | jnp.eye(N, dtype=bool)
    z = jax.random.normal(kz, (N, HIDDEN))
    pos = jax.random.uniform(kp, (N, 2), minval=0.0, maxval=8.0)
    return z, a, pos


def _params(op, content, key):
    return messages.MessageParams(hidden=HIDDEN, heads=HEADS, content=content, op=op, key=key)


@pytest.mark.parametrize("op,content", list(itertools.product(OPS, CONTENTS)))
def test_returns_node_hidden_and_finite(op, content):
    N = 4
    z, a, pos = _graph(N, jax.random.PRNGKey(1))
    p = _params(op, content, jax.random.PRNGKey(0))
    out = messages.aggregate(z, a, pos, op=op, content=content, params=p,
                             key=jax.random.PRNGKey(2), dropedge=0.0)
    assert out.shape == (N, HIDDEN)
    assert np.all(np.isfinite(np.asarray(out)))


@pytest.mark.parametrize("op,content", list(itertools.product(OPS, CONTENTS)))
def test_size_invariance_same_params_two_N(op, content):
    """SAME params instance run at N=4 and N=12 -> correct shapes, finite (all ops)."""
    p = _params(op, content, jax.random.PRNGKey(0))
    z4, a4, pos4 = _graph(4, jax.random.PRNGKey(10))
    z12, a12, pos12 = _graph(12, jax.random.PRNGKey(11))
    out4 = messages.aggregate(z4, a4, pos4, op=op, content=content, params=p,
                              key=jax.random.PRNGKey(20), dropedge=0.0)
    out12 = messages.aggregate(z12, a12, pos12, op=op, content=content, params=p,
                               key=jax.random.PRNGKey(21), dropedge=0.0)
    assert out4.shape == (4, HIDDEN)
    assert out12.shape == (12, HIDDEN)
    assert np.all(np.isfinite(np.asarray(out4)))
    assert np.all(np.isfinite(np.asarray(out12)))


def test_sum_and_mean_differ():
    """sum (ablation) must NOT equal mean on a multi-degree graph."""
    N = 8
    z, a, pos = _graph(N, jax.random.PRNGKey(3))
    p_mean = _params("mean", "value", jax.random.PRNGKey(0))
    p_sum = _params("sum", "value", jax.random.PRNGKey(0))
    out_mean = messages.aggregate(z, a, pos, op="mean", content="value", params=p_mean,
                                  key=jax.random.PRNGKey(0), dropedge=0.0)
    out_sum = messages.aggregate(z, a, pos, op="sum", content="value", params=p_sum,
                                 key=jax.random.PRNGKey(0), dropedge=0.0)
    assert not np.allclose(np.asarray(out_mean), np.asarray(out_sum))


def test_attention_rows_sum_to_one_over_neighbors():
    """The attention weight matrix rows (over in-neighbors) should sum to ~1."""
    N = 6
    z, a, pos = _graph(N, jax.random.PRNGKey(4))
    p = _params("attention", "value", jax.random.PRNGKey(0))
    w = messages.attention_weights(z, a, p)        # (N, N) row-normalized over neighbors
    assert w.shape == (N, N)
    rs = np.asarray(w.sum(-1))
    assert np.allclose(rs, 1.0, atol=1e-5)
    # weight is zero where there is no edge (after stripping self-loops)
    a_hat = np.asarray(a) & ~np.eye(N, dtype=bool)
    assert np.all(np.asarray(w)[~a_hat] == 0.0)


def test_dropedge_changes_output_and_zero_is_noop():
    N = 10
    z, a, pos = _graph(N, jax.random.PRNGKey(7))
    p = _params("mean", "value", jax.random.PRNGKey(0))
    base = messages.aggregate(z, a, pos, op="mean", content="value", params=p,
                              key=jax.random.PRNGKey(0), dropedge=0.0)
    base2 = messages.aggregate(z, a, pos, op="mean", content="value", params=p,
                               key=jax.random.PRNGKey(999), dropedge=0.0)
    # dropedge=0 -> key irrelevant
    assert np.allclose(np.asarray(base), np.asarray(base2))
    dropped = messages.aggregate(z, a, pos, op="mean", content="value", params=p,
                                 key=jax.random.PRNGKey(5), dropedge=0.8)
    assert not np.allclose(np.asarray(base), np.asarray(dropped))


def test_sum_scales_with_n_but_mean_stable():
    """On a FULL graph with identical node vectors, mean is N-stable, sum grows with N.

    This is the size-transfer rationale: sum aggregation magnitude tracks degree/N.
    """
    def full_const(N):
        a = jnp.ones((N, N), dtype=bool)
        z = jnp.ones((N, HIDDEN))
        pos = jnp.zeros((N, 2))
        return z, a, pos

    p_mean = _params("mean", "value", jax.random.PRNGKey(0))
    p_sum = _params("sum", "value", jax.random.PRNGKey(0))
    z4, a4, p4 = full_const(4)
    z12, a12, p12 = full_const(12)
    m4 = messages.aggregate(z4, a4, p4, op="mean", content="value", params=p_mean,
                            key=jax.random.PRNGKey(0), dropedge=0.0)
    m12 = messages.aggregate(z12, a12, p12, op="mean", content="value", params=p_mean,
                             key=jax.random.PRNGKey(0), dropedge=0.0)
    s4 = messages.aggregate(z4, a4, p4, op="sum", content="value", params=p_sum,
                            key=jax.random.PRNGKey(0), dropedge=0.0)
    s12 = messages.aggregate(z12, a12, p12, op="sum", content="value", params=p_sum,
                             key=jax.random.PRNGKey(0), dropedge=0.0)
    # mean of identical unit vectors over neighbors is ~1 regardless of N (per-element)
    assert np.allclose(np.asarray(m4), 1.0, atol=1e-5)
    assert np.allclose(np.asarray(m12), 1.0, atol=1e-5)
    # sum grows with the neighbor count (3 vs 11 neighbors after self-loop strip)
    assert np.asarray(s12).mean() > 2.0 * np.asarray(s4).mean()
