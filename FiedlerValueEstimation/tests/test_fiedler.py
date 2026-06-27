import numpy as np
import jax.numpy as jnp
from fiedler import fiedler


def _complete(n):
    a = np.ones((n, n), dtype=bool); np.fill_diagonal(a, True); return jnp.asarray(a)

def _path(n):
    a = np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool); np.fill_diagonal(a, True); return jnp.asarray(a)

def _two_components(n):  # two disjoint edges → disconnected
    a = np.zeros((n, n), dtype=bool); a[0, 1] = a[1, 0] = True; a[2, 3] = a[3, 2] = True
    np.fill_diagonal(a, True); return jnp.asarray(a)


def test_complete_graph_lambda2_is_n():
    assert abs(float(fiedler.true_lambda2(_complete(5))) - 5.0) < 1e-4

def test_path_graph_lambda2_known():
    n = 4
    expected = 2 * (1 - np.cos(np.pi / n))
    assert abs(float(fiedler.true_lambda2(_path(n))) - expected) < 1e-4

def test_disconnected_lambda2_zero_and_flag_false():
    a = _two_components(4)
    assert float(fiedler.true_lambda2(a)) < 1e-6
    assert bool(fiedler.connected_flag(a)) is False

def test_connected_flag_true_for_path():
    assert bool(fiedler.connected_flag(_path(4))) is True
