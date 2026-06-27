import numpy as np
import jax.numpy as jnp
from fiedler.methods import power_iteration as pi
from fiedler import fiedler


def _path(n):
    a = np.eye(n, k=1, dtype=bool) | np.eye(n, k=-1, dtype=bool); np.fill_diagonal(a, True)
    return jnp.asarray(a)


def test_converges_to_true_lambda2_on_static_path():
    adj = _path(5)
    true = float(fiedler.true_lambda2(adj))
    est = float(pi.estimate(adj, n_rounds=400, eps=0.1, seed=0))
    assert abs(est - true) / true < 0.1

def test_error_decreases_with_more_rounds():
    adj = _path(5)
    true = float(fiedler.true_lambda2(adj))
    e_few = abs(float(pi.estimate(adj, n_rounds=20, eps=0.1, seed=0)) - true)
    e_many = abs(float(pi.estimate(adj, n_rounds=400, eps=0.1, seed=0)) - true)
    assert e_many < e_few
