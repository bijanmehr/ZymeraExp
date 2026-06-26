import jax
import jax.numpy as jnp
import numpy as np

from fidler.models_eqx import GRUEstimator
from fidler import train_eqx


def _synthetic(S=64, H=3, N=4, seed=0):
    """A learnable mapping: lambda2 target correlates with mean of feature[:, :, 0]."""
    rng = np.random.default_rng(seed)
    X_node = rng.standard_normal((S, H, N, 6)).astype(np.float32)
    X_adj = (rng.random((S, H, N, N)) > 0.5)
    # target depends on the windowed features so the net can fit it; keep positive
    signal = X_node[:, -1, :, 0].mean(axis=1)            # (S,)
    y = (0.5 + 0.4 * (signal - signal.min())).astype(np.float32)
    return X_node, X_adj, y


def test_loss_decreases_on_synthetic_set():
    X_node, X_adj, y = _synthetic()
    model = GRUEstimator(in_size=6, hidden=16, key=jax.random.PRNGKey(0))

    pred0 = train_eqx.predict(model, X_node, X_adj)       # (S,N) linear
    pred0_log = jnp.log(jnp.asarray(pred0) + 1e-6)
    y_log = jnp.log(jnp.asarray(y)[:, None] + 1e-6)
    loss_before = float(jnp.mean((pred0_log - y_log) ** 2))

    trained, val_acc = train_eqx.train(model, X_node, X_adj, y, steps=300, lr=3e-3,
                                       batch=32, val_frac=0.25, seed=0)

    pred1 = train_eqx.predict(trained, X_node, X_adj)
    pred1_log = jnp.log(jnp.asarray(pred1) + 1e-6)
    loss_after = float(jnp.mean((pred1_log - y_log) ** 2))

    assert loss_after < 0.5 * loss_before
    assert 0.0 <= val_acc <= 1.0


def test_predict_returns_linear_positive_shape():
    X_node, X_adj, y = _synthetic(S=16, H=2, N=5)
    model = GRUEstimator(in_size=6, hidden=8, key=jax.random.PRNGKey(1))
    pred = train_eqx.predict(model, X_node, X_adj)
    assert pred.shape == (16, 5)
    assert np.all(np.asarray(pred) > 0.0)                 # exp(log-pred) is positive


def test_huber_loss_is_scalar_nonnegative():
    X_node, X_adj, y = _synthetic(S=8, H=2, N=4)
    model = GRUEstimator(in_size=6, hidden=8, key=jax.random.PRNGKey(2))
    loss = train_eqx.loss_fn(model, jnp.asarray(X_node), jnp.asarray(X_adj), jnp.asarray(y))
    assert loss.shape == ()
    assert float(loss) >= 0.0
