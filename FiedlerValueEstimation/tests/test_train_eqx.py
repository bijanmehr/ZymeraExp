import jax
import jax.numpy as jnp
import numpy as np

from fiedler.models_eqx import GRUEstimator, ConfigurableGCRN
from fiedler import train_eqx


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


# --------------------------------------------------------------------------------------
# train_configurable / predict_configurable (masked multi-N + full loss)
# --------------------------------------------------------------------------------------
def _multi_n_data(seed=0, H=3):
    """Two N-groups (N=4, N=6) padded to Nmax=6 with a learnable signal + a node_mask.

    Target lambda2 correlates with the windowed feature 0 mean over REAL nodes; some
    windows are 'fragmented' (y below tau) so the connected-flag head has both classes.
    """
    from fiedler import dataset
    rng = np.random.default_rng(seed)

    def grp(S, N, shift):
        Xn = rng.standard_normal((S, H, N, 6)).astype(np.float32)
        Xa = (rng.random((S, H, N, N)) > 0.4)
        Xp = rng.standard_normal((S, H, N, 2)).astype(np.float32)
        sig = Xn[:, -1, :, 0].mean(1)
        y = (0.4 + 0.5 * (sig - sig.min())).astype(np.float32) + shift
        # make ~a third of windows fragmented (tiny lambda2)
        frag = rng.random(S) < 0.3
        y = np.where(frag, 1e-5, y).astype(np.float32)
        return {"X_node": Xn, "X_adj": Xa, "X_pos": Xp, "y": y}

    return dataset.pad_batch([grp(40, 4, 0.0), grp(40, 6, 0.2)], N_max=6)


def test_train_configurable_reduces_loss():
    data = _multi_n_data(seed=0)
    model = ConfigurableGCRN(in_size=6, hidden=16, n_rounds=1, op="mean", content="value",
                             key=jax.random.PRNGKey(0))

    loss_before = train_eqx.configurable_loss(model, data["X_node"], data["X_adj"],
                                              data["X_pos"], data["y"], data["node_mask"],
                                              agree_w=0.0)
    trained, info = train_eqx.train_configurable(model, data, steps=200, lr=3e-3, batch=32,
                                                 val_frac=0.25, agree_w=0.05, seed=0)
    loss_after = train_eqx.configurable_loss(trained, data["X_node"], data["X_adj"],
                                             data["X_pos"], data["y"], data["node_mask"],
                                             agree_w=0.0)
    assert float(loss_after) < float(loss_before)
    assert 0.0 <= info["val_acc"] <= 1.0


def test_predict_configurable_shapes_and_masking():
    data = _multi_n_data(seed=1)
    model = ConfigurableGCRN(in_size=6, hidden=8, n_rounds=1, op="mean", content="value",
                             key=jax.random.PRNGKey(1))
    lam, cprob = train_eqx.predict_configurable(model, data)
    S, Nmax = data["node_mask"].shape
    assert lam.shape == (S, Nmax)
    assert cprob.shape == (S, Nmax)
    assert np.all(lam >= 0.0)                     # linear lambda2 from exp
    assert np.all((cprob >= 0.0) & (cprob <= 1.0))
    # padded nodes are masked out (set to 0)
    pad = ~data["node_mask"]
    assert np.all(lam[pad] == 0.0)
    assert np.all(cprob[pad] == 0.0)


def test_configurable_loss_ignores_padded_nodes():
    """Loss must be identical whether padded rows hold zeros or garbage (mask works)."""
    data = _multi_n_data(seed=2)
    model = ConfigurableGCRN(in_size=6, hidden=8, n_rounds=1, op="mean", content="value",
                             key=jax.random.PRNGKey(2))
    base = train_eqx.configurable_loss(model, data["X_node"], data["X_adj"], data["X_pos"],
                                       data["y"], data["node_mask"], agree_w=0.1)
    # corrupt the padded region of the node features
    Xn = np.array(data["X_node"])
    pad = ~data["node_mask"]                       # (S,Nmax)
    Xn[pad[:, None, :].repeat(Xn.shape[1], 1)] = 1e3
    corrupt = train_eqx.configurable_loss(model, Xn, data["X_adj"], data["X_pos"],
                                          data["y"], data["node_mask"], agree_w=0.1)
    assert np.allclose(float(base), float(corrupt), atol=1e-4)
