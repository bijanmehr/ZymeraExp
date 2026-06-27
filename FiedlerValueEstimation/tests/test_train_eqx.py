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


# --------------------------------------------------------------------------------------
# PARITY: the chunked-scan train_configurable must reproduce the OLD per-step loop exactly
# --------------------------------------------------------------------------------------
def _train_configurable_reference(model, data, *, steps, lr, weight_decay, batch, val_frac,
                                  agree_w, patience, seed, eval_every):
    """Faithful copy of the PRE-REFACTOR train_configurable: a Python per-step loop launching
    one jit'd step per iteration, with the val median-rel-err computed in numpy. The current
    `train_configurable` must match this bit-for-bit (identical RNG order) -> it is the parity
    oracle. (No checkpointing here; that path is covered by tests/test_checkpoint.py.)"""
    import equinox as eqx
    import optax
    from fiedler import dataset as _dataset

    EPS = train_eqx.EPS
    CONNECTED_TAU = train_eqx.CONNECTED_TAU

    X_node = jnp.asarray(data["X_node"], jnp.float32)
    X_adj = jnp.asarray(data["X_adj"])
    X_pos = jnp.asarray(data["X_pos"], jnp.float32)
    y = jnp.asarray(data["y"], jnp.float32)
    node_mask = jnp.asarray(data["node_mask"])
    S = X_node.shape[0]

    tr_idx, va_idx = _dataset.train_val_split(S, val_frac=val_frac, seed=seed)
    tr_idx = jnp.asarray(tr_idx)
    Xn_tr, Xa_tr, Xp_tr = X_node[tr_idx], X_adj[tr_idx], X_pos[tr_idx]
    y_tr, m_tr = y[tr_idx], node_mask[tr_idx]
    va = {k: np.asarray(data[k])[np.asarray(va_idx)] for k in
          ("X_node", "X_adj", "X_pos", "y", "node_mask")}
    va_local_idx = np.arange(len(va_idx))

    def old_val(model, vdata, idx):
        sub = {k: (np.asarray(vdata[k])[idx]) for k in
               ("X_node", "X_adj", "X_pos", "y", "node_mask")}
        lam, _ = train_eqx.predict_configurable(model, sub)
        yy = np.asarray(sub["y"], np.float32)
        mask = np.asarray(sub["node_mask"], bool)
        conn = (yy > CONNECTED_TAU)[:, None] & mask
        if not conn.any():
            return np.inf
        true = np.broadcast_to(yy[:, None], lam.shape)
        rel = np.abs(lam - true) / np.maximum(np.abs(true), EPS)
        return float(np.median(rel[conn]))

    sched = optax.cosine_decay_schedule(init_value=lr, decay_steps=max(steps, 1))
    opt = optax.chain(optax.clip_by_global_norm(1.0),
                      optax.adamw(sched, weight_decay=weight_decay))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def step_fn(model, opt_state, xn, xa, xp, yb, mb, dkey):
        loss, grads = eqx.filter_value_and_grad(train_eqx.configurable_loss)(
            model, xn, xa, xp, yb, mb, agree_w=agree_w, key=dkey)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    n_tr = Xn_tr.shape[0]
    bs = min(batch, n_tr)
    key = jax.random.PRNGKey(seed + 1)

    best_model = model
    best_err = np.inf
    no_improve = 0
    it = -1
    for it in range(steps):
        key, sk, dk = jax.random.split(key, 3)
        idx = jax.random.randint(sk, (bs,), 0, n_tr)
        model, opt_state, _ = step_fn(model, opt_state, Xn_tr[idx], Xa_tr[idx], Xp_tr[idx],
                                      y_tr[idx], m_tr[idx], dk)
        if (it + 1) % eval_every == 0 or it == steps - 1:
            err = old_val(model, va, va_local_idx)
            if err < best_err - 1e-5:
                best_err = err
                best_model = model
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    val_acc = float(np.clip(1.0 - best_err, 0.0, 1.0)) if np.isfinite(best_err) else 0.0
    info = {"val_acc": val_acc, "val_err": float(best_err), "steps_run": it + 1}
    return best_model, info


def _parity_assert(steps, eval_every, patience, seed=0, agree_w=0.05, batch=32):
    data = _multi_n_data(seed=seed)
    kw = dict(steps=steps, lr=3e-3, weight_decay=1e-4, batch=batch, val_frac=0.25,
              agree_w=agree_w, patience=patience, seed=seed, eval_every=eval_every)

    m_ref = ConfigurableGCRN(in_size=6, hidden=16, n_rounds=1, op="mean", content="value",
                             key=jax.random.PRNGKey(0))
    m_new = ConfigurableGCRN(in_size=6, hidden=16, n_rounds=1, op="mean", content="value",
                             key=jax.random.PRNGKey(0))

    _, ref = _train_configurable_reference(m_ref, data, **kw)
    _, new = train_eqx.train_configurable(m_new, data, **kw)

    # steps_run must match exactly; val metrics bit-exact in practice (identical RNG order),
    # allow a tiny tolerance only for float reassociation inside the scan.
    assert new["steps_run"] == ref["steps_run"], (new["steps_run"], ref["steps_run"])
    assert abs(new["val_err"] - ref["val_err"]) < 1e-4, (new["val_err"], ref["val_err"])
    assert abs(new["val_acc"] - ref["val_acc"]) < 1e-4, (new["val_acc"], ref["val_acc"])
    return ref, new


def test_train_configurable_parity_with_old_loop():
    """New chunked-scan trainer == old per-step loop on val_acc / val_err / steps_run."""
    ref, new = _parity_assert(steps=300, eval_every=50, patience=15)
    # ran the full budget (no early stop in this fixed config) -> steps_run == steps
    assert new["steps_run"] == 300


def test_train_configurable_parity_partial_final_chunk():
    """Parity when steps is NOT a multiple of eval_every (final short chunk path)."""
    _parity_assert(steps=140, eval_every=50, patience=99)


def test_train_configurable_parity_early_stop():
    """Parity when early-stopping triggers (patience small) -> same break step & metrics."""
    ref, new = _parity_assert(steps=600, eval_every=50, patience=1, seed=3)
    assert new["steps_run"] < 600                       # early stop actually fired
    assert new["steps_run"] == ref["steps_run"]
