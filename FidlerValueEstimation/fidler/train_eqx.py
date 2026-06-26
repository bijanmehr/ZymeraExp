"""Train the Equinox lambda2 estimators (Huber on log-target) + predict in linear space.

Loss: Huber(delta=1) on (pred_log - log(y+1e-6)), per-node prediction broadcast against
the scalar (per-window) target, meaned over nodes and batch. Training uses AdamW with
global-norm grad-clipping, eqx.filter_jit / eqx.filter_value_and_grad, and early-stops on
the validation **median relative error in linear space** (exp of the log-prediction).
"""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import dataset as _dataset

EPS = 1e-6


def _huber(residual, delta=1.0):
    a = jnp.abs(residual)
    return jnp.where(a <= delta, 0.5 * a ** 2, delta * (a - 0.5 * delta))


def loss_fn(model, X_node, X_adj, y):
    """Mean Huber(delta=1) of (per-node log-pred - log(y+eps)). Scalar."""
    pred_log = jax.vmap(model)(X_node, X_adj)             # (B, N) log-lambda2
    target = jnp.log(y + EPS)[:, None]                    # (B, 1) broadcast over nodes
    return jnp.mean(_huber(pred_log - target, delta=1.0))


def predict(model, X_node, X_adj):
    """-> (S, N) LINEAR lambda2 (exp of per-node log-prediction)."""
    X_node = jnp.asarray(X_node, jnp.float32)
    X_adj = jnp.asarray(X_adj)
    pred_log = jax.vmap(model)(X_node, X_adj)             # (S, N)
    return np.asarray(jnp.exp(pred_log))


def _val_median_rel_err(model, X_node, X_adj, y):
    """Median relative error in LINEAR space over (sample, node) pairs."""
    pred = predict(model, X_node, X_adj)                  # (S, N) linear
    true = np.asarray(y, np.float32)[:, None]
    rel = np.abs(pred - true) / np.maximum(np.abs(true), EPS)
    return float(np.median(rel))


def train(model, X_node, X_adj, y, *, steps=1500, lr=3e-4, batch=128, val_frac=0.2,
          seed=0, eval_every=25, patience=10):
    """Train `model`; return (trained_model, val_accuracy_linear).

    val_accuracy_linear = clip(1 - median-rel-error-in-linear-space, 0, 1) on the val split.
    Early-stop after `patience` evaluations without improvement; keep the best params.
    """
    X_node = jnp.asarray(X_node, jnp.float32)
    X_adj = jnp.asarray(X_adj)
    y = jnp.asarray(y, jnp.float32)
    S = X_node.shape[0]

    tr_idx, va_idx = _dataset.train_val_split(S, val_frac=val_frac, seed=seed)
    tr_idx = jnp.asarray(tr_idx)
    Xn_tr, Xa_tr, y_tr = X_node[tr_idx], X_adj[tr_idx], y[tr_idx]
    Xn_va = np.asarray(X_node[va_idx]); Xa_va = np.asarray(X_adj[va_idx]); y_va = np.asarray(y[va_idx])

    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(lr))
    params = eqx.filter(model, eqx.is_array)
    opt_state = opt.init(params)

    @eqx.filter_jit
    def step_fn(model, opt_state, xb, ab, yb):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, xb, ab, yb)
        updates, opt_state = opt.update(grads, opt_state, eqx.filter(model, eqx.is_array))
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

    n_tr = Xn_tr.shape[0]
    bs = min(batch, n_tr)
    key = jax.random.PRNGKey(seed + 1)

    best_model = model
    best_err = np.inf
    no_improve = 0

    for it in range(steps):
        key, sk = jax.random.split(key)
        idx = jax.random.randint(sk, (bs,), 0, n_tr)
        model, opt_state, _ = step_fn(model, opt_state, Xn_tr[idx], Xa_tr[idx], y_tr[idx])

        if (it + 1) % eval_every == 0 or it == steps - 1:
            err = _val_median_rel_err(model, Xn_va, Xa_va, y_va)
            if err < best_err - 1e-5:
                best_err = err
                best_model = model
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    break

    val_acc = float(np.clip(1.0 - best_err, 0.0, 1.0))
    return best_model, val_acc
