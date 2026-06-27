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

from . import checkpoint as _ckpt
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


# ======================================================================================
# Configurable multi-N training: full loss (Huber + BCE + heteroscedastic NLL +
# node-agreement reg), all MASKED by node_mask; cosine + adamw(weight_decay) + clip.
# ======================================================================================
CONNECTED_TAU = 1e-3


def _bce_with_logits(logits, labels):
    """Numerically stable elementwise BCE: max(x,0) - x*z + log(1+exp(-|x|))."""
    return jnp.maximum(logits, 0) - logits * labels + jnp.log1p(jnp.exp(-jnp.abs(logits)))


def _masked_mean(x, mask):
    """Mean of x over entries where mask is True (mask float/bool, same shape)."""
    m = mask.astype(x.dtype)
    return (x * m).sum() / jnp.maximum(m.sum(), 1.0)


def _per_window_masked_var(x, mask):
    """Per-row variance over masked (real) entries -> (B,). x,mask are (B,N)."""
    m = mask.astype(x.dtype)
    cnt = jnp.maximum(m.sum(-1), 1.0)                        # (B,)
    mean = (x * m).sum(-1) / cnt                             # (B,)
    var = ((x - mean[:, None]) ** 2 * m).sum(-1) / cnt       # (B,)
    return var


def configurable_loss(model, X_node, X_adj, X_pos, y, node_mask, *, agree_w=0.0, key=None):
    """Full masked loss for ConfigurableGCRN. Scalar.

    L = Huber_1(logl2 - log(y+eps))                            [regression]
      + 0.3 * BCE(cflag, connected = y > tau)                  [connected flag]
      + 0.1 * ( 0.5*exp(-2 logsig)*(logl2-target)^2 + logsig ) [heteroscedastic NLL]
      + agree_w * mean_window( Var_realnodes(logl2) )          [node-agreement reg]
    Every per-node term is masked by `node_mask` (padded nodes never contribute). The
    target lambda2 is a per-window scalar broadcast over that window's real nodes.
    """
    def fwd(xn, xa, xp):
        return model(xn, xa, xp, key=key)
    out = jax.vmap(fwd)(X_node, X_adj, X_pos)               # each (B, N)
    logl2 = out["logl2"]
    cflag = out["cflag"]
    logsig = out["logsig"]

    B, N = logl2.shape
    mask = node_mask.astype(logl2.dtype)                    # (B,N)
    target = jnp.log(y + EPS)[:, None]                      # (B,1) broadcast
    connected = (y > CONNECTED_TAU).astype(logl2.dtype)[:, None]  # (B,1)

    huber = _huber(logl2 - target, delta=1.0)              # (B,N)
    bce = _bce_with_logits(cflag, jnp.broadcast_to(connected, cflag.shape))
    logsig_c = jnp.clip(logsig, -6.0, 6.0)                  # keep NLL well-conditioned
    nll = 0.5 * jnp.exp(-2.0 * logsig_c) * (logl2 - target) ** 2 + logsig_c

    reg_loss = _masked_mean(huber, mask)
    bce_loss = _masked_mean(bce, mask)
    nll_loss = _masked_mean(nll, mask)

    agree = jnp.mean(_per_window_masked_var(logl2, mask)) if agree_w else 0.0

    return reg_loss + 0.3 * bce_loss + 0.1 * nll_loss + agree_w * agree


def predict_configurable(model, data):
    """-> (linear lambda2 (S,Nmax), connected-prob (S,Nmax)); padded nodes set to 0."""
    X_node = jnp.asarray(data["X_node"], jnp.float32)
    X_adj = jnp.asarray(data["X_adj"])
    X_pos = jnp.asarray(data["X_pos"], jnp.float32)
    mask = np.asarray(data["node_mask"], bool)

    def fwd(xn, xa, xp):
        return model(xn, xa, xp)                            # eval: no dropedge
    # Use lax.map (sequential per-sample) rather than jax.vmap here: vmap over the model
    # miscompiles on GPU XLA for some (batch, N) layouts -- single-head attention at N=20
    # raised "INVALID_ARGUMENT: Reshape ... 8x1024 -> 8x2x32x32" (an XLA layout/fusion bug;
    # the per-sample/bare forward always compiles). lax.map compiles the single-sample forward
    # and scans it, dodging the bug. Eval is not the training hot loop, so the cost is moot.
    out = jax.lax.map(lambda xs: fwd(*xs), (X_node, X_adj, X_pos))
    lam = np.asarray(jnp.exp(out["logl2"]))                 # (S,Nmax) linear
    cprob = np.asarray(jax.nn.sigmoid(out["cflag"]))        # (S,Nmax) prob
    lam = np.where(mask, lam, 0.0)
    cprob = np.where(mask, cprob, 0.0)
    return lam, cprob


@eqx.filter_jit
def _val_median_rel_err_connected_dev(model, X_node, X_adj, X_pos, y, node_mask):
    """On-device median rel-err over connected real nodes -> a single scalar (jnp).

    Equivalent to the host version below, but every op stays on the accelerator so the
    only host transfer is the final scalar. The forward uses `lax.map` (sequential
    per-sample) for the SAME reason `predict_configurable` does -- vmap over the model
    miscompiles on GPU XLA for single-head attention at some (batch, N) layouts.

    Median parity with ``np.median(rel[conn])``: select the connected real-node entries
    (set the rest to +inf so they sort to the end), sort, then average the two middle
    order statistics at indices ``(n-1)//2`` and ``n//2`` (n = #valid). This reproduces
    numpy's even/odd-count median *bit-for-bit* (verified). When no window is connected
    the result is +inf, matching the host version's ``np.inf`` early return.
    """
    def fwd(xn, xa, xp):
        return model(xn, xa, xp)                            # eval: no dropedge
    out = jax.lax.map(lambda xs: fwd(*xs), (X_node, X_adj, X_pos))
    lam = jnp.exp(out["logl2"])                             # (s,Nmax) linear
    mask = node_mask.astype(bool)
    conn = (y > CONNECTED_TAU)[:, None] & mask              # (s,Nmax) connected real nodes
    true = jnp.broadcast_to(y[:, None], lam.shape)
    rel = jnp.abs(lam - true) / jnp.maximum(jnp.abs(true), EPS)
    relf = jnp.where(conn, rel, jnp.inf).reshape(-1)        # invalid -> +inf (sort to end)
    n = conn.sum()
    sv = jnp.sort(relf)
    lo = (n - 1) // 2
    hi = n // 2
    med = 0.5 * (sv[lo] + sv[hi])
    return jnp.where(n > 0, med, jnp.inf)


def _val_median_rel_err_connected(model, data, idx):
    """Median rel-err in LINEAR space over CONNECTED-window real nodes (idx subset).

    Thin host wrapper around the jit'd on-device kernel: slices the val arrays by `idx`
    (on device) and returns one Python float. Kept as the per-eval entry point (callers
    and tests reference it by name), so the numeric result is identical to the previous
    pure-numpy implementation while the per-eval GPU->CPU array round-trip is removed.
    """
    idx = jnp.asarray(idx)
    Xn = jnp.asarray(data["X_node"], jnp.float32)[idx]
    Xa = jnp.asarray(data["X_adj"])[idx]
    Xp = jnp.asarray(data["X_pos"], jnp.float32)[idx]
    y = jnp.asarray(data["y"], jnp.float32)[idx]
    m = jnp.asarray(data["node_mask"])[idx]
    return float(_val_median_rel_err_connected_dev(model, Xn, Xa, Xp, y, m))


def train_configurable(model, data, *, steps=8000, lr=3e-4, weight_decay=1e-4, batch=128,
                       val_frac=0.2, agree_w=0.0, patience=15, seed=0, eval_every=50,
                       ckpt_path=None, ckpt_every=1000):
    """Train a ConfigurableGCRN on masked multi-N data; return (model, info).

    `data`: dict X_node (S,H,Nmax,6), X_adj (S,H,Nmax,Nmax), X_pos (S,H,Nmax,2),
            y (S,), node_mask (S,Nmax).
    optax: adamw(lr, weight_decay) + clip_by_global_norm(1.0) + cosine schedule over `steps`.
    Early-stop on val linear-space median-rel-err over CONNECTED real nodes; keep best.
    info = {"val_acc", "val_err", "steps_run"}.

    Crash-safe checkpointing: if `ckpt_path` is set, the full training state (model, optimizer
    state, best-so-far model, RNG key, and the step/best/patience scalars) is written there
    every `ckpt_every` steps, and the run RESUMES from it when the file already exists (e.g.
    after a crash). The optimizer state carries the cosine-schedule step, so resume keeps the
    LR schedule exact. The checkpoint is deleted on clean completion.
    """
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

    sched = optax.cosine_decay_schedule(init_value=lr, decay_steps=max(steps, 1))
    opt = optax.chain(optax.clip_by_global_norm(1.0), optax.adamw(sched, weight_decay=weight_decay))
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    n_tr = Xn_tr.shape[0]
    bs = min(batch, n_tr)
    key = jax.random.PRNGKey(seed + 1)

    # --- chunked training: run `chunk_len` steps as ONE compiled lax.scan ----------------
    # The scan body is the SAME `split -> randint -> grad step` as the old per-step loop, in
    # the SAME order, so the RNG stream (and therefore every model update) is bit-identical;
    # it just collapses `chunk_len` host launches into one. The carry is (params, opt_state,
    # key) -- only the array leaves of the model travel through the scan; the static skeleton
    # is closed over and re-combined. `chunk_len` is static (recompiles only for a final
    # short chunk when steps % eval_every != 0). Built lazily and cached per length.
    _runner_cache = {}

    def _get_runner(chunk_len):
        runner = _runner_cache.get(chunk_len)
        if runner is None:
            @eqx.filter_jit
            def runner(model, opt_state, key, Xn_tr, Xa_tr, Xp_tr, y_tr, m_tr, _cl=chunk_len):
                static = eqx.filter(model, eqx.is_array, inverse=True)

                def body(carry, _):
                    params, ostate, k = carry
                    model_c = eqx.combine(params, static)
                    k, sk, dk = jax.random.split(k, 3)
                    idx = jax.random.randint(sk, (bs,), 0, n_tr)
                    # dkey enables DropEdge if the model was built with dropedge>0 (train only)
                    loss, grads = eqx.filter_value_and_grad(configurable_loss)(
                        model_c, Xn_tr[idx], Xa_tr[idx], Xp_tr[idx], y_tr[idx], m_tr[idx],
                        agree_w=agree_w, key=dk)
                    updates, ostate = opt.update(grads, ostate, eqx.filter(model_c, eqx.is_array))
                    model_c = eqx.apply_updates(model_c, updates)
                    return (eqx.filter(model_c, eqx.is_array), ostate, k), None

                init = (eqx.filter(model, eqx.is_array), opt_state, key)
                (params_f, ostate_f, key_f), _ = jax.lax.scan(body, init, None, length=_cl)
                return eqx.combine(params_f, static), ostate_f, key_f
            _runner_cache[chunk_len] = runner
        return runner

    best_model = model
    best_err = np.inf
    no_improve = 0
    start_it = 0

    # resume from a prior checkpoint if one exists (crash recovery); template gives structure
    if _ckpt.exists(ckpt_path):
        template = {"model": model, "opt_state": opt_state, "best_model": best_model,
                    "key": key, "scalars": jnp.zeros((3,), jnp.float32)}
        st = _ckpt.load_state(ckpt_path, template)
        model, opt_state, best_model, key = (st["model"], st["opt_state"],
                                             st["best_model"], st["key"])
        it_f, be_f, ni_f = (float(x) for x in np.asarray(st["scalars"]))
        start_it, best_err, no_improve = int(it_f) + 1, be_f, int(ni_f)
        print(f"[ckpt] resume {ckpt_path}: step {start_it}/{steps}, best_err={best_err:.4f}",
              flush=True)

    # Outer host loop over chunks. Each chunk runs the steps in [done, done+chunk_len) on
    # device, then evals / early-stops / checkpoints at chunk granularity. Because evals in
    # the old loop fired exactly at every `eval_every` boundary (and at the final step), the
    # chunk boundaries land on the SAME steps -- so early-stop and checkpoint decisions, and
    # `steps_run`, are identical to the per-step loop.
    last_it = start_it - 1
    done = start_it
    while done < steps:
        chunk_len = min(eval_every, steps - done)
        model, opt_state, key = _get_runner(chunk_len)(
            model, opt_state, key, Xn_tr, Xa_tr, Xp_tr, y_tr, m_tr)
        done += chunk_len
        last_it = done - 1                                  # 0-based index of last step run

        err = _val_median_rel_err_connected(model, va, va_local_idx)
        if err < best_err - 1e-5:
            best_err = err
            best_model = model
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= patience:
            break                                           # early stop (ckpt dropped below)

        if ckpt_path and ((last_it + 1) % ckpt_every == 0 or last_it == steps - 1):
            _ckpt.save_state(ckpt_path, {
                "model": model, "opt_state": opt_state, "best_model": best_model,
                "key": key, "scalars": jnp.asarray([last_it, best_err, no_improve],
                                                    jnp.float32)})

    _ckpt.remove(ckpt_path)                    # clean completion -> drop the resume checkpoint
    val_acc = float(np.clip(1.0 - best_err, 0.0, 1.0)) if np.isfinite(best_err) else 0.0
    info = {"val_acc": val_acc, "val_err": float(best_err), "steps_run": last_it + 1}
    return best_model, info
