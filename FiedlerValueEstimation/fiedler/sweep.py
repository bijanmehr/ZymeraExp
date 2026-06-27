"""Sweep runner for the configurable message-design study.

`run_config(cfg, base_seed=0)` builds multi-N datasets (datagen across `cfg['train_N']`),
windows + pads them at `H = cfg['H']`, trains a `ConfigurableGCRN` with cfg's message /
size / regularization knobs, then evaluates and returns a metrics dict:

    accuracy                  overall (all real-node predictions, in-distribution eval)
    connected_accuracy        connected-window real nodes only
    connected_flag_accuracy   cflag-head classification accuracy (real nodes)
    cv20_mean / cv20_std      5-fold CV mean +/- std at N = cv_N (default 20)
    extrap                    {N: zero-shot accuracy} for N in cfg['extrap_N'] (default 24, 30)

`run_sweep(configs, out_jsonl, base_seed=0)` runs each config and APPENDS one JSON line
per config to `out_jsonl` as it finishes (partial results survive a crash), returns the list.

All evaluation is on FRESHLY generated held-out episodes (separate seed offset from the
training pool) and is masked: padded nodes never enter any metric.
"""
import json
import os
import time

import jax
import numpy as np

from .config import DataCfg
from . import datagen, dataset, dynamics, identity, margin, metrics, signal
from .models_eqx import ConfigurableGCRN
from . import checkpoint as _ckpt
from . import train_eqx

CONNECTED_TAU = train_eqx.CONNECTED_TAU

# datagen knob keys (with defaults) pulled from a cfg.
_DATA_DEFAULTS = dict(grid=16, comm_r=5, n_obstacles=0, spawn_radius=2,
                      n_episodes=8, n_steps=100)


def _data_kwargs(cfg):
    return {k: cfg.get(k, v) for k, v in _DATA_DEFAULTS.items()}


def _select_grid(cfg, N):
    """Per-N grid: use cfg['grid_for_n'](N) if provided (to hold density ~fixed),
    else the static cfg['grid'] default."""
    g4n = cfg.get("grid_for_n")
    if callable(g4n):
        return int(g4n(N))
    return int(cfg.get("grid", _DATA_DEFAULTS["grid"]))


def _generate_ds(cfg, N, seed):
    """Build the per-N DataCfg and dispatch to the guardrail or random generator.

    `cfg['data']` selects the data regime: 'guardrail' (default) draws ALWAYS-CONNECTED
    dispersed rollouts via `generate_dataset_guardrail`; 'random' keeps the legacy
    `generate_dataset` (random-policy) behavior. The grid is chosen per-N (see
    `_select_grid`) so a `grid_for_n` mapping can hold node density roughly fixed.
    """
    dk = dict(_data_kwargs(cfg))
    dk["grid"] = _select_grid(cfg, N)
    data_cfg = DataCfg(n_agents=N, seed=seed, **dk)
    if cfg.get("data", "guardrail") == "random":
        return datagen.generate_dataset(data_cfg)
    return datagen.generate_dataset_guardrail(data_cfg)


def _gen_group(N, H, seed, cfg):
    """Roll out N-agent episodes and window them -> a single-N window-set dict."""
    ds = _generate_ds(cfg, N, seed)
    Xn, y = dataset.make_windows(ds["features"], ds["lambda2"], H)
    Xa = dataset.make_adj_windows(ds["adjacency"], H)
    Xp = dataset.make_pos_windows(ds["positions"], H)
    return {"X_node": Xn, "X_adj": Xa, "X_pos": Xp, "y": y}


# --------------------------------------------------------------------------------------
# agent-identity augmentation (keyed off cfg['id_mode']; default 'none' is a no-op)
# --------------------------------------------------------------------------------------
def _id_in_size(cfg):
    """Model in_size after the agent-identity augmentation: 6 (+id_dim random | +1 index)."""
    id_mode = cfg.get("id_mode", "none")
    if id_mode == "random":
        return 6 + int(cfg.get("id_dim", 4))
    if id_mode == "index":
        return 6 + 1
    return 6


def _model_in_size(cfg):
    """Final model in_size: id-augmented width, +1 per active node-feature overlay.

    Each of the connectivity-margin (`margin_mode=='on'`) and signal-strength
    (`signal_mode=='on'`) overlays appends exactly ONE per-agent node feature, and they
    compose independently with the agent-identity width.
    """
    return (_id_in_size(cfg)
            + (1 if cfg.get("margin_mode", "off") == "on" else 0)
            + (1 if _signal_on(cfg) else 0)
            + (3 if cfg.get("dynamics_mode", "off") == "on" else 0))


def _augment_id(data, cfg, seed):
    """Append agent-ID features to a padded batch dict's X_node (in place on a copy).

    Uses cfg['id_mode'] / cfg['id_dim'] and the given `seed` (so the random tags are
    deterministic and reproducible). 'none' leaves the data untouched.
    """
    id_mode = cfg.get("id_mode", "none")
    if id_mode == "none":
        return data
    data = dict(data)
    data["X_node"], _ = identity.augment_with_id(
        data["X_node"], data["node_mask"],
        id_mode=id_mode, id_dim=int(cfg.get("id_dim", 4)), seed=int(seed))
    return data


# --------------------------------------------------------------------------------------
# connectivity-margin augmentation (keyed off cfg['margin_mode']; default 'off' is a no-op)
# --------------------------------------------------------------------------------------
def _margin_on(cfg):
    """True iff cfg requests the connectivity-margin overlay (margin_mode == 'on')."""
    return cfg.get("margin_mode", "off") == "on"


def _augment_margin(data, cfg):
    """Append the connectivity-margin node feature to a padded batch dict (after any ID).

    No-op unless cfg['margin_mode']=='on'. Computes the per-agent max-neighbor-distance /
    comm_r from the batch's adjacency + position windows (last step), zero for isolated /
    padded agents. Applied AFTER `_augment_id` so it lands on the final node axis.
    """
    if not _margin_on(cfg):
        return data
    data = dict(data)
    data["X_node"], _ = margin.augment_with_margin(
        data["X_node"], data["X_adj"], data["X_pos"],
        comm_r=float(cfg.get("comm_r", 5)))
    return data


# --------------------------------------------------------------------------------------
# signal-strength augmentation (keyed off cfg['signal_mode']; default 'off' is a no-op)
# --------------------------------------------------------------------------------------
def _signal_on(cfg):
    """True iff cfg requests the signal-strength overlay (signal_mode == 'on')."""
    return cfg.get("signal_mode", "off") == "on"


def _augment_signal(data, cfg):
    """Apply the signal-strength overlay to a padded batch (after ID + margin).

    No-op unless cfg['signal_mode']=='on'. When on, it does TWO things, in order:
      1. Replaces the BINARY X_adj with the FLOAT path-loss-weighted adjacency
         (`signal.signal_weighted_adj`), so the message-passing ops in
         `messages.aggregate` become soft-weighted automatically (the soft adjacency that
         aligns with the soft Laplacian governing lambda2). The weighting is computed from
         the still-binary X_adj + positions, so it must run BEFORE the adj is overwritten.
      2. Appends the per-agent mean-neighbor-signal-strength NODE feature (on the final
         node axis, after any ID + margin features), zero for isolated / padded agents.
    """
    if not _signal_on(cfg):
        return data
    data = dict(data)
    comm_r = float(cfg.get("comm_r", 5))
    binary_adj = data["X_adj"]                           # still bool here (margin left it alone)
    # node feature first (from the binary adj), then overwrite the adj with soft weights.
    data["X_node"], _ = signal.augment_with_signal(
        data["X_node"], binary_adj, data["X_pos"], comm_r=comm_r)
    data["X_adj"] = signal.signal_weighted_adj(binary_adj, data["X_pos"], comm_r=comm_r)
    return data


def _dynamics_on(cfg):
    """True iff cfg requests the dynamic-features overlay (dynamics_mode == 'on')."""
    return cfg.get("dynamics_mode", "off") == "on"


def _augment_dynamics(data, cfg):
    """Append the temporal-trend node features (Delta-degree, neighbor approach-rate, own speed)
    to a padded batch (after ID + margin, BEFORE signal so degree reads the binary adjacency).
    No-op unless cfg['dynamics_mode']=='on'."""
    if not _dynamics_on(cfg):
        return data
    data = dict(data)
    data["X_node"], _ = dynamics.augment_with_dynamics(
        data["X_node"], data["X_adj"], data["X_pos"], comm_r=float(cfg.get("comm_r", 5)))
    return data


def _augment(data, cfg, seed):
    """Full node-feature augmentation pipeline: ID, margin, dynamics, then signal.

    Signal runs LAST so its node feature lands after the others and its adjacency re-weighting
    reads the binary adjacency the earlier steps left untouched (dynamics also reads binary degree).
    """
    return _augment_signal(_augment_dynamics(
        _augment_margin(_augment_id(data, cfg, seed), cfg), cfg), cfg)


def _build_pool(N_list, H, seed, cfg, N_max=None):
    """Generate + window + pad a list of N's into one padded multi-N batch dict.

    The agent-identity augmentation (cfg['id_mode']) is applied AFTER padding so the ID
    features land on the final node axis and padded agents are zeroed. The augmentation
    seed is derived from `seed` so each pool is reproducible.
    """
    if N_max is None:
        N_max = max(N_list)
    groups = [_gen_group(N, H, seed + i, cfg) for i, N in enumerate(N_list)]
    data = dataset.pad_batch(groups, N_max)
    return _augment(data, cfg, seed=seed + 123)


def _build_model(cfg, key):
    # margin_mode='on' OR signal_mode='on' forces the message round to the "margin"
    # content (per-edge dist/comm_r inside the messages -- a monotone transform of the
    # signal strength, so it carries the same per-edge link quality); else use the
    # configured content. margin's own setting takes precedence (identical result here).
    content = "margin" if (_margin_on(cfg) or _signal_on(cfg)) else cfg.get("content", "value")
    return ConfigurableGCRN(
        in_size=_model_in_size(cfg),
        hidden=int(cfg.get("hidden", 128)),
        n_rounds=int(cfg.get("n_rounds", 2)),
        op=cfg.get("op", "mean"),
        content=content,
        heads=int(cfg.get("heads", 4)),
        dropedge=float(cfg.get("dropedge", 0.0)),
        comm_r=float(cfg.get("comm_r", 5)),
        key=key,
    )


def _ckpt_path(cfg, tag):
    """Per-training checkpoint path under cfg['ckpt_dir'] (default results/ckpt), or None if
    checkpointing is disabled (cfg['ckpt'] = False) or no tag is given."""
    if not tag or not cfg.get("ckpt", True):
        return None
    return os.path.join(cfg.get("ckpt_dir", "results/ckpt"), f"{tag}.train.eqx")


def _train(model, data, cfg, seed, ckpt_tag=None):
    return train_eqx.train_configurable(
        model, data,
        steps=int(cfg.get("steps", 12000)),
        lr=float(cfg.get("lr", 3e-4)),
        weight_decay=float(cfg.get("weight_decay", 1e-4)),
        batch=int(cfg.get("batch", 128)),
        val_frac=float(cfg.get("val_frac", 0.2)),
        agree_w=float(cfg.get("agree_w", 0.0)),
        patience=int(cfg.get("patience", 15)),
        seed=seed,
        ckpt_path=_ckpt_path(cfg, ckpt_tag),
        ckpt_every=int(cfg.get("ckpt_every", 1000)),
    )


# --------------------------------------------------------------------------------------
# scoring (masked)
# --------------------------------------------------------------------------------------
def _score(model, data):
    """Overall accuracy, connected-only accuracy, connected-flag accuracy on `data` (masked)."""
    lam, cprob = train_eqx.predict_configurable(model, data)        # (S,Nmax)
    y = np.asarray(data["y"], np.float32)
    mask = np.asarray(data["node_mask"], bool)
    true = np.broadcast_to(y[:, None], lam.shape)

    overall = metrics.accuracy(lam[mask], true[mask])

    conn = (y > CONNECTED_TAU)[:, None] & mask
    conn_acc = metrics.accuracy(lam[conn], true[conn]) if conn.any() else 0.0

    true_flag = np.broadcast_to((y > CONNECTED_TAU)[:, None], cprob.shape)
    pred_flag = cprob > 0.5
    flag_acc = metrics.connected_accuracy(pred_flag[mask], true_flag[mask])
    return float(overall), float(conn_acc), float(flag_acc)


def _cv_at_N(cfg, base_seed):
    """5-fold CV at N = cv_N: rotate held-out folds of N-episodes -> (mean, std) accuracy.

    Faithful to ARCHITECTURES sec 5: for each fold, train on the full multi-N pool minus
    that fold's N-episodes and evaluate on the held-out fold. Returns (None, None) if
    disabled (`cv_N` falsy or `cv_folds` < 2).
    """
    cv_N = cfg.get("cv_N", 20)
    folds = int(cfg.get("cv_folds", 5))
    if not cv_N or folds < 2:
        return None, None

    H = int(cfg["H"])
    train_N = list(cfg["train_N"])
    # generate the cv_N episodes once; split episodes into `folds` groups.
    n_ep = int(cfg.get("n_episodes", _DATA_DEFAULTS["n_episodes"]))
    cv_seed = base_seed + 7000
    ds = _generate_ds(cfg, cv_N, cv_seed)
    ep_idx = np.arange(n_ep)
    rng = np.random.default_rng(base_seed + 17)
    rng.shuffle(ep_idx)
    fold_splits = np.array_split(ep_idx, min(folds, n_ep))

    # base pool from the other train sizes (excluding cv_N to avoid leakage with the folds)
    other_N = [N for N in train_N if N != cv_N]

    accs = []
    for f, held in enumerate(fold_splits):
        held = np.asarray(held)
        if held.size == 0:
            continue
        keep_ep = np.setdiff1d(ep_idx, held)
        # train group = cv_N episodes minus the held fold
        tr_groups = []
        if other_N:
            for i, N in enumerate(other_N):
                tr_groups.append(_gen_group(N, H, base_seed + 100 * (f + 1) + i, cfg))
        tr_groups.append(_window_episode_subset(ds, keep_ep, H))
        N_max = max(train_N)
        tr_data = _augment(dataset.pad_batch(tr_groups, N_max), cfg, seed=base_seed + 200 * (f + 1))

        held_group = _window_episode_subset(ds, held, H)
        held_data = _augment(dataset.pad_batch([held_group], N_max), cfg, seed=base_seed + 201 * (f + 1))

        model = _build_model(cfg, jax.random.PRNGKey(base_seed + 31 + f))
        cv_name = cfg.get("name", cfg.get("op", "cfg"))
        model, _ = _train(model, tr_data, cfg, seed=base_seed + 41 + f,
                          ckpt_tag=f"{cv_name}-fold{f}")
        overall, _, _ = _score(model, held_data)
        accs.append(overall)

    if not accs:
        return None, None
    return float(np.mean(accs)), float(np.std(accs))


def _window_episode_subset(ds, ep_indices, H):
    """Window only the given episodes of a generated dataset -> single-N window-set dict."""
    ep_indices = np.asarray(ep_indices)
    feats = ds["features"][ep_indices]
    adj = ds["adjacency"][ep_indices]
    lam = ds["lambda2"][ep_indices]
    pos = ds["positions"][ep_indices]
    Xn, y = dataset.make_windows(feats, lam, H)
    Xa = dataset.make_adj_windows(adj, H)
    Xp = dataset.make_pos_windows(pos, H)
    return {"X_node": Xn, "X_adj": Xa, "X_pos": Xp, "y": y}


def _extrapolate(model, cfg, base_seed):
    """Zero-shot accuracy at each N in cfg['extrap_N'] (default {24,30}); no retrain."""
    extrap_N = cfg.get("extrap_N", [24, 30])
    H = int(cfg["H"])
    out = {}
    for j, N in enumerate(extrap_N):
        data = dataset.pad_batch([_gen_group(N, H, base_seed + 5000 + j, cfg)], N)
        data = _augment(data, cfg, seed=base_seed + 6000 + j)
        overall, _, _ = _score(model, data)
        out[str(N)] = float(overall)
    return out


# --------------------------------------------------------------------------------------
# public API
# --------------------------------------------------------------------------------------
def _json_safe_cfg(cfg):
    """Echo-able copy of a cfg: callables (e.g. `grid_for_n`) -> a string marker, so the
    result dict survives `json.dumps` in `run_sweep`."""
    out = {}
    for k, v in cfg.items():
        out[k] = f"<callable:{getattr(v, '__name__', 'fn')}>" if callable(v) else v
    return out


def run_config(cfg, *, base_seed=0):
    """Train + evaluate one configuration. Returns a metrics dict (includes echoed config)."""
    t0 = time.time()
    H = int(cfg["H"])
    train_N = list(cfg["train_N"])
    eval_N = list(cfg.get("eval_N", train_N))

    # --- train on the multi-N pool (checkpointed; resumes a crashed config mid-training) ---
    name = cfg.get("name", cfg.get("op", "cfg"))
    pool = _build_pool(train_N, H, base_seed + 1, cfg)
    model = _build_model(cfg, jax.random.PRNGKey(base_seed))
    model, info = _train(model, pool, cfg, seed=base_seed, ckpt_tag=f"{name}-main")
    # persist the deployable trained estimator (+ hyperparam meta to rebuild its skeleton)
    if cfg.get("ckpt", True):
        meta = {"name": name, "op": cfg.get("op"), "content": cfg.get("content", "value"),
                "hidden": int(cfg.get("hidden", 128)), "n_rounds": int(cfg.get("n_rounds", 2)),
                "heads": int(cfg.get("heads", 4)), "comm_r": float(cfg.get("comm_r", 5)),
                "in_size": _model_in_size(cfg), "id_mode": cfg.get("id_mode", "none"),
                "margin_mode": cfg.get("margin_mode", "off"),
                "signal_mode": cfg.get("signal_mode", "off")}
        _ckpt.save_model(os.path.join(cfg.get("ckpt_dir", "results/ckpt"),
                                      f"{name}.model.eqx"), model, meta)

    # --- in-distribution eval on a FRESH held-out set across eval_N ---
    eval_data = _build_pool(eval_N, H, base_seed + 9000, cfg, N_max=max(train_N + eval_N))
    accuracy, conn_acc, flag_acc = _score(model, eval_data)

    # --- 5-fold CV at N = cv_N (default 20) ---
    cv_mean, cv_std = _cv_at_N(cfg, base_seed)

    # --- zero-shot extrapolation ---
    extrap = _extrapolate(model, cfg, base_seed)

    return {
        "config": _json_safe_cfg(cfg),
        "accuracy": accuracy,
        "connected_accuracy": conn_acc,
        "connected_flag_accuracy": flag_acc,
        "cv20_mean": cv_mean,
        "cv20_std": cv_std,
        "extrap": extrap,
        "val_acc": info["val_acc"],
        "val_err": info["val_err"],
        "steps_run": info["steps_run"],
        "wall_s": round(time.time() - t0, 2),
    }


def run_sweep(configs, out_jsonl, base_seed=0):
    """Run each config; append one JSON line per config to `out_jsonl` as it finishes.

    Returns the list of per-config metric dicts. A failing config is recorded with an
    "error" key (so one bad config does not lose the rest of the sweep).
    """
    os.makedirs(os.path.dirname(out_jsonl) or ".", exist_ok=True)
    results = []
    for i, cfg in enumerate(configs):
        try:
            res = run_config(cfg, base_seed=base_seed + i)
        except Exception as e:  # keep partial results; record the failure
            res = {"config": _json_safe_cfg(cfg), "error": repr(e)}
        results.append(res)
        with open(out_jsonl, "a") as fh:
            fh.write(json.dumps(res) + "\n")
    return results
