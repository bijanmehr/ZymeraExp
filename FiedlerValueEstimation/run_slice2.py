"""Slice-2a end-to-end: single-N learned lambda2 estimators (GRU own-history + GCRN
spatio-temporal) vs the analytic power-iteration baseline, across history length H.

For each H we window the rolled-out episodes (split by episode -> no leakage), train both
Equinox estimators on the train episodes, and report **test** accuracy (1 - median rel-err
in linear space) over per-(sample, node) predictions vs the broadcast true lambda2. The
power-iteration baseline runs on each test window's last-step adjacency (rounds = H * rounds_per_H).

Run with the zymera venv:
    zymera_lab/.venv/bin/python run_slice2.py
"""
import json
import os
from datetime import datetime, timezone

import jax
import numpy as np

from fiedler.config import DataCfg
from fiedler import datagen, dataset, metrics
from fiedler.models_eqx import GRUEstimator, GCRNEstimator
from fiedler import train_eqx
from fiedler.methods import power_iteration as pi

EXP_ID = "slice2_singleN"
PURPOSE = ("Slice-2a single-N: accuracy-vs-H for two learned estimators (per-agent GRU "
           "own-history + GCRN spatio-temporal, normalized-mean aggregation) against the "
           "analytic decentralized power-iteration baseline, on held-out test episodes.")


def _episode_split(n_episodes, test_frac=0.25, seed=0):
    """Split episode indices into (train, test); guarantee >=1 test and >=1 train."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_episodes)
    n_test = max(1, int(round(n_episodes * test_frac)))
    n_test = min(n_test, n_episodes - 1) if n_episodes > 1 else 0
    return np.sort(perm[n_test:]), np.sort(perm[:n_test])


def run_slice2(n_agents=16, H_list=(1, 3, 5, 8), n_episodes=8, n_steps=100,
               grid=16, comm_r=5, spawn_radius=2, n_obstacles=0,
               steps=1500, lr=3e-4, hidden=64, batch=128,
               pi_eps=0.1, pi_rounds_per_H=20, seed=0,
               out_path="results/accuracy_vs_h_slice2.json", record_path=None):
    ds = datagen.generate_dataset(DataCfg(
        n_agents=n_agents, grid=grid, comm_r=comm_r, n_obstacles=n_obstacles,
        spawn_radius=spawn_radius, n_episodes=n_episodes, n_steps=n_steps, seed=seed))
    feats = ds["features"]          # (E, T+1, N, 6)
    adjs = ds["adjacency"]          # (E, T+1, N, N) bool
    lams = ds["lambda2"]            # (E, T+1)
    N = int(ds["n_agents"])

    tr_ep, te_ep = _episode_split(feats.shape[0], test_frac=0.25, seed=seed)

    gru_acc, gcrn_acc, pi_acc = [], [], []
    for H in H_list:
        # window train / test episodes separately (same order for node + adj windows)
        Xn_tr, y_tr = dataset.make_windows(feats[tr_ep], lams[tr_ep], H)
        Xa_tr = dataset.make_adj_windows(adjs[tr_ep], H)
        Xn_te, y_te = dataset.make_windows(feats[te_ep], lams[te_ep], H)
        Xa_te = dataset.make_adj_windows(adjs[te_ep], H)
        true_bcast = np.broadcast_to(y_te[:, None], (y_te.shape[0], N))  # (S, N)

        # --- GRU (own-history) ---
        gru = GRUEstimator(in_size=6, hidden=hidden, key=jax.random.PRNGKey(seed))
        gru, _ = train_eqx.train(gru, Xn_tr, Xa_tr, y_tr, steps=steps, lr=lr,
                                 batch=batch, seed=seed)
        gru_pred = train_eqx.predict(gru, Xn_te, Xa_te)                  # (S, N) linear
        gru_acc.append(metrics.accuracy(gru_pred.reshape(-1), true_bcast.reshape(-1)))

        # --- GCRN (spatio-temporal) ---
        gcrn = GCRNEstimator(in_size=6, hidden=hidden, key=jax.random.PRNGKey(seed + 1))
        gcrn, _ = train_eqx.train(gcrn, Xn_tr, Xa_tr, y_tr, steps=steps, lr=lr,
                                  batch=batch, seed=seed)
        gcrn_pred = train_eqx.predict(gcrn, Xn_te, Xa_te)
        gcrn_acc.append(metrics.accuracy(gcrn_pred.reshape(-1), true_bcast.reshape(-1)))

        # --- power-iteration baseline (per test window's last-step adjacency) ---
        last_adj = Xa_te[:, -1]                                          # (S, N, N)
        pi_preds = np.array([float(pi.estimate(np.asarray(last_adj[g]),
                                               n_rounds=int(H) * pi_rounds_per_H,
                                               eps=pi_eps, seed=0))
                             for g in range(last_adj.shape[0])])
        pi_acc.append(metrics.accuracy(pi_preds, y_te))

    result = {"gru": gru_acc, "gcrn": gcrn_acc, "power_iteration": pi_acc}

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    if record_path is not None:
        record = {
            "experiment": EXP_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": PURPOSE,
            "config": {
                "n_agents": n_agents, "H_list": list(H_list), "n_episodes": n_episodes,
                "n_steps": n_steps, "grid": grid, "comm_r": comm_r,
                "spawn_radius": spawn_radius, "n_obstacles": n_obstacles,
                "train_episodes": tr_ep.tolist(), "test_episodes": te_ep.tolist(),
                "models": {
                    "gru": {"type": "GRUEstimator (per-agent own-history GRU)",
                            "hidden": hidden, "in_size": 6},
                    "gcrn": {"type": "GCRNEstimator (spatio-temporal, normalized-mean agg)",
                             "hidden": hidden, "in_size": 6, "mp_rounds": 1},
                },
                "training": {"steps": steps, "lr": lr, "batch": batch,
                             "optimizer": "adamw + clip_by_global_norm(1.0)",
                             "loss": "Huber(delta=1) on log(lambda2), target broadcast over nodes",
                             "early_stop": "val median-rel-err (linear), patience 10"},
                "power_iteration": {"eps": pi_eps, "rounds_per_H": pi_rounds_per_H,
                                    "evaluated_on": "test window last-step adjacency"},
                "policy": "zymera.random_policy",
                "label": "true_lambda2 = eigvalsh(L)[1] on potential adjacency",
                "metric": "accuracy = 1 - median(|pred - true| / true) in linear space",
                "seed": seed,
            },
            "results": {"H_list": list(H_list), **result},
        }
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        with open(record_path, "w") as fh:
            json.dump(record, fh, indent=2)

    return result


if __name__ == "__main__":
    res = run_slice2(record_path="experiments/slice2_singleN.json")
    print("accuracy-vs-H (single N):")
    print(json.dumps(res, indent=2))
