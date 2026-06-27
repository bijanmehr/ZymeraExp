"""Slice-1 end-to-end: generate small datasets, run the two reference estimators across H,
report accuracy-vs-H, and write a self-documenting experiment record.

Run with the zymera venv:
    zymera_lab/.venv/bin/python run_slice.py
"""
import json
import os
from datetime import datetime, timezone
import numpy as np

from fiedler.config import DataCfg
from fiedler import datagen, metrics
from fiedler.methods import power_iteration as pi
from fiedler.methods import degree_regression as dr

EXP_ID = "slice1_reference"
PURPOSE = ("Slice-1 pipeline validation: accuracy-vs-H for the two reference estimators "
           "(analytic decentralized power-iteration + degree-regression floor).")


def run_slice(n_agents_list=(4, 8), H_list=(1, 2, 3, 5), n_episodes=4, n_steps=50,
              grid=16, comm_r=5, pi_eps=0.1, pi_rounds_per_H=20, dr_degree=2, dr_ridge=1e-4,
              out_path="results/accuracy_vs_h.json", record_path=None):
    feats, adjs, lams = [], [], []
    for n in n_agents_list:
        ds = datagen.generate_dataset(DataCfg(n_agents=n, grid=grid, comm_r=comm_r,
                                              n_episodes=n_episodes, n_steps=n_steps, seed=n))
        feats.append(ds["features"].reshape(-1, n, 6))
        adjs.append(ds["adjacency"].reshape(-1, n, n))
        lams.append(ds["lambda2"].reshape(-1))

    # degree-regression floor: mean degree per graph -> lambda2
    deg_X = np.concatenate([f[:, :, 0].mean(1) for f in feats])
    lam_y = np.concatenate(lams)
    dr_model = dr.fit(deg_X, lam_y, degree=dr_degree, ridge=dr_ridge)
    dr_acc = metrics.accuracy(dr.predict(dr_model, deg_X), lam_y)

    # power-iteration: estimate per graph for each H (rounds = H * pi_rounds_per_H)
    pi_acc = {}
    for H in H_list:
        preds, trues = [], []
        for a_group, l_group in zip(adjs, lams):
            for g in range(a_group.shape[0]):
                preds.append(float(pi.estimate(np.asarray(a_group[g]),
                                               n_rounds=int(H) * pi_rounds_per_H, eps=pi_eps, seed=0)))
                trues.append(float(l_group[g]))
        pi_acc[H] = metrics.accuracy(np.array(preds), np.array(trues))

    result = {
        "power_iteration": [pi_acc[H] for H in H_list],
        "degree_regression": [dr_acc for _ in H_list],   # flat floor
    }

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)

    if record_path is not None:
        record = {
            "experiment": EXP_ID,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "purpose": PURPOSE,
            "config": {
                "n_agents_list": list(n_agents_list), "H_list": list(H_list),
                "n_episodes": n_episodes, "n_steps": n_steps, "grid": grid, "comm_r": comm_r,
                "power_iteration": {"eps": pi_eps, "rounds_per_H": pi_rounds_per_H},
                "degree_regression": {"poly_degree": dr_degree, "ridge": dr_ridge},
                "policy": "zymera.random_policy",
                "label": "true_lambda2 = eigvalsh(L)[1] on potential adjacency",
            },
            "results": {"H_list": list(H_list), **result},
        }
        os.makedirs(os.path.dirname(record_path) or ".", exist_ok=True)
        with open(record_path, "w") as fh:
            json.dump(record, fh, indent=2)

    return result


if __name__ == "__main__":
    res = run_slice(record_path="experiments/slice1_reference.json")
    print(json.dumps(res, indent=2))
