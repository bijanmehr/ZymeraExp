"""Verify the WORKING estimator: decentralized power-iteration precision vs (rounds, N).

No training -- runs the decentralized algorithm on guardrail comm graphs and compares to the
oracle lambda2 (eigendecomposition). Answers the deployment question: how many communication
rounds buy high precision at each team size N -- i.e. the precision-vs-communication-cost curve
for the team (this is exactly RQ5: comm/energy budget vs connectivity-estimation quality).
Also a permutation-invariance sanity check (true lambda2 and the estimate must be invariant to
relabeling the agents).

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u verify_power.py
"""
import json

import jax
import jax.numpy as jnp
import numpy as np

from fiedler.config import DataCfg
from fiedler import datagen, metrics
from fiedler.fiedler import true_lambda2
from fiedler.methods import power_iteration as pi


def grid_for_n(n):
    return max(8, round((n / 0.04) ** 0.5))          # ~fixed 0.04 agents/cell, as in the sweeps


NS = [4, 8, 12, 16, 20, 24, 30]
ROUNDS = [2, 4, 8, 16, 32, 64, 128]


def main():
    print("=== decentralized power-iteration: accuracy (1 - median rel-err) vs (N, rounds) ===",
          flush=True)
    print(f"{'N':>4} | " + "  ".join(f"r={r:<3}" for r in ROUNDS), flush=True)
    summary = {}
    for N in NS:
        cfg = DataCfg(n_agents=N, grid=grid_for_n(N), comm_r=5, n_episodes=4, n_steps=50, seed=0)
        ds = datagen.generate_dataset_guardrail(cfg)
        adjs = jnp.asarray(ds["adjacency"].reshape(-1, N, N))         # (B,N,N) bool
        trues = np.asarray(ds["lambda2"].reshape(-1))                 # (B,)
        keep = trues > 1e-3
        adjs, trues = adjs[np.asarray(keep)], trues[keep]
        row = {}
        for r in ROUNDS:
            ests = np.asarray(jax.vmap(lambda a: pi.estimate(a, r))(adjs))
            row[r] = float(metrics.accuracy(ests, trues))
        summary[str(N)] = row
        print(f"{N:>4} | " + "  ".join(f"{row[r]:.3f}" for r in ROUNDS), flush=True)

    print("\n=== rounds needed for >=0.95 / >=0.99 per team size ===", flush=True)
    for N in NS:
        r95 = next((r for r in ROUNDS if summary[str(N)][r] >= 0.95), ">128")
        r99 = next((r for r in ROUNDS if summary[str(N)][r] >= 0.99), ">128")
        print(f"  N={N:>3}:  >=0.95 at {r95} rounds   |   >=0.99 at {r99} rounds", flush=True)

    # permutation-invariance sanity check
    cfg = DataCfg(n_agents=12, grid=grid_for_n(12), comm_r=5, n_episodes=1, n_steps=5, seed=1)
    ds = datagen.generate_dataset_guardrail(cfg)
    adj = jnp.asarray(ds["adjacency"][0, 1])
    perm = np.random.RandomState(0).permutation(12)
    adj_p = adj[jnp.asarray(perm)][:, jnp.asarray(perm)]
    t0, tp = float(true_lambda2(adj)), float(true_lambda2(adj_p))
    e0, ep = float(pi.estimate(adj, 64)), float(pi.estimate(adj_p, 64))
    print(f"\n=== permutation-invariance: true {t0:.6f} vs {tp:.6f} (delta {abs(t0 - tp):.1e}) | "
          f"power-iter {e0:.6f} vs {ep:.6f} (delta {abs(e0 - ep):.1e}) ===", flush=True)

    with open("results/verify_power.json", "w") as f:
        json.dump(summary, f, indent=2)
    print("\nwrote results/verify_power.json", flush=True)


if __name__ == "__main__":
    main()
