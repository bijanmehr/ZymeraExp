"""The permutation idea + cheap ensembling at the mission-affordable LOW-round regime.

Two things, both no-training, on guardrail comm graphs:
1. PERMUTATION: confirm the decentralized estimator is invariant to relabeling agents -- run
   power-iteration on a graph and on K random node-permutations of it; the lambda2 estimate must
   be identical (permutation diversity gives NOTHING to ensemble for an invariant estimator).
2. ENSEMBLE: since the low-round regime (K=2-8 rounds, what a 100-step mission can afford) is where
   precision collapses, test whether ensembling M independent runs rescues it. For power-iteration
   the real diversity is the random INIT (not permutation), so we ensemble M random restarts and
   take the median -- does that lift low-round accuracy toward usable?
"""
import jax
import jax.numpy as jnp
import numpy as np

from fiedler.config import DataCfg
from fiedler import datagen, metrics
from fiedler.methods import power_iteration as pi


def grid_for_n(n):
    return max(8, round((n / 0.04) ** 0.5))


NS = [8, 16, 20]
KS = [2, 4, 8]
M = 8                                            # ensemble size (random restarts)


def main():
    # 1) permutation-invariance: estimate is identical under relabeling -> no permutation diversity
    print("=== permutation-invariance of the estimator (should be ~0 spread) ===", flush=True)
    cfg = DataCfg(n_agents=16, grid=grid_for_n(16), comm_r=5, n_episodes=1, n_steps=4, seed=3)
    adj = jnp.asarray(datagen.generate_dataset_guardrail(cfg)["adjacency"][0, 2])
    rng = np.random.RandomState(0)
    ests = []
    for _ in range(6):
        p = jnp.asarray(rng.permutation(16))
        ests.append(float(pi.estimate(adj[p][:, p], 64)))
    print(f"  6 permutations -> estimates spread = {max(ests) - min(ests):.2e} "
          f"(mean {np.mean(ests):.5f}) -> permutation gives no ensemble diversity", flush=True)

    # 2) random-restart ensemble at the mission-affordable low-round regime
    print("\n=== ensemble (M=8 random restarts, median) vs single-shot, low rounds ===", flush=True)
    for N in NS:
        cfg = DataCfg(n_agents=N, grid=grid_for_n(N), comm_r=5, n_episodes=3, n_steps=40, seed=0)
        ds = datagen.generate_dataset_guardrail(cfg)
        adjs = jnp.asarray(ds["adjacency"].reshape(-1, N, N))
        trues = np.asarray(ds["lambda2"].reshape(-1))
        keep = trues > 1e-3
        adjs, trues = adjs[np.asarray(keep)], trues[keep]
        parts = []
        for K in KS:
            single = np.asarray(jax.vmap(lambda a: pi.estimate(a, K, seed=0))(adjs))
            ens = np.stack([np.asarray(jax.vmap(lambda a: pi.estimate(a, K, seed=s))(adjs))
                            for s in range(M)])
            ens_med = np.median(ens, axis=0)
            a1 = float(metrics.accuracy(single, trues))
            aM = float(metrics.accuracy(ens_med, trues))
            parts.append(f"K={K}: 1-shot={a1:.3f} ens{M}={aM:.3f}")
        print(f"  N={N:>3} | " + "  |  ".join(parts), flush=True)


if __name__ == "__main__":
    main()
