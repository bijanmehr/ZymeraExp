"""Continuous warm-started power-iteration TRACKING: does it hold precision at FEW rounds/step?

The cold-start cost (~128 rounds at N=20) does not fit a 100-step mission. But you do not
cold-start every step -- the comm graph barely moves between consecutive steps, so a warm-started
iterate (carried forward) should stay locked onto the slowly-drifting Fiedler vector with just a
couple of rounds per step. This measures tracking accuracy at K rounds/step (K=1,2,4,8) carried
across a moving guardrail episode, vs cold-start at the SAME K. If warm tracking holds at K=1-2,
the mission gets lambda2 continuously at ~1-2 comm rounds/step (amortized) instead of 128.
"""
import jax
import jax.numpy as jnp
import numpy as np

from fiedler.config import DataCfg
from fiedler import datagen
from fiedler.fiedler import _laplacian


def grid_for_n(n):
    return max(8, round((n / 0.04) ** 0.5))


def _iterate(x, lap, rounds, eps=0.1):
    for _ in range(rounds):
        x = x - jnp.mean(x)                         # deflate constant (consensus)
        x = x - eps * (lap @ x)                     # local diffusion
        x = x / (jnp.linalg.norm(x) + 1e-12)
    return x


def _rayleigh(x, lap):
    return float((x @ lap @ x) / (x @ x + 1e-12))


NS = [8, 16, 20, 30]
KS = [1, 2, 4, 8]


def main():
    print("=== warm-started TRACKING vs COLD-start, accuracy (1 - median rel-err), K rounds/step ===",
          flush=True)
    for N in NS:
        cfg = DataCfg(n_agents=N, grid=grid_for_n(N), comm_r=5, n_episodes=2, n_steps=80, seed=0)
        ds = datagen.generate_dataset_guardrail(cfg)
        adj, lam = ds["adjacency"], ds["lambda2"]            # (E,T,N,N) bool, (E,T)
        cells = []
        for K in KS:
            warm_e, cold_e = [], []
            for e in range(adj.shape[0]):
                x = jax.random.normal(jax.random.PRNGKey(0), (N,))   # carried across the episode
                for t in range(adj.shape[1]):
                    L = _laplacian(jnp.asarray(adj[e, t]))
                    x = _iterate(x, L, K)                            # WARM: carry x forward
                    est_w = _rayleigh(x, L)
                    xc = _iterate(jax.random.normal(jax.random.PRNGKey(1), (N,)), L, K)  # COLD
                    est_c = _rayleigh(xc, L)
                    tr = float(lam[e, t])
                    if tr > 1e-3 and t >= 5:                         # skip the startup ramp
                        warm_e.append(abs(est_w - tr) / max(tr, 1e-6))
                        cold_e.append(abs(est_c - tr) / max(tr, 1e-6))
            aw = 1.0 - float(np.median(warm_e)) if warm_e else float("nan")
            ac = 1.0 - float(np.median(cold_e)) if cold_e else float("nan")
            cells.append((K, aw, ac))
        print(f"N={N:>3} | " + "  ".join(f"K={K}: warm={aw:.3f} cold={ac:.3f}" for K, aw, ac in cells),
              flush=True)


if __name__ == "__main__":
    main()
