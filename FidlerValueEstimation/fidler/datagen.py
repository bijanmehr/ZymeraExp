"""Roll out episodes -> per-step features, adjacency, and true lambda2 -> .npz."""
import jax
import numpy as np
import zymera

from .config import DataCfg
from . import fiedler, features


def generate_dataset(cfg: DataCfg) -> dict:
    env = zymera.make("comm-coverage", grid=cfg.grid, n_agents=cfg.n_agents,
                      comm_r=cfg.comm_r, n_obstacles=cfg.n_obstacles, spawn_radius=cfg.spawn_radius)

    def one_episode(key):
        traj = zymera.rollout(env, zymera.random_policy, n_steps=cfg.n_steps, key=key, keep="all")
        pos = traj["world"].body.position                      # (T+1, N, 2)

        def per_step(p):
            adj = fiedler.potential_adjacency(p, cfg.comm_r)    # (N,N) bool
            return (features.node_features(p, adj, cfg.comm_r), adj, fiedler.true_lambda2(adj))

        feats, adj, lam = jax.vmap(per_step)(pos)               # (T+1,N,6),(T+1,N,N),(T+1,)
        return feats, adj, lam

    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.n_episodes)
    feats, adj, lam = jax.vmap(one_episode)(keys)               # leading (E, T+1, ...)
    return {
        "features": np.asarray(feats, np.float32),
        "adjacency": np.asarray(adj, bool),
        "lambda2": np.asarray(lam, np.float32),
        "n_agents": np.int32(cfg.n_agents),
        "comm_r": np.int32(cfg.comm_r),
    }


def save_npz(path: str, ds: dict) -> None:
    np.savez_compressed(path, **ds)
