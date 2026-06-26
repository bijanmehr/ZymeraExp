"""Roll out episodes -> per-step features, adjacency, and true lambda2 -> .npz."""
import jax
import numpy as np
import zymera

from .config import DataCfg
from . import fiedler, features, policies


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
        return feats, adj, lam, pos                             # also return positions (T+1,N,2)

    keys = jax.random.split(jax.random.PRNGKey(cfg.seed), cfg.n_episodes)
    feats, adj, lam, pos = jax.vmap(one_episode)(keys)          # leading (E, T+1, ...)
    return {
        "features": np.asarray(feats, np.float32),
        "adjacency": np.asarray(adj, bool),
        "lambda2": np.asarray(lam, np.float32),
        "positions": np.asarray(pos, np.float32),               # (E, T+1, N, 2)
        "n_agents": np.int32(cfg.n_agents),
        "comm_r": np.int32(cfg.comm_r),
    }


# --------------------------------------------------------------------------------------
# hard-connectivity-guardrail dataset (realistic always-connected, dispersed regime)
# --------------------------------------------------------------------------------------
def _per_step_arrays(positions, comm_r):
    """positions (N,2) int -> (node_features (N,6), adjacency (N,N) bool, lambda2 float)."""
    p = np.asarray(positions, np.int32)
    adj = fiedler.potential_adjacency(p, comm_r)               # (N,N) bool (Chebyshev)
    feats = features.node_features(p, adj, comm_r)             # (N,6)
    lam = fiedler.true_lambda2(adj)                            # scalar
    return (np.asarray(feats, np.float32),
            np.asarray(adj, bool),
            float(lam))


def generate_dataset_guardrail(cfg: DataCfg) -> dict:
    """Roll out episodes under the HARD-CONNECTIVITY GUARDRAIL dispersion policy.

    A plain Python loop (not jit) over `cfg.n_episodes` x `cfg.n_steps`: reset to the
    clustered spawn (spawn_radius -> a connected start), then at every step pick the
    guardrail dispersion actions (`policies.guardrail_disperse_actions`) and step the
    env. Records the reset frame + each stepped frame (T+1 total) of node features,
    adjacency, true lambda2, and integer positions.

    Returns the SAME dict schema as `generate_dataset` plus
    `connected_frac` = fraction of all (episode,step) with lambda2 > 1e-3.
    """
    env = zymera.make("comm-coverage", grid=cfg.grid, n_agents=cfg.n_agents,
                      comm_r=cfg.comm_r, n_obstacles=cfg.n_obstacles,
                      spawn_radius=cfg.spawn_radius)
    T = cfg.n_steps + 1
    N = cfg.n_agents

    feats_E = np.zeros((cfg.n_episodes, T, N, 6), np.float32)
    adj_E = np.zeros((cfg.n_episodes, T, N, N), bool)
    lam_E = np.zeros((cfg.n_episodes, T), np.float32)
    pos_E = np.zeros((cfg.n_episodes, T, N, 2), np.float32)

    base_key = jax.random.PRNGKey(cfg.seed)
    for e in range(cfg.n_episodes):
        ep_key = jax.random.fold_in(base_key, e)
        reset_key, ep_key = jax.random.split(ep_key)
        _, state = env.reset(reset_key)
        pos = np.asarray(state.body.position, np.int32)         # (N,2)

        # frame 0 = the reset state
        f, a, l = _per_step_arrays(pos, cfg.comm_r)
        feats_E[e, 0], adj_E[e, 0], lam_E[e, 0], pos_E[e, 0] = f, a, l, pos

        for t in range(1, T):
            act_key, step_key, ep_key = jax.random.split(ep_key, 3)
            actions = policies.guardrail_disperse_actions(
                pos, grid=cfg.grid, comm_r=cfg.comm_r, key=act_key)
            _, state, _, _, _ = env.step(state, np.asarray(actions, np.int32), step_key)
            pos = np.asarray(state.body.position, np.int32)
            f, a, l = _per_step_arrays(pos, cfg.comm_r)
            feats_E[e, t], adj_E[e, t], lam_E[e, t], pos_E[e, t] = f, a, l, pos

    connected_frac = float(np.mean(lam_E > fiedler.CONNECTED_TAU))
    return {
        "features": feats_E,
        "adjacency": adj_E,
        "lambda2": lam_E,
        "positions": pos_E,
        "n_agents": np.int32(cfg.n_agents),
        "comm_r": np.int32(cfg.comm_r),
        "connected_frac": np.float32(connected_frac),
    }


def generate_multi_guardrail(n_agents_list, grid_for_n, *, comm_r, n_episodes,
                             n_steps, seed0) -> list:
    """Guardrail datasets across agent-counts, one `generate_dataset_guardrail` per N.

    `grid_for_n(N) -> int` sets each N's grid side (e.g. to hold density roughly fixed).
    Each N gets its own seed (seed0 + i) so the episodes differ. Returns a list of the
    per-N dataset dicts in the order of `n_agents_list`.
    """
    out = []
    for i, N in enumerate(n_agents_list):
        grid = int(grid_for_n(N))
        cfg = DataCfg(n_agents=int(N), grid=grid, comm_r=int(comm_r),
                      n_obstacles=0, spawn_radius=2,
                      n_episodes=int(n_episodes), n_steps=int(n_steps),
                      seed=int(seed0) + i)
        out.append(generate_dataset_guardrail(cfg))
    return out


def save_npz(path: str, ds: dict) -> None:
    np.savez_compressed(path, **ds)
