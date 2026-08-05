"""Wall-occlusion ("wall RF") eval: run a trained mvprop policy with occlusion OFF vs ON and
report the connectivity / coverage gap. OFF = status-quo distance comms (what we trained/graded on);
ON = walls inflate effective comm distance (d_eff = d + c*k), so links through walls attenuate/drop.
Same policy in both — measures how a non-occlusion-trained policy fares when RF is actually blocked.

    PYTHONPATH=.:../../../FiedlerValueEstimation RUN_DIR=runs/mvprop/rooms24_mvprop_setattn \
        SEEDS=8 $PY -m t5lab.occ_eval
"""
import os, json, dataclasses
import numpy as np
import jax, jax.numpy as jnp

from ctde_v0 import env_utils, ppo
from ctde_v0.config import from_dict
from t5lab.planner import load_planner

RUN_DIR = os.environ["RUN_DIR"]
SEEDS = int(os.environ.get("SEEDS", 8))
KPROP = int(os.environ.get("KPROP", 32))


def rollout_once(cfg, planner, actor_state, seed):
    """One un-jitted episode; returns (coverage_frac, connectivity_real_frac)."""
    env = env_utils.build_env(cfg)
    state0 = ppo.init_state_from_checkpoint(env, cfg, os.path.join(RUN_DIR, "model.eqx"),
                                            jax.random.PRNGKey(seed))
    actor = state0.actor
    stencil = ppo.make_stencil(cfg)
    edge_msg = cfg.backbone.message_content != "learned"
    use_roles = cfg.role_picker == "expl_relay"
    k = jax.random.PRNGKey(seed + 7)
    rk0, kk = jax.random.split(k)
    obs, st = env.reset(rk0)
    h = actor.init_hidden(st.n_agents)
    worlds = [st]
    for _t in range(cfg.world.horizon):
        kk, ak, rk, sk = jax.random.split(kk, 4)
        adj = env_utils.kb_adjacency(st.body.position, cfg, st.wall)             # <- occlusion-aware
        dist = env_utils.kb_distance(st.body.position, cfg, st.wall) if edge_msg else None
        goal_logits, role_logits, _v, _l2, _z, h = actor(obs, adj, dist=dist, h=h, inference=True)
        role_idx = jax.random.categorical(rk, role_logits, axis=-1).astype(jnp.int32) if use_roles else None
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        goal = jax.random.categorical(ak, jnp.where(gmask, goal_logits, ppo._NEG), axis=-1)
        move, _ = ppo._goal_to_move(env, st, goal, stencil, role_idx, cfg, planner)
        obs, st, _r, _d, _i = env.step(st, move, sk)
        worlds.append(st)
    traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
    pos = traj.body.position                                                    # (T+1,N,2)
    wall_last = traj.wall[-1]
    l2 = jax.vmap(lambda p: env_utils.true_lambda2(p, cfg, wall_last))(pos)      # (T+1,) occlusion-aware
    conn_real = float((np.asarray(l2) > cfg.connectivity.real_threshold).mean())
    free = ~np.asarray(wall_last).astype(bool)
    P = np.asarray(pos).reshape(-1, 2).astype(int)
    visited = np.zeros_like(free); visited[P[:, 0], P[:, 1]] = True
    cov = float((visited & free).sum()) / float(free.sum())
    return cov, conn_real


def main():
    cfg0 = from_dict(json.load(open(os.path.join(RUN_DIR, "config.json"))))
    meta = os.path.join(RUN_DIR, "mvprop_run.json")
    ppath = json.load(open(meta))["planner"] if os.path.exists(meta) else "mvprop_distilled.eqx"
    planner = load_planner(ppath, in_ch=2, K=KPROP, gamma=0.9) if cfg0.action_head.controller == "mvprop" else None
    print(f"=== occlusion eval [{RUN_DIR}] terrain={cfg0.world.terrain} grid={cfg0.world.grid} "
          f"comm_r={cfg0.world.comm_r} c=comm_r/3={cfg0.world.comm_r/3:.1f} seeds={SEEDS} ===", flush=True)
    out = {}
    for occ in (False, True):
        cfg = dataclasses.replace(cfg0, world=dataclasses.replace(cfg0.world, occlusion=occ))
        covs, conns = [], []
        for s in range(SEEDS):
            c, k = rollout_once(cfg, planner, None, 100 + s)
            covs.append(c); conns.append(k)
        out[occ] = (float(np.mean(covs)), float(np.mean(conns)))
        tag = "ON (walls block RF)" if occ else "OFF (distance comms)"
        print(f"  occlusion {tag:22s}: coverage={100*out[occ][0]:5.1f}%   connectivity_real={out[occ][1]:.3f}", flush=True)
    dcov = 100 * (out[True][0] - out[False][0]); dconn = out[True][1] - out[False][1]
    print(f"  --> Δ under occlusion:  coverage {dcov:+.1f} pts   connectivity_real {dconn:+.3f}", flush=True)
    print("=== OCC EVAL DONE ===", flush=True)


if __name__ == "__main__":
    main()
