"""Render a trained MVProp-controller policy to a GIF — for visual audit (Bijan's requirement).

Same un-jitted rollout as ctde_v0.render, but loads the frozen distilled planner and threads it
into ``ppo._goal_to_move(..., mvplanner=)`` so the moves are the LEARNED planner's, matching the
trained pipeline exactly. Colours agents by role when the role picker is on.

    PYTHONPATH=.:../../../FiedlerValueEstimation $PY -m t5lab.render_mvprop \
        --run-dir runs/mvprop/open_32_seed0 --out gifs/mvprop_open32.gif --seed 3
"""
import argparse
import json
import os
import dataclasses
import jax
import jax.numpy as jnp

from ctde_v0 import env_utils, ppo
from ctde_v0.config import from_dict
from t5lab.planner import load_planner
from zymera.viz import render_gif


def render_run(run_dir, out, *, steps=100, seed=0, fps=8, planner_path=None, kprop=32,
               terrain=None, n_obstacles=40, rooms=3, corner=None):
    cfg = from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    if steps:
        cfg = dataclasses.replace(cfg, world=dataclasses.replace(cfg.world, horizon=steps))
    if terrain:                                            # zero-shot terrain override
        cfg = dataclasses.replace(cfg, world=dataclasses.replace(
            cfg.world, terrain=terrain, n_obstacles=n_obstacles, rooms=rooms))
    # planner path: explicit arg > run's mvprop_run.json > default beside run_dir
    if planner_path is None:
        meta = os.path.join(run_dir, "mvprop_run.json")
        if os.path.exists(meta):
            info = json.load(open(meta))
            planner_path, kprop = info["planner"], info.get("kprop", kprop)
        else:
            planner_path = "mvprop_distilled.eqx"
    planner = load_planner(planner_path, in_ch=2, K=kprop, gamma=0.9)

    env = env_utils.build_env(cfg)
    state = ppo.init_state_from_checkpoint(env, cfg, os.path.join(run_dir, "model.eqx"),
                                           jax.random.PRNGKey(seed))
    actor = state.actor
    stencil = ppo.make_stencil(cfg)
    edge_msg = cfg.backbone.message_content != "learned"
    use_roles = cfg.role_picker == "expl_relay"

    k = jax.random.PRNGKey(seed + 7)
    rk0, kk = jax.random.split(k)
    obs, st = env.reset(rk0)
    if corner:
        # override the (wall-straddling) cluster spawn with a CONTIGUOUS block in one corner:
        # BFS-flood N free cells from the corner-most free cell -> one region, one side.
        import numpy as _np
        from collections import deque
        wall = _np.asarray(st.wall).astype(bool); H, W = wall.shape
        cr = {"tl": range(H), "bl": range(H - 1, -1, -1)}.get(corner, range(H))
        cc = {"tl": range(W), "bl": range(W)}.get(corner, range(W))
        start = next(((r, c) for r in cr for c in cc if not wall[r, c]), (0, 0))
        seen = {start}; q = deque([start]); cells = []
        n = int(st.n_agents)
        while q and len(cells) < n:
            r, c = q.popleft(); cells.append((r, c))
            for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W and not wall[nr, nc] and (nr, nc) not in seen:
                    seen.add((nr, nc)); q.append((nr, nc))
        pos = jnp.asarray(_np.array(cells[:n], dtype=_np.int32))
        st = st.replace(body=st.body.replace(position=pos))
    h = actor.init_hidden(st.n_agents)
    worlds = [st]
    for _t in range(cfg.world.horizon):
        kk, ak, rk, sk = jax.random.split(kk, 4)
        adj = env_utils.kb_adjacency(st.body.position, cfg)
        dist = env_utils.kb_distance(st.body.position, cfg) if edge_msg else None
        goal_logits, role_logits, _v, _l2, _z, h = actor(obs, adj, dist=dist, h=h, inference=True)
        tag = None
        if use_roles:
            role_idx = jax.random.categorical(rk, role_logits, axis=-1)
            tag = role_idx.astype(jnp.int32)
        else:
            role_idx = None
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        masked = jnp.where(gmask, goal_logits, ppo._NEG)
        goal = jax.random.categorical(ak, masked, axis=-1)
        move, _ = ppo._goal_to_move(env, st, goal, stencil, role_idx, cfg, planner)   # <-- planner
        obs, st, _r, _d, _i = env.step(st, move, sk)
        if tag is not None:
            st = st.replace(group=tag)
        worlds.append(st)

    traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
    # metrics of THIS rollout: final coverage of free cells + connectivity_real (λ₂>real_threshold).
    import numpy as np
    pos = traj.body.position                                           # (T+1, N, 2)
    l2 = jax.vmap(lambda p: env_utils.true_lambda2(p, cfg))(pos)       # (T+1,)
    conn_real = float((np.asarray(l2) > cfg.connectivity.real_threshold).mean())
    # coverage @ cover_r=0 = fraction of FREE cells any agent stood on over the episode.
    # (state.covered is NOT the accumulated map — count unique visited cells from positions.)
    free = ~np.asarray(traj.wall[-1]).astype(bool)
    P = np.asarray(pos).reshape(-1, 2).astype(int)
    visited = np.zeros_like(free); visited[P[:, 0], P[:, 1]] = True
    cov = float((visited & free).sum()) / float(free.sum())
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    render_gif(traj, out, comm_radius=cfg.world.comm_r, fps=fps)
    print(f"METRICS terrain={cfg.world.terrain} ctrl={cfg.action_head.controller} "
          f"seed={seed} coverage={cov*100:.1f}% connectivity_real={conn_real:.3f}", flush=True)
    print(f"saved {out}", flush=True)
    return {"out": out, "coverage": cov, "connectivity_real": conn_real,
            "terrain": cfg.world.terrain, "controller": cfg.action_head.controller}


def main(argv=None):
    p = argparse.ArgumentParser(description="render a trained MVProp-controller policy to a GIF")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--planner", default=None, help="planner .eqx (default: from run's mvprop_run.json)")
    p.add_argument("--terrain", default=None, help="zero-shot terrain override (rooms/walls/clutter/mixed)")
    p.add_argument("--n-obstacles", type=int, default=40)
    p.add_argument("--rooms", type=int, default=3)
    p.add_argument("--corner", default=None, choices=["tl", "bl"],
                   help="spawn all agents in one contiguous corner block (tl=top-left, bl=bottom-left)")
    a = p.parse_args(argv)
    render_run(a.run_dir, a.out, steps=a.steps, seed=a.seed, fps=a.fps, planner_path=a.planner,
               terrain=a.terrain, n_obstacles=a.n_obstacles, rooms=a.rooms, corner=a.corner)


if __name__ == "__main__":
    main()
