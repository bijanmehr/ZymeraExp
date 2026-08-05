"""Wall-RF ("occlusion") viz: render a trained mvprop policy under occlusion, drawing the
REAL (line-of-sight) comm links solid green and the wall-BLOCKED (in-range but occluded) links
dotted red — so the wall cutting the radio graph is visible. Agents coloured by role
(relay = orange, explorer = blue). Coverage heat + per-frame λ₂ overlay.

    PYTHONPATH=.:../../../FiedlerValueEstimation RUN_DIR=runs/mvprop/rooms24_occ_relayB \
        OUT=gifs/relayB.gif CORNER=bl SEED=3 $PY -m t5lab.render_occ
"""
import os, json, dataclasses
import numpy as np
import jax, jax.numpy as jnp
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from ctde_v0 import env_utils, ppo
from ctde_v0.config import from_dict
from t5lab.planner import load_planner

RUN_DIR = os.environ["RUN_DIR"]
OUT = os.environ.get("OUT", "gifs/occ.gif")
SEED = int(os.environ.get("SEED", 3))
CORNER = os.environ.get("CORNER", "bl")
OCC = os.environ.get("OCC", "1") == "1"
FPS = int(os.environ.get("FPS", 8))


def corner_spawn(st, corner):
    from collections import deque
    wall = np.asarray(st.wall).astype(bool); H, W = wall.shape
    cr = {"tl": range(H), "bl": range(H - 1, -1, -1)}.get(corner, range(H))
    cc = {"tl": range(W), "bl": range(W)}.get(corner, range(W))
    start = next(((r, c) for r in cr for c in cc if not wall[r, c]), (0, 0))
    seen = {start}; q = deque([start]); cells = []; n = int(st.n_agents)
    while q and len(cells) < n:
        r, c = q.popleft(); cells.append((r, c))
        for dr, dc in ((0, 1), (1, 0), (0, -1), (-1, 0)):
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and not wall[nr, nc] and (nr, nc) not in seen:
                seen.add((nr, nc)); q.append((nr, nc))
    pos = jnp.asarray(np.array(cells[:n], dtype=np.int32))
    return st.replace(body=st.body.replace(position=pos))


def rollout(cfg, planner):
    env = env_utils.build_env(cfg)
    state = ppo.init_state_from_checkpoint(env, cfg, os.path.join(RUN_DIR, "model.eqx"),
                                           jax.random.PRNGKey(SEED))
    actor = state.actor; stencil = ppo.make_stencil(cfg)
    edge_msg = cfg.backbone.message_content != "learned"
    use_roles = cfg.role_picker == "expl_relay"
    k = jax.random.PRNGKey(SEED + 7); rk0, kk = jax.random.split(k)
    obs, st = env.reset(rk0)
    if CORNER:
        st = corner_spawn(st, CORNER)
    h = actor.init_hidden(st.n_agents); worlds = [st]
    for _t in range(cfg.world.horizon):
        kk, ak, rk, sk = jax.random.split(kk, 4)
        adj = env_utils.kb_adjacency(st.body.position, cfg, st.wall)
        dist = env_utils.kb_distance(st.body.position, cfg, st.wall) if edge_msg else None
        gl, rl, _v, _l2, _z, h = actor(obs, adj, dist=dist, h=h, inference=True)
        tag = jax.random.categorical(rk, rl, axis=-1).astype(jnp.int32) if use_roles else None
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        goal = jax.random.categorical(ak, jnp.where(gmask, gl, ppo._NEG), axis=-1)
        move, _ = ppo._goal_to_move(env, st, goal, stencil, tag, cfg, planner)
        obs, st, _r, _d, _i = env.step(st, move, sk)
        if tag is not None:
            st = st.replace(group=tag)
        worlds.append(st)
    return worlds, use_roles


def draw(worlds, cfg, use_roles):
    comm_r = cfg.world.comm_r
    thr = cfg.connectivity.real_threshold
    frames = []
    for st in worlds:
        pos = np.asarray(st.body.position).astype(float)
        wall = np.asarray(st.wall).astype(bool)
        explored = np.asarray(st.explored) > 0
        grp = np.asarray(st.group).astype(int) if use_roles else np.zeros(len(pos), int)
        H, W = wall.shape
        fig, ax = plt.subplots(figsize=(4.2, 4.2), dpi=110)
        bg = np.ones((H, W, 3)) * 0.98
        bg[explored] = [0.79, 0.90, 0.82]           # covered = soft green
        bg[wall] = [0.24, 0.24, 0.28]               # walls = dark
        ax.imshow(bg, origin="upper", interpolation="nearest")
        d = np.max(np.abs(pos[:, None] - pos[None]), axis=-1)
        pen = np.asarray(env_utils.occlusion_penalty(jnp.asarray(pos), jnp.asarray(wall), cfg)) if OCC else 0.0
        deff = d + pen
        for i in range(len(pos)):
            for j in range(i + 1, len(pos)):
                if d[i, j] <= comm_r:
                    real = deff[i, j] <= comm_r
                    ax.plot([pos[i, 1], pos[j, 1]], [pos[i, 0], pos[j, 0]],
                            color="#2f9e6e" if real else "#d0533a",
                            lw=1.8 if real else 0.9, ls="-" if real else ":",
                            alpha=0.9 if real else 0.55, zorder=2)
        cols = np.where(grp == 1, "#e08a2b", "#2b6fe0")   # relay orange, explorer blue
        ax.scatter(pos[:, 1], pos[:, 0], c=cols, s=95, edgecolors="black", linewidths=1.3, zorder=3)
        l2 = float(env_utils.true_lambda2(jnp.asarray(pos), cfg, jnp.asarray(wall) if OCC else None))
        cov = 100 * explored.sum() / max((~wall).sum(), 1)
        ok = "connected" if l2 > thr else "BROKEN"
        ax.set_title(f"coverage {cov:.0f}%    λ₂={l2:.2f}  ({ok})", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(H - 0.5, -0.5)
        fig.tight_layout(pad=0.2)
        fig.canvas.draw()
        frames.append(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())
        plt.close(fig)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    imgs = [Image.fromarray(f) for f in frames]
    imgs[0].save(OUT, save_all=True, append_images=imgs[1:], duration=int(1000 / FPS), loop=0, optimize=True)
    return cov, l2


def main():
    cfg0 = from_dict(json.load(open(os.path.join(RUN_DIR, "config.json"))))
    cfg = dataclasses.replace(cfg0, world=dataclasses.replace(cfg0.world, occlusion=OCC))
    meta = os.path.join(RUN_DIR, "mvprop_run.json")
    ppath = json.load(open(meta))["planner"] if os.path.exists(meta) else "mvprop_distilled.eqx"
    planner = load_planner(ppath, in_ch=2, K=32, gamma=0.9) if cfg.action_head.controller == "mvprop" else None
    worlds, use_roles = rollout(cfg, planner)
    cov, l2 = draw(worlds, cfg, use_roles)
    print(f"saved {OUT}  final cov={cov:.0f}% l2={l2:.2f} roles={use_roles} occ={OCC}", flush=True)


if __name__ == "__main__":
    main()
