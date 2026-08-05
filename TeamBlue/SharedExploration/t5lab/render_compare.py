"""Train a short open policy and render a rollout GIF. Env: GRID(16) NAG(4) ITERS(2500) CTRL(greedy|mvprop)."""
import os
import jax
import jax.numpy as jnp
from t5lab.run_phase1a import base_config, build_actor, _env_with_wall
from ctde_v0 import env_utils as eu
from ctde_v0.ppo import make_stencil, _navfield_blocked
from ctde_v0.controller import goal_targets
from t5lab.rollout import resolve_conflicts, greedy_toward
from t5lab.train import train
from zymera.viz import render_gif

G = int(os.environ.get("GRID", 16))
NA = int(os.environ.get("NAG", 4))
IT = int(os.environ.get("ITERS", 2500))
CTRL = os.environ.get("CTRL", "greedy")
key = jax.random.PRNGKey(0)
cfg = base_config(terrain="open", grid=G, n_agents=NA, iters=IT)
env = eu.build_env(cfg)
actor = build_actor(cfg, env, key)
print("[%s] training open %dx%d/%d for %d iters..." % (CTRL, G, G, NA, IT), flush=True)
actor, _ = train(env, actor, cfg, key, controller=CTRL)
stencil = make_stencil(cfg)
os.makedirs("t5lab/gifs", exist_ok=True)
e = _env_with_wall(cfg, jnp.zeros((G, G), bool))
obs, st = e.reset(key)
worlds = [st]
k = key
for _ in range(100):
    k, ak, mk, sk = jax.random.split(k, 4)
    pos = st.body.position
    n = pos.shape[0]
    adj = eu.kb_adjacency(pos, cfg)
    blk = _navfield_blocked(st, cfg)
    z = actor.belief(obs, adj, inference=True)
    gl, _, _ = actor.heads(z)
    gi = jax.random.categorical(ak, gl, -1)
    h, w = st.wall.shape
    goal = goal_targets(pos, stencil, h, w)[jnp.arange(n), gi]
    if CTRL == "greedy":
        mv = greedy_toward(pos, goal)
    else:
        mv = jax.random.categorical(mk, actor.move_logits(pos, goal, blk), -1)
    mv = resolve_conflicts(mv, e.dynamics.targets(st))
    obs, st, *_ = e.step(st, mv, sk)
    worlds.append(st)
traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
out = "t5lab/gifs/%s_%d.gif" % (CTRL, G)
render_gif(traj, out, comm_radius=cfg.world.comm_r, fps=10)
print("saved", out, flush=True)
print("=== RENDER DONE ===", flush=True)
