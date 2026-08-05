"""Open-terrain train+eval driver. Env vars: GRID(32), ITERS(300), NAG(10), SEED(0),
CTRL(mvprop|greedy). Trains, then evals COVERAGE + OVERLAP + CONNECTIVITY (lambda2 / reach /
connected-fraction) for greedy and sampled goal-selection. Writes a results JSON for the report.
CTRL=greedy drops MVProp (deterministic step-toward-goal) — the no-planner control.
"""
import os
import json
import jax
import numpy as np
import jax.numpy as jnp
from collections import Counter
from t5lab.run_phase1a import base_config, build_actor, _env_with_wall
from ctde_v0 import env_utils as eu
from ctde_v0.ppo import make_stencil, _navfield_blocked
from ctde_v0.controller import goal_targets
from t5lab.rollout import resolve_conflicts, greedy_toward
from t5lab.train import train

G = int(os.environ.get("GRID", 32))
IT = int(os.environ.get("ITERS", 300))
NA = int(os.environ.get("NAG", 10))
SEED = int(os.environ.get("SEED", 0))
CTRL = os.environ.get("CTRL", "mvprop")
key = jax.random.PRNGKey(SEED)
cfg = base_config(terrain="open", grid=G, n_agents=NA, iters=IT)
env = eu.build_env(cfg)
actor = build_actor(cfg, env, key)
print(f"[{CTRL}] training open {G}x{G}/{NA} for {IT} iters (seed {SEED})...", flush=True)
actor, hist = train(env, actor, cfg, key, controller=CTRL)
print("TRAINED final:", {k: round(v, 2) for k, v in hist[-1].items()}, flush=True)
stencil = make_stencil(cfg)
CR = cfg.world.comm_r


def conn_stats(pos):
    P = np.asarray(pos)
    N = len(P)
    d = np.max(np.abs(P[:, None, :] - P[None, :, :]), -1)
    A = (d <= CR).astype(int)
    np.fill_diagonal(A, 1)
    R = A.copy()
    for _ in range(6):
        R = ((R @ R) > 0).astype(int)
    return ((R.sum(1) - 1) / (N - 1)).mean(), len({tuple(r) for r in R})


def ev(sample, steps=100):
    e = _env_with_wall(cfg, jnp.zeros((G, G), bool))
    obs, st = e.reset(key)
    k = key
    ov, l2, rf, conn = [], [], [], []
    for _ in range(steps):
        k, ak, mk, sk = jax.random.split(k, 4)
        pos = st.body.position
        n = pos.shape[0]
        P = [tuple(x) for x in np.asarray(pos)]
        c = Counter(P)
        ov.append(sum(v for v in c.values() if v > 1))
        l2.append(float(eu.true_lambda2(pos, cfg)))
        r, cc = conn_stats(pos)
        rf.append(r)
        conn.append(cc == 1)
        adj = eu.kb_adjacency(pos, cfg)
        blk = _navfield_blocked(st, cfg)
        z = actor.belief(obs, adj, inference=True)
        gl, _, _ = actor.heads(z)
        gi = jax.random.categorical(ak, gl, -1) if sample else jnp.argmax(gl, -1)
        h, w = st.wall.shape
        goal = goal_targets(pos, stencil, h, w)[jnp.arange(n), gi]
        if CTRL == "greedy":
            mv = greedy_toward(pos, goal)
        else:
            ml = actor.move_logits(pos, goal, blk)
            mv = jax.random.categorical(mk, ml, -1) if sample else jnp.argmax(ml, -1)
        mv = resolve_conflicts(mv, e.dynamics.targets(st))
        obs, st, *_ = e.step(st, mv, sk)
    cov = np.asarray(st.covered).astype(bool)
    return dict(cov=round(100 * cov.sum() / (G * G), 1), overlap=round(float(np.mean(ov)), 2),
                l2=round(float(np.mean(l2)), 2), reach=round(float(np.mean(rf)), 2),
                conn_frac=round(float(np.mean(conn)), 2))


res = {"ctrl": CTRL, "grid": G, "seed": SEED, "final_ret": round(hist[-1]["ret"], 1)}
for name, samp in [("greedy", False), ("sampled", True)]:
    res[name] = ev(samp)
    print(f"OPEN {name:8}", res[name], flush=True)
outp = f"t5lab/results_{CTRL}_{G}_s{SEED}.json"
json.dump(res, open(outp, "w"), indent=1)
print("wrote", outp, flush=True)
print("=== OPEN RUN DONE ===", flush=True)
