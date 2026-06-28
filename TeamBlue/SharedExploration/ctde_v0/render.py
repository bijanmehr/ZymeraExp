"""Render a trained ctde_v0 policy rollout to a GIF — for visual audit.

Loads a run's ``config.json`` + ``model.eqx``, runs ONE rollout under the trained actor
(the SAME policy step as ``ppo._single_rollout``, un-jitted + keeping the World each step),
colors agents by their chosen SKILL (selector arms) or ROLE (role arm), and writes an
animated GIF (coverage heat + comm graph + agents) via ``zymera.viz``.

    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m ctde_v0.render \
        --run-dir runs/skills/seed0/ladder/role/32x32x10 --out gifs/role_32.gif
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import jax
import jax.numpy as jnp

if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ctde_v0 import env_utils, ppo
    from ctde_v0.config import from_dict
else:
    from . import env_utils, ppo
    from .config import from_dict

from zymera.viz import render_gif


def render_run(run_dir: str, out: str, *, steps: int = 100, seed: int = 0,
               fps: int = 8) -> str:
    """Render the trained policy in ``run_dir`` to a GIF at ``out``."""
    cfg = from_dict(json.load(open(os.path.join(run_dir, "config.json"))))
    if steps:
        import dataclasses
        cfg = dataclasses.replace(cfg, world=dataclasses.replace(cfg.world, horizon=steps))
    env = env_utils.build_env(cfg)
    ckpt = os.path.join(run_dir, "model.eqx")
    state = ppo.init_state_from_checkpoint(env, cfg, ckpt, jax.random.PRNGKey(seed))
    actor = state.actor
    stencil = ppo.make_stencil(cfg)
    edge_msg = cfg.backbone.message_content != "learned"
    use_selector = cfg.selector == "on"
    use_roles = cfg.role_picker == "expl_relay"

    k = jax.random.PRNGKey(seed + 7)
    rk0, kk = jax.random.split(k)
    obs, st = env.reset(rk0)
    h = actor.init_hidden(st.n_agents)
    worlds = [st]
    for _t in range(cfg.world.horizon):
        kk, ak, rk, sk = jax.random.split(kk, 4)
        adj = env_utils.kb_adjacency(st.body.position, cfg)
        dist = env_utils.kb_distance(st.body.position, cfg) if edge_msg else None
        tag = None                                            # per-agent colour code
        if use_selector:
            ck = jax.random.fold_in(rk, 0x5E1)
            skill_logits, offset_logits, _feat, h = actor.skill_forward(
                obs, adj, st.body.position, dist=dist, h=h, inference=True)
            skill = jax.random.categorical(ck, skill_logits, axis=-1)      # (N,) skill m
            goal_logits = jnp.take_along_axis(
                offset_logits, skill[None, :, None], axis=0)[0]            # (N,K)
            role_idx = None
            tag = skill.astype(jnp.int32)                     # colour by skill (0/1/2)
        else:
            goal_logits, role_logits, _v, _l2, _z, h = actor(
                obs, adj, dist=dist, h=h, inference=True)
            if use_roles:
                role_idx = jax.random.categorical(rk, role_logits, axis=-1)
                tag = role_idx.astype(jnp.int32)              # colour by role (expl/relay)
            else:
                role_idx = None
        gmask = ppo._goal_mask(env, st, cfg, stencil)
        masked = jnp.where(gmask, goal_logits, ppo._NEG)
        goal = jax.random.categorical(ak, masked, axis=-1)
        move, _ = ppo._goal_to_move(env, st, goal, stencil, role_idx, cfg)
        obs, st, _r, _d, _i = env.step(st, move, sk)
        if tag is not None:                                   # recolour agents by skill/role
            st = st.replace(group=tag)
        worlds.append(st)

    traj = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *worlds)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    render_gif(traj, out, comm_radius=cfg.world.comm_r, fps=fps)
    label = (f"selector flock={cfg.flock} congestion={cfg.congestion}" if use_selector
             else ("role(expl/relay)" if use_roles else "base"))
    print(f"saved {out}  [{cfg.scale}  {label}]", flush=True)
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="render a trained ctde_v0 policy to a GIF")
    p.add_argument("--run-dir", required=True, help="dir with config.json + model.eqx")
    p.add_argument("--out", required=True, help="output .gif path")
    p.add_argument("--steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--fps", type=int, default=8)
    a = p.parse_args(argv)
    render_run(a.run_dir, a.out, steps=a.steps, seed=a.seed, fps=a.fps)


if __name__ == "__main__":
    main()
