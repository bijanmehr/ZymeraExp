"""Hard-collision guarantee: with --collision-mask on, NO two agents ever share a cell.

Stress setup: many agents on a small grid with cluster spawn, so agents are packed and constantly
try to converge on the same cells — the case the old occupied_cell_mask alone did NOT cover
(simultaneous claims on an empty cell). resolve_target_conflicts (index-priority) must drive the
per-step same-cell overlap count to exactly 0. collision_mask=off is the control (collisions DO
occur), proving the test can detect them.

    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation <python> -m ctde_v0.tests.test_collision_free
"""
from __future__ import annotations

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from ctde_v0 import env_utils, ppo, train_ctde

_ARGV = lambda cm, g, n, steps: [
    "--grid", str(g), "--n-agents", str(n), "--comm-r", "5", "--horizon", str(steps),
    "--explorer-tool", "frontier_attn", "--collision-mask", cm, "--mechanism", "lagrangian",
    "--conn-signal", "global_lambda2", "--constraint-threshold", "0.5", "--role-picker", "expl_relay",
    "--w-coverage", "3", "--sense-walls", "--sense-free", "--boundary", "--cover-r", "0"]


def _rollout_overlaps(collision_mask, grid=10, n=10, steps=60):
    """Roll a fresh (random-init) policy through the real _goal_to_move + env.step path; return
    (spawn overlaps, max simultaneous same-cell overlaps over the episode, #steps with any)."""
    cfg, *_ = train_ctde._parse_args(_ARGV(collision_mask, grid, n, steps))
    env = env_utils.build_env(cfg)
    tstate = ppo.init_state(env, cfg, jax.random.PRNGKey(0))
    actor, stencil = tstate.actor, ppo.make_stencil(cfg)
    use_roles = cfg.role_picker == "expl_relay"

    rk0, kk = jax.random.split(jax.random.PRNGKey(1))
    obs, st = env.reset(rk0)
    h = actor.init_hidden(st.n_agents)

    def dup(pos):
        p = [tuple(int(x) for x in c) for c in np.asarray(pos)]
        return len(p) - len(set(p))

    spawn_dup = dup(st.body.position)
    max_dup, bad_steps = 0, 0
    for _ in range(steps):
        kk, ak, rk, sk = jax.random.split(kk, 4)
        adj = env_utils.kb_adjacency(st.body.position, cfg)
        gl, rl, _v, _l2, _z, h = actor(obs, adj, dist=None, h=h, inference=True)
        role_idx = jax.random.categorical(rk, rl, axis=-1) if use_roles else None
        gm = ppo._goal_mask(env, st, cfg, stencil)
        goal = jax.random.categorical(ak, jnp.where(gm, gl, ppo._NEG), axis=-1)
        move, _ = ppo._goal_to_move(env, st, goal, stencil, role_idx, cfg)
        obs, st, _r, _d, _i = env.step(st, move, sk)
        d = dup(st.body.position)
        max_dup = max(max_dup, d)
        bad_steps += int(d > 0)
    return spawn_dup, max_dup, bad_steps


def test_collision_free_when_masked():
    spawn_dup, max_dup, bad = _rollout_overlaps("on", grid=10, n=10, steps=60)
    assert spawn_dup == 0, f"spawn placed {spawn_dup} agents on shared cells (precondition broken)"
    assert max_dup == 0, f"collision_mask=on still produced {max_dup} same-cell overlaps ({bad} steps)"


def test_control_detects_collisions_when_off():
    # sanity: the detector CAN see collisions — the unmasked policy should produce some.
    _sd, max_dup, _bad = _rollout_overlaps("off", grid=10, n=10, steps=60)
    assert max_dup >= 1, "control (mask off) produced no collisions — the stress setup is too easy"


if __name__ == "__main__":
    for cm in ("off", "on"):
        sd, md, bad = _rollout_overlaps(cm, grid=10, n=10, steps=60)
        flag = "  <-- MUST be 0" if cm == "on" else "  (control)"
        print(f"collision_mask={cm:3s}: spawn_overlap={sd}  max_same_cell_overlap={md}  "
              f"steps_with_overlap={bad}/60{flag}")
    sd, md, bad = _rollout_overlaps("on", 10, 10, 60)
    print("\nHARD-COLLISION:", "PASS ✅  (zero overlaps with mask on)" if md == 0
          else f"FAIL ❌  ({md} overlaps leaked)")
    raise SystemExit(0 if md == 0 else 1)
