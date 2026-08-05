"""Distil the MVProp planner toward the classical wavefront (controller.nav_distance_field).

WHY (probe_mvprop.py): RL-only MVProp never sharpens — at init the flood dies within ~5 cells,
the field is flat where agents are, and the gradient through K max-prop sweeps vanishes. The fix
is a dense supervised target: for random (walls, goal) samples, the classical BFS wavefront gives
the exact distance-to-goal field; MVProp learns to reproduce ``gamma^distance`` everywhere, which
makes its argmax-value move equal the classical shortest-path move. The teacher is used ONLY at
training time — at inference the planner is pure MVProp (no classical code runs), so the system
stays fully learned.

Output: a frozen MVPropPlanner checkpoint plugged into ctde_v0 as ``action_head.controller='mvprop'``.

Run (GPU):
  PY=$HOME/ZymeraLab/.venv/bin/python
  PYTHONPATH=. GRID=32 STEPS=4000 OUT=mvprop_distilled.eqx $PY -m t5lab.distill
"""
import os
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from t5lab.mvprop import MVProp, propagate, _DELTAS
from t5lab.planner import MVPropPlanner, mvprop_input

GRID = int(os.environ.get("GRID", 32))
KPROP = int(os.environ.get("KPROP", 32))
GAMMA = float(os.environ.get("GAMMA", 0.9))
NDATA = int(os.environ.get("NDATA", 4096))
STEPS = int(os.environ.get("STEPS", 4000))
BATCH = int(os.environ.get("BATCH", 64))
LR = float(os.environ.get("LR", 2e-3))
OUT = os.environ.get("OUT", "mvprop_distilled.eqx")
SEED = int(os.environ.get("SEED", 0))


def gen_wall(key, grid, max_rects=5, p_open=0.30):
    """(H,W) bool — jittable random terrain: with prob ``p_open`` fully open, else up to
    ``max_rects`` random filled axis-aligned rectangles (varied clutter for general routing)."""
    ko, kr = jax.random.split(key)
    rows = jnp.arange(grid)[:, None]
    cols = jnp.arange(grid)[None, :]

    def add_rect(carry, rk):
        wall, = carry
        k1, k2, k3, k4, k5 = jax.random.split(rk, 5)
        r0 = jax.random.randint(k1, (), 0, grid)
        c0 = jax.random.randint(k2, (), 0, grid)
        rh = jax.random.randint(k3, (), 1, grid // 3 + 2)
        rw = jax.random.randint(k4, (), 1, grid // 3 + 2)
        place = jax.random.uniform(k5) < 0.8                        # some rect slots empty
        rect = (rows >= r0) & (rows < r0 + rh) & (cols >= c0) & (cols < c0 + rw)
        return (wall | (rect & place),), None

    wall0 = jnp.zeros((grid, grid), bool)
    (wall,), _ = jax.lax.scan(add_rect, (wall0,), jax.random.split(kr, max_rects))
    return jnp.where(jax.random.uniform(ko) < p_open, jnp.zeros_like(wall), wall)


def sample_free_cell(key, wall):
    """(2,) int32 — a uniformly random FREE cell (jittable categorical over free cells)."""
    free = (~wall).astype(jnp.float32).ravel()
    logits = jnp.where(free > 0, 0.0, -1e9)
    idx = jax.random.categorical(key, logits)
    return jnp.stack([idx // wall.shape[1], idx % wall.shape[1]]).astype(jnp.int32)


def oracle_field(wall, goal, grid):
    """(H,W) target value field = the 4-connected geodesic gamma^dist to ``goal``, computed by
    ``propagate`` with ORACLE passability ``w = 1 - wall``. This is the EXACT field a 4-move
    agent should descend (von-Neumann geometry, same as the env action space and as MVProp's
    own flood) — so the student can reproduce it exactly by learning ``w -> 1-wall``. Walls and
    unreached cells sit at ~0."""
    goal_oh = jnp.zeros((grid, grid)).at[goal[0], goal[1]].set(1.0)
    w = (~wall).astype(jnp.float32)
    return propagate(goal_oh, w, KPROP, GAMMA)                       # (H,W) gamma^(4-conn dist)


def make_sample(key, grid):
    kw, kg = jax.random.split(key)
    wall = gen_wall(kw, grid)
    goal = sample_free_cell(kg, wall)
    x = mvprop_input(wall, goal)                                     # (2,H,W)
    target = oracle_field(wall, goal, grid)                          # (H,W) 4-conn gamma^dist
    mask = jnp.ones((grid, grid), jnp.float32)                       # supervise all cells (walls->0)
    return x, target, mask


def main():
    key = jax.random.PRNGKey(SEED)
    print(f"[distill] grid={GRID} K={KPROP} gamma={GAMMA} ndata={NDATA} steps={STEPS} "
          f"batch={BATCH} lr={LR}", flush=True)

    # ---- precompute dataset (vectorised on device) ----
    key, dk = jax.random.split(key)
    DX, DT, DM = jax.vmap(lambda k: make_sample(k, GRID))(jax.random.split(dk, NDATA))
    print(f"[distill] dataset ready: x{tuple(DX.shape)} target{tuple(DT.shape)}", flush=True)

    net = MVProp(in_ch=2, K=KPROP, key=jax.random.PRNGKey(SEED + 1), gamma=GAMMA)
    opt = optax.adam(LR)
    opt_state = opt.init(eqx.filter(net, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(net, opt_state, xs, ts, ms):
        def loss_fn(net):
            V = jax.vmap(net.field)(xs)                              # (B,H,W)
            return jnp.sum(ms * (V - ts) ** 2) / jnp.sum(ms)
        loss, grads = eqx.filter_value_and_grad(loss_fn)(net)
        updates, opt_state = opt.update(grads, opt_state)
        return eqx.apply_updates(net, updates), opt_state, loss

    for it in range(STEPS):
        key, sk = jax.random.split(key)
        idx = jax.random.randint(sk, (BATCH,), 0, NDATA)
        net, opt_state, loss = step(net, opt_state, DX[idx], DT[idx], DM[idx])
        if it % 500 == 0 or it == STEPS - 1:
            print(f"  [it {it:5d}] mse={float(loss):.6f}", flush=True)

    # ---- held-out move OPTIMALITY vs the 4-conn oracle geodesic ----
    # The scientific metric is not "identical to a tie-break" but "does the chosen move achieve
    # the minimum distance-to-goal among the 5 neighbours" (i.e. is it an optimal 4-move step).
    planner = MVPropPlanner(net)
    field_j = eqx.filter_jit(net.field)
    oracle_j = eqx.filter_jit(lambda wall, goal: oracle_field(wall, goal, GRID))
    lim = jnp.array([GRID - 1, GRID - 1])
    TOL = 1e-3

    def eval_cell(V, O, cell):
        nb = jnp.clip(cell[None] + _DELTAS, 0, lim)                  # (5,2)
        Vnb = V[nb[:, 0], nb[:, 1]]; Onb = O[nb[:, 0], nb[:, 1]]     # (5,)
        a = int(jnp.argmax(Vnb))                                     # MVProp's move
        optimal = float(Onb[a]) >= float(jnp.max(Onb)) - TOL        # achieves best oracle value
        progress = float(Onb[a]) > float(O[cell[0], cell[1]]) + TOL  # strictly closer than STAY
        return optimal, progress

    key = jax.random.PRNGKey(SEED + 999)
    for tag, force_open in [("open", True), ("clutter", False)]:
        nopt = nprog = tot = 0
        for _ in range(400):
            key, kw, kg, ka = jax.random.split(key, 4)
            wall = jnp.zeros((GRID, GRID), bool) if force_open else gen_wall(kw, GRID, p_open=0.0)
            goal = sample_free_cell(kg, wall)
            cell = sample_free_cell(ka, wall)
            if int(jnp.all(cell == goal)):
                continue
            V = field_j(mvprop_input(wall, goal)); O = oracle_j(wall, goal)
            opt, prog = eval_cell(V, O, cell)
            nopt += int(opt); nprog += int(prog); tot += 1
        print(f"[distill] {tag:7s}: optimal-move={100*nopt/tot:.1f}%  makes-progress={100*nprog/tot:.1f}%  "
              f"(n={tot})", flush=True)

    eqx.tree_serialise_leaves(OUT, planner)
    print(f"[distill] saved planner -> {OUT}", flush=True)
    print("=== DISTILL DONE ===", flush=True)


if __name__ == "__main__":
    main()
