"""Root-cause probe + distillation feasibility for the MVProp planner.

Part A — WHY RL-only MVProp never sharpens: at init the value field is near-flat / near-zero
where agents actually are, so the derived move is uninformative and the RL gradient through
K max-prop sweeps (max sparsity x w^K flood-decay) vanishes -> stuck.

Part B — the fix: distil MVProp's field toward the classical wavefront (controller.nav_distance_field,
free + already in the codebase). Dense per-cell target -> gradients everywhere. Show the field
then matches the teacher and the derived von-Neumann move matches the classical navfield move.

Run:
  V=/Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python
  PYTHONPATH=.:/Users/bijanmehr/Project.Zymera/zymera_lab $V -m t5lab.probe_mvprop
"""
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

from t5lab.mvprop import MVProp, _DELTAS
from ctde_v0.controller import nav_distance_field

H = W = 16
GAMMA_T = 0.9          # teacher field gamma^D (0.9 -> discriminative over short goal distances)
KPROP = 24


def make_input(blocked, goal):
    goal_oh = jnp.zeros((H, W)).at[goal[0], goal[1]].set(1.0)
    return jnp.stack([blocked.astype(jnp.float32), goal_oh])       # (2,H,W)


def teacher_field(blocked, goal):
    D = nav_distance_field(goal, blocked, "wavefront")             # (H,W), _FAR where blocked/unreached
    return jnp.where(D < 1e8, GAMMA_T ** D, 0.0)                   # (H,W) target value


def vn_move_from_field(V, cell):
    """argmax over the 5 von-Neumann neighbour values -> action index (STAY,N,E,S,W)."""
    nb = jnp.clip(cell[None] + _DELTAS, 0, jnp.array([H - 1, W - 1]))
    return int(jnp.argmax(V[nb[:, 0], nb[:, 1]]))


def vn_move_from_dist(D, cell):
    """argmin over the 5 von-Neumann neighbour distances -> the classical navfield move."""
    nb = jnp.clip(cell[None] + _DELTAS, 0, jnp.array([H - 1, W - 1]))
    return int(jnp.argmin(D[nb[:, 0], nb[:, 1]]))


def rand_wall(key):
    """~half open, ~half a few random axis-aligned wall segments (like local clutter)."""
    k1, k2, k3 = jax.random.split(key, 3)
    if jax.random.uniform(k1) < 0.5:
        return jnp.zeros((H, W), bool)
    b = np.zeros((H, W), bool)
    for kk in jax.random.split(k2, 3):
        r = int(jax.random.randint(kk, (), 1, H - 1))
        c0 = int(jax.random.randint(kk, (), 0, W // 2))
        b[r, c0:c0 + W // 3] = True
    return jnp.asarray(b)


def rand_goal_cell(key, blocked):
    free = np.argwhere(~np.asarray(blocked))
    idx = int(jax.random.randint(key, (), 0, len(free)))
    return jnp.asarray(free[idx], jnp.int32)


# ---------------------------------------------------------------- Part A: root cause
print("=" * 70)
print("PART A — MVProp at init (open 16x16, goal at centre, agent at corner)")
print("=" * 70)
m0 = MVProp(in_ch=2, K=KPROP, key=jax.random.PRNGKey(0))
blk = jnp.zeros((H, W), bool)
goal = jnp.array([8, 8], jnp.int32)
x = make_input(blk, goal)
V0 = m0.field(x)
agent = jnp.array([0, 0], jnp.int32)
nb = jnp.clip(agent[None] + _DELTAS, 0, jnp.array([H - 1, W - 1]))
logits0 = V0[nb[:, 0], nb[:, 1]]
D = nav_distance_field(goal, blk, "wavefront")
print(f"field V: min={float(V0.min()):.4g} max={float(V0.max()):.4g} std={float(V0.std()):.4g}")
print(f"  value AT goal cell (8,8)  = {float(V0[8,8]):.4g}   (should dominate)")
print(f"  value AT agent cell (0,0) = {float(V0[0,0]):.4g}")
print(f"move logits at agent (STAY,N,E,S,W) = {[round(float(v),4) for v in logits0]}")
print(f"  -> argmax move = {vn_move_from_field(V0, agent)}  (correct toward (8,8) = S or E = 3/2)")
print(f"  logit spread (max-min) = {float(logits0.max()-logits0.min()):.4g}  "
      f"(near 0 => flat softmax => entropy stuck, no learning signal)")
frac_reached = float((V0 > 0.1 * V0.max()).mean())
print(f"  fraction of cells with value > 10% of max = {frac_reached:.3f}  "
      f"(flood dies fast if small)")

# ---------------------------------------------------------------- Part B: distillation
print("\n" + "=" * 70)
print("PART B — distil MVProp toward the classical wavefront")
print("=" * 70)

# Precompute a FIXED dataset ONCE (teacher recompute per-step is the CPU bottleneck).
NDATA = 256
_k = jax.random.PRNGKey(2)
_xs, _ts, _masks = [], [], []
for _ in range(NDATA):
    _k, kb, kg = jax.random.split(_k, 3)
    b = rand_wall(kb)
    g = rand_goal_cell(kg, b)
    _xs.append(make_input(b, g)); _ts.append(teacher_field(b, g)); _masks.append(~b)
DX = jnp.stack(_xs); DT = jnp.stack(_ts); DM = jnp.stack(_masks).astype(jnp.float32)
print(f"dataset: {NDATA} samples precomputed  (x {DX.shape}, teacher {DT.shape})")

m = MVProp(in_ch=2, K=KPROP, key=jax.random.PRNGKey(1))
opt = optax.adam(2e-3)
opt_state = opt.init(eqx.filter(m, eqx.is_inexact_array))
BATCH = 32


@eqx.filter_jit
def step(m, opt_state, xs, ts, masks):
    def loss_fn(m):
        V = jax.vmap(m.field)(xs)                                   # (B,H,W)
        return jnp.sum(masks * (V - ts) ** 2) / jnp.sum(masks)
    loss, grads = eqx.filter_value_and_grad(loss_fn)(m)
    updates, opt_state = opt.update(grads, opt_state)
    return eqx.apply_updates(m, updates), opt_state, loss


key = jax.random.PRNGKey(7)
for it in range(800):
    key, sk = jax.random.split(key)
    idx = jax.random.randint(sk, (BATCH,), 0, NDATA)
    m, opt_state, loss = step(m, opt_state, DX[idx], DT[idx], DM[idx])
    if it % 100 == 0 or it == 799:
        print(f"  [distill it {it:3d}] mse={float(loss):.5f}", flush=True)

# ---- held-out move-match: distilled MVProp move vs classical navfield move ----
print("\nHeld-out move-match (distilled MVProp argmax V  vs  classical navfield argmin D):")
_field_j = eqx.filter_jit(m.field)
_nav_j = jax.jit(lambda g, b: nav_distance_field(g, b, "wavefront"))
key = jax.random.PRNGKey(999)
for tag, force_open in [("open", True), ("walled", False)]:
    match = tot = 0
    for _ in range(120):
        key, kb, kg, ka = jax.random.split(key, 4)
        b = jnp.zeros((H, W), bool) if force_open else rand_wall(kb)
        g = rand_goal_cell(kg, b)
        Vd = _field_j(make_input(b, g))
        Dd = _nav_j(g, b)
        cell = rand_goal_cell(ka, b)                                # random free agent cell
        if int(jnp.all(cell == g)):
            continue
        match += int(vn_move_from_field(Vd, cell) == vn_move_from_dist(Dd, cell))
        tot += 1
    print(f"  {tag:7s}: move-match vs classical navfield = {100*match/tot:.1f}%  ({match}/{tot})")
print("\n=== PROBE DONE ===")
