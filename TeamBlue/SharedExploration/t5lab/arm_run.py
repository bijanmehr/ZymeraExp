"""Phase-2 planner-arm runner: for ARCH in {mvprop, highway, gppn, msp} run the three experiment
pieces uniformly so the arms compare fairly:
  1. RL-ONLY init probe  — field value + move-score spread + init gradient norm at initialization
     (the mechanistic test for the vanishing-gradient pathology: is RL-only training even possible?).
  2. DISTILLATION        — supervised fit to the 4-connected wavefront teacher (masked MSE), the
     "does it learn to route at all" ceiling; saves planners/<arch>_distilled.eqx.
  3. GENERALIZATION      — optimal-move% + reach% across 8 unseen map types (the zero-shot metric).

    PYTHONPATH=.:../../../FiedlerValueEstimation ARCH=highway GRID=32 STEPS=4000 $PY -m t5lab.arm_run
"""
import os, json, time
import numpy as np
import jax, jax.numpy as jnp, equinox as eqx, optax

from t5lab.planner_arms import make_net
from t5lab.mvprop import propagate, _DELTAS
from t5lab.planner import mvprop_input
from t5lab.distill import gen_wall, sample_free_cell
from t5lab.planner_generalization import GENS

ARCH = os.environ.get("ARCH", "mvprop")
GRID = int(os.environ.get("GRID", 32))
KPROP = int(os.environ.get("KPROP", 32))
GAMMA = float(os.environ.get("GAMMA", 0.9))
STEPS = int(os.environ.get("STEPS", 4000))
NDATA = int(os.environ.get("NDATA", 4096))
BATCH = int(os.environ.get("BATCH", 64))
LR = float(os.environ.get("LR", 2e-3))
SEED = int(os.environ.get("SEED", 0))
OUT = os.environ.get("OUT", f"planners/{ARCH}_distilled.eqx")


def oracle_field(wall, goal):
    goal_oh = jnp.zeros((GRID, GRID)).at[goal[0], goal[1]].set(1.0)
    return propagate(goal_oh, (~wall).astype(jnp.float32), KPROP, GAMMA)


def make_sample(key):
    kw, kg = jax.random.split(key)
    wall = gen_wall(kw, GRID); goal = sample_free_cell(kg, wall)
    return mvprop_input(wall, goal), oracle_field(wall, goal), jnp.ones((GRID, GRID), jnp.float32)


def probe():
    net = make_net(ARCH, 2, KPROP, key=jax.random.PRNGKey(1), gamma=GAMMA)
    g = 16
    wall = jnp.zeros((g, g), bool)
    goal = jnp.array([g // 2, g // 2]); corner = jnp.array([0, 0])
    x = mvprop_input(wall, goal)
    V = net.field(x); sc = net.move_scores(x, corner)
    def loss_fn(nn):                                   # a stand-in reward-gradient at the far corner
        return -nn.field(x)[0, 0]
    _, grads = eqx.filter_value_and_grad(loss_fn)(net)
    gnorm = float(optax.global_norm(grads))
    print(f"[probe {ARCH}] field@corner={float(V[0,0]):.2e}  spread(move_scores)={float(sc.max()-sc.min()):.2e}  "
          f"init_gradnorm={gnorm:.2e}  -> {'RL-viable' if gnorm>1e-4 and float(sc.max()-sc.min())>1e-4 else 'DEAD (needs distill)'}", flush=True)


def distill():
    key = jax.random.PRNGKey(SEED)
    _, dk = jax.random.split(key)
    DX, DT, DM = jax.vmap(make_sample)(jax.random.split(dk, NDATA))
    net = make_net(ARCH, 2, KPROP, key=jax.random.PRNGKey(SEED + 1), gamma=GAMMA)
    opt = optax.adam(LR); ost = opt.init(eqx.filter(net, eqx.is_inexact_array))

    @eqx.filter_jit
    def step(net, ost, xs, ts, ms):
        def lf(nn):
            V = jax.vmap(nn.field)(xs)
            return jnp.sum(ms * (V - ts) ** 2) / jnp.sum(ms)
        l, gr = eqx.filter_value_and_grad(lf)(net)
        up, ost = opt.update(gr, ost)
        return eqx.apply_updates(net, up), ost, l

    t0 = time.time(); l = jnp.nan
    for it in range(STEPS):
        idx = jax.random.randint(jax.random.fold_in(key, it), (BATCH,), 0, NDATA)
        net, ost, l = step(net, ost, DX[idx], DT[idx], DM[idx])
        if it % 1000 == 0 or it == STEPS - 1:
            print(f"  [distill {ARCH} it{it:5d}] mse={float(l):.6f}", flush=True)
    print(f"[distill {ARCH}] {time.time()-t0:.0f}s final_mse={float(l):.6f}", flush=True)
    os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True)
    eqx.tree_serialise_leaves(OUT, net)
    return net, float(l)


def generalize(net):
    field_j = eqx.filter_jit(net.field)
    orc_j = eqx.filter_jit(oracle_field)
    D = np.asarray(_DELTAS); lim = [GRID - 1, GRID - 1]
    nbrs = lambda c: np.clip(c[None] + D, 0, lim)
    def opt_move(V, O, c):
        nb = nbrs(c); Vn = np.asarray(V)[nb[:, 0], nb[:, 1]]; On = np.asarray(O)[nb[:, 0], nb[:, 1]]
        return float(On[int(np.argmax(Vn))]) >= float(On.max()) - 1e-3
    def reaches(V, c, gl, bud):
        V = np.asarray(V); cur = c.copy(); seen = set()
        for _ in range(bud):
            if tuple(cur) == tuple(gl): return True
            nb = nbrs(cur); nxt = nb[int(np.argmax(V[nb[:, 0], nb[:, 1]]))]
            if tuple(nxt) == tuple(cur) or tuple(nxt) in seen: return False
            seen.add(tuple(cur)); cur = nxt
        return tuple(cur) == tuple(gl)
    rng = np.random.default_rng(0); res = {}
    for name, gen in GENS.items():
        no = nr = tot = 0
        for _ in range(200):
            wall = gen(rng); fc = np.argwhere(~wall)
            gl = fc[rng.integers(len(fc))]; s = fc[rng.integers(len(fc))]
            if tuple(s) == tuple(gl): continue
            V = field_j(mvprop_input(jnp.asarray(wall), jnp.asarray(gl)))
            O = orc_j(jnp.asarray(wall), jnp.asarray(gl))
            if float(np.asarray(O)[s[0], s[1]]) <= 1e-6: continue
            no += int(opt_move(V, O, s)); nr += int(reaches(V, s, gl, 4 * GRID)); tot += 1
        res[name] = {"optimal": round(100 * no / max(tot, 1), 1), "reach": round(100 * nr / max(tot, 1), 1), "n": tot}
        print(f"[gen {ARCH}] {name:16s} opt={res[name]['optimal']:5.1f}%  reach={res[name]['reach']:5.1f}%  n={tot}", flush=True)
    os.makedirs("planners", exist_ok=True)
    json.dump(res, open(f"planners/{ARCH}_gen.json", "w"), indent=2)
    return res


if __name__ == "__main__":
    print(f"=== ARM {ARCH}  grid={GRID} K={KPROP} steps={STEPS} ===", flush=True)
    probe()
    net, final_mse = distill()
    res = generalize(net)
    summ = {"arch": ARCH, "grid": GRID, "steps": STEPS, "final_mse": final_mse, "gen": res}
    json.dump(summ, open(f"planners/{ARCH}_summary.json", "w"), indent=2)
    print(f"=== ARM {ARCH} DONE ===", flush=True)
