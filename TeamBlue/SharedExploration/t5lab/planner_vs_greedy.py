"""The planner's reason to exist: mvprop routes AROUND walls; greedy gets trapped by them.

Single-agent navigation on rooms / corridors / maze / random-obstacle maps (the frozen distilled
planner, no policy, no GPU). For each (map, start, goal) we descend two controllers to the goal:

  * greedy   — step to the free 4-neighbour that most reduces Chebyshev distance to the goal;
               STAY if none strictly improves (exactly ctde_v0's greedy_move, single-agent). This
               is the reactive baseline: it CANNOT climb "away" to get around a wall, so a wall on
               the straight line traps it.
  * mvprop   — step to the free 4-neighbour of maximum learned value V; STAY at a local max. The
               value field floods around walls, so descending it follows a true geodesic.

Reports reach-rate + path/optimal length for each, per map type, and renders a composite PNG of
cases where greedy is trapped but mvprop routes through.

Run: PYTHONPATH=. PLANNER=/tmp/mvprop_distilled.eqx JAX_PLATFORMS=cpu $PY -m t5lab.planner_vs_greedy
"""
import os
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from PIL import Image, ImageDraw

from t5lab.mvprop import propagate
from t5lab.planner import load_planner, mvprop_input

GRID = int(os.environ.get("GRID", 28))
KPROP = int(os.environ.get("KPROP", 40))
GAMMA = float(os.environ.get("GAMMA", 0.9))
PLANNER = os.environ.get("PLANNER", "/tmp/mvprop_distilled.eqx")
N = int(os.environ.get("NSAMP", 250))
OUT = os.environ.get("OUT", "report/planner_vs_greedy.png")
_D4 = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])           # N,E,S,W (4-connected)


# ------------------------------- maps (rooms / corridors / maze / random) -------------------
def rng_free(rng, wall):
    f = np.argwhere(~wall); return f[rng.integers(len(f))]


def m_rooms(rng, g):
    w = np.zeros((g, g), bool)
    for c in range(g // 4, g, g // 4):
        w[:, c] = True; d = rng.integers(0, g - 3); w[d:d + 3, c] = False
    for r in range(g // 4, g, g // 4):
        w[r, :] = True; d = rng.integers(0, g - 3); w[r, d:d + 3] = False
    return w


def m_corridors(rng, g):
    w = np.zeros((g, g), bool)
    for c in range(3, g - 1, 4):
        w[:, c] = True; d = rng.integers(0, g - 4); w[d:d + 4, c] = False
    return w


def m_maze(rng, g):
    w = np.zeros((g, g), bool)
    for i, r in enumerate(range(2, g - 1, 3)):
        w[r, :] = True
        if i % 2 == 0:
            w[r, g - 2:] = False
        else:
            w[r, :2] = False
    return w


def m_random(rng, g):
    w = np.zeros((g, g), bool)
    idx = rng.choice(g * g, size=int(0.14 * g * g), replace=False); w.flat[idx] = True
    return w


MAPS = {"rooms": m_rooms, "corridors": m_corridors, "maze": m_maze, "random_obstacle": m_random}


# ------------------------------- oracle + descents ------------------------------------------
def oracle_field(wall, goal):
    oh = jnp.zeros((GRID, GRID)).at[goal[0], goal[1]].set(1.0)
    return np.asarray(propagate(oh, (~jnp.asarray(wall)).astype(jnp.float32), KPROP, GAMMA))


def free_nbrs(cell, wall):
    nb = cell + _D4
    ok = (nb[:, 0] >= 0) & (nb[:, 0] < GRID) & (nb[:, 1] >= 0) & (nb[:, 1] < GRID)
    nb = nb[ok]
    return nb[~wall[nb[:, 0], nb[:, 1]]]


def cheby(a, b):
    return int(np.max(np.abs(np.asarray(a) - np.asarray(b))))


def descend(start, goal, wall, score, budget, greedy):
    """score(cell)->higher-is-better for mvprop; for greedy we use -Chebyshev(cell,goal).
    Returns (path, reached)."""
    cur = np.asarray(start); path = [tuple(cur)]; seen = {tuple(cur)}
    for _ in range(budget):
        if tuple(cur) == tuple(goal):
            return path, True
        nb = free_nbrs(cur, wall)
        if len(nb) == 0:
            return path, False
        if greedy:
            here = -cheby(cur, goal); vals = np.array([-cheby(c, goal) for c in nb])
        else:
            here = score[cur[0], cur[1]]; vals = score[nb[:, 0], nb[:, 1]]
        j = int(np.argmax(vals))
        if vals[j] <= here + 1e-9:                 # no strict improvement -> trapped / local max
            return path, False
        cur = nb[j]
        if tuple(cur) in seen:                     # cycle -> trapped
            return path, False
        seen.add(tuple(cur)); path.append(tuple(cur))
    return path, tuple(cur) == tuple(goal)


# ------------------------------- render ------------------------------------------------------
def tile(wall, start, goal, mv_path, gr_path, sc=11):
    g = wall.shape[0]; img = Image.new("RGB", (g * sc, g * sc), (238, 236, 230))
    dr = ImageDraw.Draw(img)
    for r in range(g):
        for c in range(g):
            if wall[r, c]:
                dr.rectangle([c * sc, r * sc, c * sc + sc - 1, r * sc + sc - 1], fill=(52, 56, 64))

    def line(path, col, w):
        for (r0, c0), (r1, c1) in zip(path, path[1:]):
            dr.line([c0 * sc + sc // 2, r0 * sc + sc // 2, c1 * sc + sc // 2, r1 * sc + sc // 2],
                    fill=col, width=w)
    line(gr_path, (192, 64, 46), 5)                # greedy = red
    line(mv_path, (47, 140, 100), 3)               # mvprop = teal-green
    gr_end = gr_path[-1]                            # mark greedy trap
    dr.ellipse([gr_end[1] * sc + 1, gr_end[0] * sc + 1, gr_end[1] * sc + sc - 2, gr_end[0] * sc + sc - 2],
               outline=(192, 64, 46), width=3)
    for cell, col in [(start, (40, 90, 200)), (goal, (30, 160, 80))]:
        dr.rectangle([cell[1] * sc, cell[0] * sc, cell[1] * sc + sc - 1, cell[0] * sc + sc - 1], fill=col)
    return img


def main():
    planner = load_planner(PLANNER, in_ch=2, K=KPROP, gamma=GAMMA)
    field_j = eqx.filter_jit(planner.net.field)
    print(f"=== planner (mvprop) vs greedy — routing around walls @ {GRID}x{GRID} ===", flush=True)
    print(f"{'map':16s} {'mvprop reach':>13s} {'greedy reach':>13s} {'mvprop len/opt':>15s}", flush=True)
    rng = np.random.default_rng(0); budget = 4 * GRID
    gallery = []; PER_TYPE = 2
    for name, gen in MAPS.items():
        mv_ok = gr_ok = tot = 0; mv_ratio = []; taken = 0
        for _ in range(N):
            wall = gen(rng, GRID); goal = rng_free(rng, wall); start = rng_free(rng, wall)
            if tuple(start) == tuple(goal):
                continue
            O = oracle_field(wall, goal)
            if O[start[0], start[1]] <= 1e-6:
                continue                            # goal not reachable from start (skip)
            V = np.asarray(field_j(mvprop_input(jnp.asarray(wall), jnp.asarray(goal))))
            mv_path, mv_r = descend(start, goal, wall, V, budget, greedy=False)
            gr_path, gr_r = descend(start, goal, wall, None, budget, greedy=True)
            tot += 1; mv_ok += mv_r; gr_ok += gr_r
            if mv_r:
                opt = np.log(max(O[start[0], start[1]], 1e-9)) / np.log(GAMMA)
                mv_ratio.append(len(mv_path) / max(opt, 1))
            if mv_r and not gr_r and taken < PER_TYPE:    # the money shot: greedy trapped, mvprop through
                gallery.append((name, wall.copy(), tuple(start), tuple(goal), mv_path, gr_path)); taken += 1
        print(f"{name:16s} {100*mv_ok/tot:11.1f}% {100*gr_ok/tot:11.1f}% {np.mean(mv_ratio):13.2f}x   n={tot}",
              flush=True)

    # composite PNG of trapped-greedy / routing-mvprop cases
    if gallery:
        tiles = [tile(w, s, g, mv, gr) for (_, w, s, g, mv, gr) in gallery]
        tw, th = tiles[0].size; cols = 3; rows = (len(tiles) + cols - 1) // cols; pad = 14; cap = 22
        comp = Image.new("RGB", (cols * tw + (cols + 1) * pad, rows * (th + cap) + (rows + 1) * pad),
                         (247, 245, 240))
        dr = ImageDraw.Draw(comp)
        for i, (t, (nm, *_)) in enumerate(zip(tiles, gallery)):
            r, c = divmod(i, cols); x = pad + c * (tw + pad); y = pad + r * (th + cap)
            comp.paste(t, (x, y)); dr.text((x + 2, y + th + 4), f"{nm}: greedy trapped (o), mvprop routes",
                                           fill=(70, 74, 82))
        os.makedirs(os.path.dirname(OUT) or ".", exist_ok=True); comp.save(OUT)
        print(f"saved composite -> {OUT}  ({len(gallery)} cases)", flush=True)
    print("=== DONE ===", flush=True)


if __name__ == "__main__":
    main()
