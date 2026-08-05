"""Distillation-bias check: does the distilled MVProp planner generalize to map types it never
saw in training? Training used random rectangles + open (gen_wall). Here we test STRUCTURALLY
DIFFERENT maps — rooms, serpentine mazes, pillar lattices, DIAGONAL barriers (the hardest for a
conv trained on axis-aligned rectangles), dense clutter, and the held-out SAR set if present.

Two metrics per map type, vs the 4-connected geodesic oracle:
  * optimal-move : fraction of (map, goal, cell) where the planner's argmax-V move achieves the
                   minimum 4-connected distance-to-goal among the 5 neighbours (local correctness).
  * reach        : fraction of (map, goal, start) where FOLLOWING the field from start actually
                   arrives at the goal within a step budget (GLOBAL correctness — catches a field
                   whose only flaw is a local maximum that traps the agent). This is the strong test.

Run: PYTHONPATH=. GRID=32 PLANNER=mvprop_distilled.eqx $PY -m t5lab.planner_generalization
"""
import os
import json
import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx

from t5lab.mvprop import propagate, _DELTAS
from t5lab.planner import load_planner, mvprop_input

GRID = int(os.environ.get("GRID", 32))
KPROP = int(os.environ.get("KPROP", 32))
GAMMA = float(os.environ.get("GAMMA", 0.9))
PLANNER = os.environ.get("PLANNER", "mvprop_distilled.eqx")
N = int(os.environ.get("NSAMP", 300))
SARP = os.environ.get("SAR", "maps_sar.json")


# ---------------- map generators (host/numpy), DIFFERENT from training's gen_wall ------------
def m_open(rng):
    return np.zeros((GRID, GRID), bool)


def m_rooms(rng):
    w = np.zeros((GRID, GRID), bool)
    for c in range(GRID // 4, GRID, GRID // 4):          # vertical partitions with a door gap
        w[:, c] = True
        w[rng.integers(0, GRID - 3):][:3, c] = False
    for r in range(GRID // 4, GRID, GRID // 4):          # horizontal partitions with a door gap
        w[r, :] = True
        g = rng.integers(0, GRID - 3); w[r, g:g + 3] = False
    return w


def m_maze(rng):
    w = np.zeros((GRID, GRID), bool)                     # serpentine comb: alternating long walls
    for i, r in enumerate(range(2, GRID - 1, 3)):
        w[r, :] = True
        if i % 2 == 0:
            w[r, GRID - 2:] = False                      # gap on the right
        else:
            w[r, :2] = False                             # gap on the left
    return w


def m_pillars(rng):
    w = np.zeros((GRID, GRID), bool)                     # regular lattice of 2x2 blocks
    for r in range(2, GRID - 1, 4):
        for c in range(2, GRID - 1, 4):
            w[r:r + 2, c:c + 2] = True
    return w


def m_diag(rng):
    w = np.zeros((GRID, GRID), bool)                     # diagonal barrier w/ a gap (non-axis)
    for i in range(GRID):
        j = i + int(rng.integers(-1, 2))
        if 0 <= j < GRID:
            w[i, j] = True
    gap = rng.integers(0, GRID - 3)
    for k in range(gap, gap + 4):
        if 0 <= k < GRID:
            w[k, :] = w[k, :] & False                    # clear a horizontal band = a doorway
            w[k] = False
    return w


def m_clutter_dense(rng):
    w = np.zeros((GRID, GRID), bool)                     # many small blocks, denser than training
    for _ in range(GRID):
        r = rng.integers(0, GRID - 2); c = rng.integers(0, GRID - 2)
        w[r:r + 2, c:c + 2] = True
    return w


def m_random_obstacle(rng):
    w = np.zeros((GRID, GRID), bool)                     # scattered single-cell obstacles (~12%)
    idx = rng.choice(GRID * GRID, size=int(0.12 * GRID * GRID), replace=False)
    w.flat[idx] = True
    return w


def m_corridors(rng):
    w = np.zeros((GRID, GRID), bool)                     # long parallel corridors, staggered doors
    for i, c in enumerate(range(3, GRID - 1, 4)):
        w[:, c] = True
        g = rng.integers(0, GRID - 4)
        w[g:g + 4, c] = False                            # one doorway per wall, staggered
    return w


GENS = {"open": m_open, "rooms": m_rooms, "maze": m_maze, "pillars": m_pillars,
        "diag": m_diag, "clutter_dense": m_clutter_dense,
        "random_obstacle": m_random_obstacle, "corridors": m_corridors}


def free_cell(rng, wall):
    free = np.argwhere(~wall)
    return free[rng.integers(0, len(free))]


# ---------------- oracle (4-conn geodesic) + metrics ----------------
def oracle(wall, goal):
    goal_oh = jnp.zeros((GRID, GRID)).at[goal[0], goal[1]].set(1.0)
    return propagate(goal_oh, (~jnp.asarray(wall)).astype(jnp.float32), KPROP, GAMMA)


def main():
    planner = load_planner(PLANNER, in_ch=2, K=KPROP, gamma=GAMMA)
    field_j = eqx.filter_jit(planner.net.field)
    oracle_j = eqx.filter_jit(oracle)
    lim = jnp.array([GRID - 1, GRID - 1])
    D = np.asarray(_DELTAS)

    def nbrs(cell):
        return np.clip(cell[None] + D, 0, [GRID - 1, GRID - 1])

    def opt_move(V, O, cell):
        nb = nbrs(cell); Vn = np.asarray(V)[nb[:, 0], nb[:, 1]]; On = np.asarray(O)[nb[:, 0], nb[:, 1]]
        a = int(np.argmax(Vn))
        return float(On[a]) >= float(On.max()) - 1e-3          # achieves best oracle value

    def reaches(V, wall, start, goal, budget):
        V = np.asarray(V); cur = start.copy(); seen = set()
        for _ in range(budget):
            if tuple(cur) == tuple(goal):
                return True
            nb = nbrs(cur); a = int(np.argmax(V[nb[:, 0], nb[:, 1]])); nxt = nb[a]
            if tuple(nxt) == tuple(cur) or tuple(nxt) in seen:  # stuck / cycle -> trapped
                return False
            seen.add(tuple(cur)); cur = nxt
        return tuple(cur) == tuple(goal)

    tasks = list(GENS.items())
    # add held-out SAR maps if available (32x32)
    sar = None
    if GRID == 32 and os.path.exists(SARP):
        sar = [np.zeros((32, 32), bool) for _ in json.load(open(SARP))]
        for m, wa in zip(json.load(open(SARP)), sar):
            wa.flat[m["walls"]] = True
        tasks.append(("SAR_heldout", None))

    print(f"=== planner generalization: {PLANNER} @ {GRID}x{GRID}, {N} samples/type ===", flush=True)
    print(f"{'map type':16s} {'optimal-move':>13s} {'reach-goal':>12s}   (trained on: open+random-rects only)", flush=True)
    rng = np.random.default_rng(0)
    budget = 4 * GRID
    results = {}
    for name, gen in tasks:
        nopt = nreach = tot = 0
        for i in range(N):
            wall = (sar[i % len(sar)] if name == "SAR_heldout" else gen(rng))
            goal = free_cell(rng, wall); start = free_cell(rng, wall)
            if tuple(start) == tuple(goal):
                continue
            V = field_j(mvprop_input(jnp.asarray(wall), jnp.asarray(goal)))
            O = oracle_j(wall, goal)
            # only count reach when the goal is actually reachable from start under the oracle
            if float(np.asarray(O)[start[0], start[1]]) <= 1e-6:
                continue
            nopt += int(opt_move(V, O, start)); nreach += int(reaches(V, wall, start, goal, budget)); tot += 1
        om, rc = 100 * nopt / max(tot, 1), 100 * nreach / max(tot, 1)
        results[name] = {"optimal_move": round(om, 1), "reach": round(rc, 1), "n": tot}
        flag = "" if rc >= 95 else ("  <-- WEAK" if rc >= 80 else "  <-- FAILS")
        print(f"{name:16s} {om:11.1f}% {rc:10.1f}%   n={tot}{flag}", flush=True)
    json.dump(results, open(f"planner_gen_{GRID}.json", "w"), indent=2)
    print("=== GEN DONE ===", flush=True)


if __name__ == "__main__":
    main()
