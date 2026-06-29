"""Scripted-compass ceiling probe — the locality falsification test (LOCALITY_DESIGN §5).

A hand-coded Voronoi + in-cell-frontier controller (god-view ORACLE): each agent heads to the
nearest uncovered free cell in ITS OWN Voronoi cell. Measures, across the density-pinned ladder,
whether LOCALITY alone (perfect partitioning) reaches the coverage target within the 100-step budget,
and whether redundancy inverts toward 1 (disjoint coverage).

The diagnostic this settles: if even this oracle can't hit ~90% @32²/100-steps, then ~45% is a
decentralized INFO/architecture CEILING (no teacher — human teleop or scripted — can exceed it), not a
training failure. If it sails past 90%, the gap is a training/exploration failure (a warm-start / better
objective CAN help). Gates the distillation + coordination direction.

    PYTHONPATH=. <zymera_lab venv>/bin/python ctde_v0/compass_ceiling.py
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

import zymera

DELTAS = np.asarray(zymera.ACTION_DELTAS)          # (A, 2) drow, dcol  (STAY=0, then moves)
BIG = 1 << 20


def voronoi_targets(pos, covered, wall):
    """(N,2) target cell per agent: nearest UNCOVERED free cell in its Voronoi cell (Manhattan).
    Falls back to the global nearest uncovered cell when the agent's own cell is fully covered."""
    h, w = wall.shape
    n = pos.shape[0]
    yy, xx = np.mgrid[0:h, 0:w]
    d = np.abs(pos[:, 0][:, None, None] - yy[None]) + np.abs(pos[:, 1][:, None, None] - xx[None])
    assign = np.argmin(d, axis=0)                  # (H,W) nearest-agent id
    uncov = (~wall) & (~covered)                   # (H,W) free & uncovered
    targets = pos.copy()
    for i in range(n):
        mine = uncov & (assign == i)
        cand = mine if mine.any() else uncov
        if not cand.any():
            continue                               # nothing left -> stay
        di = np.where(cand, np.abs(yy - pos[i, 0]) + np.abs(xx - pos[i, 1]), BIG)
        idx = int(np.argmin(di))
        targets[i] = (idx // w, idx % w)
    return targets


def greedy_actions(pos, targets, wall):
    """(N,) action per agent: the move that most reduces Manhattan distance to target,
    never stepping into a wall or off-grid (STAY if no move helps)."""
    h, w = wall.shape
    n = pos.shape[0]
    acts = np.zeros(n, np.int32)
    for i in range(n):
        best_a = 0
        best_d = abs(pos[i, 0] - targets[i, 0]) + abs(pos[i, 1] - targets[i, 1])
        for a in range(DELTAS.shape[0]):
            nr, nc = int(pos[i, 0] + DELTAS[a, 0]), int(pos[i, 1] + DELTAS[a, 1])
            if 0 <= nr < h and 0 <= nc < w and not wall[nr, nc]:
                dd = abs(nr - targets[i, 0]) + abs(nc - targets[i, 1])
                if dd < best_d:
                    best_d, best_a = dd, a
        acts[i] = best_a
    return acts


def lambda2(pos, comm_r):
    """Algebraic connectivity of the Chebyshev disk comm graph."""
    n = pos.shape[0]
    if n < 2:
        return 0.0
    dist = np.max(np.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)
    a = (dist <= comm_r).astype(float)
    np.fill_diagonal(a, 0.0)
    lap = np.diag(a.sum(1)) - a
    return float(np.sort(np.linalg.eigvalsh(lap))[1])


def run(grid, n, comm_r=5, horizon=100, seed=0, n_obstacles=0, cover_r=0):
    env = zymera.make("comm-coverage", grid=grid, n_agents=n, comm_r=comm_r,
                      sense_walls=True, spawn_radius=2, n_obstacles=n_obstacles,
                      cover_r=cover_r)
    key = jax.random.PRNGKey(seed)
    _obs, world = env.reset(key)
    l2s = []
    for _ in range(horizon):
        pos = np.asarray(world.body.position)
        covered = np.asarray(world.covered)
        wall = np.asarray(world.wall)
        tgt = voronoi_targets(pos, covered, wall)
        acts = greedy_actions(pos, tgt, wall)
        key, sk = jax.random.split(key)
        _obs, world, _r, _done, _info = env.step(world, jnp.asarray(acts), sk)
        l2s.append(lambda2(np.asarray(world.body.position), comm_r))
    covered = np.asarray(world.covered)
    wall = np.asarray(world.wall)
    seen_by = np.asarray(world.seen_by)
    cov = covered.sum() / max((~wall).sum(), 1)
    redund = seen_by.sum() / max(covered.sum(), 1)     # 1.0 = disjoint, N = all overlap
    return cov, redund, float(np.mean(l2s)), float(np.mean([x > 0.5 for x in l2s]))


def neighbor_voronoi_targets(pos, beliefs, comm_r):
    """DECENTRALIZED compass target per agent: nearest cell the agent does NOT yet
    believe it knows, inside its Voronoi cell among only its IN-RANGE neighbours.
    ``beliefs[i]`` = agent i's own post-gossip belief (covered-free ∪ sensed walls);
    so ``~beliefs[i]`` = unknown/uncovered ground, and known walls are excluded for free."""
    n = pos.shape[0]
    h, w = beliefs.shape[1], beliefs.shape[2]
    yy, xx = np.mgrid[0:h, 0:w]
    cheb = np.max(np.abs(pos[:, None, :] - pos[None, :, :]), axis=-1)   # (N,N)
    targets = pos.copy()
    for i in range(n):
        d_self = np.abs(yy - pos[i, 0]) + np.abs(xx - pos[i, 1])
        owned = np.ones((h, w), bool)
        for j in range(n):
            if j == i or cheb[i, j] > comm_r:                          # only in-range neighbours
                continue
            owned &= d_self <= (np.abs(yy - pos[j, 0]) + np.abs(xx - pos[j, 1]))
        unknown = ~beliefs[i]
        cand = owned & unknown
        if not cand.any():
            cand = unknown                                             # my cell fully known -> help globally
        if not cand.any():
            continue
        idx = int(np.argmin(np.where(cand, d_self, BIG)))
        targets[i] = (idx // w, idx % w)
    return targets


def run_decentralized(grid, n, comm_r=5, horizon=100, seed=0, cover_r=0):
    """The DECENTRALIZED scripted compass — belief-only + in-range-neighbour Voronoi
    (no god view, no connectivity enforcement). The realistic ceiling for compass-as-feature."""
    env = zymera.make("comm-coverage", grid=grid, n_agents=n, comm_r=comm_r,
                      sense_walls=True, spawn_radius=2, cover_r=cover_r)
    key = jax.random.PRNGKey(seed)
    _obs, world = env.reset(key)
    l2s = []
    for _ in range(horizon):
        pos = np.asarray(world.body.position)
        beliefs = np.asarray(world.channel.shared)        # (N,H,W) per-agent belief
        wall = np.asarray(world.wall)
        tgt = neighbor_voronoi_targets(pos, beliefs, comm_r)
        acts = greedy_actions(pos, tgt, wall)
        key, sk = jax.random.split(key)
        _obs, world, _r, _d, _i = env.step(world, jnp.asarray(acts), sk)
        l2s.append(lambda2(np.asarray(world.body.position), comm_r))
    covered = np.asarray(world.covered)
    wall = np.asarray(world.wall)
    seen_by = np.asarray(world.seen_by)
    cov = covered.sum() / max((~wall).sum(), 1)
    redund = seen_by.sum() / max(covered.sum(), 1)
    return cov, redund, float(np.mean(l2s)), float(np.mean([x > 0.5 for x in l2s]))


def main():
    ladder = [(16, 4), (24, 6), (32, 10)]
    for label, fn in [("god-view oracle (global pos + true coverage)", run),
                      ("decentralized compass (belief + in-range Voronoi)", run_decentralized)]:
        print(f"\n=== {label} ===")
        print(f"{'scale':>11} {'cov%':>6} {'redund':>7} {'meanλ2':>7} {'conn>.5':>8}   (3 seeds)")
        for grid, n in ladder:
            rows = [fn(grid, n, seed=s) for s in range(3)]
            cov, red, l2, sc = (np.mean([r[k] for r in rows]) for k in range(4))
            print(f"{f'{grid}x{grid}/{n}':>11} {100*cov:>5.1f} {red:>7.2f} {l2:>7.2f} {100*sc:>6.0f}%")


if __name__ == "__main__":
    main()
