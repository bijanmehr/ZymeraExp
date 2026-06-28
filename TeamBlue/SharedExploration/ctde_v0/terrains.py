"""Connectivity-safe crowded terrains for ctde_v0 (experiment-level).

These conform to ``zymera.worldgen.Terrain``: ``walls(key, h, w) -> (H, W) bool``
mask (True = wall), a frozen dataclass with a pure-JAX method (runs inside the
jitted reset, so NO python branching on traced values).

Every generator GUARANTEES the free space is one connected region. We don't
generate-and-reject (not JAX-traceable); we *construct* connectivity: scatter
obstacles, then a fixed-iteration BFS flood-fill from a central free seed turns
every cell the wavefront can't reach into wall. The surviving free set IS the
seed's 4-connected component — connected by definition, coverage well-posed
(100% reachable), spawn always has free cells. The styles (random clutter /
regular pillars / rooms+clutter) give the obstacle-field DIVERSITY; the fill
makes each one solvable.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp


def _bfs_reach(seed, free, n_iter):
    """(H,W) bool of cells 4-connected-reachable from ``seed`` through ``free``,
    after ``n_iter`` wavefront steps. ``seed`` must be a subset of ``free``."""
    h, w = free.shape

    def step(r, _):
        up = jnp.concatenate([r[1:, :], jnp.zeros((1, w), bool)], axis=0)
        dn = jnp.concatenate([jnp.zeros((1, w), bool), r[:-1, :]], axis=0)
        lf = jnp.concatenate([r[:, 1:], jnp.zeros((h, 1), bool)], axis=1)
        rt = jnp.concatenate([jnp.zeros((h, 1), bool), r[:, :-1]], axis=1)
        return (r | up | dn | lf | rt) & free, None

    reach, _ = jax.lax.scan(step, seed, None, length=n_iter)
    return reach


def connected_fill(wall, n_iter=None):
    """Wall off every free cell not reachable from the central seed → the
    remaining free space is a single 4-connected component.

    The center cell is forced free and used as the seed (for moderate obstacle
    density it lies in the giant component). ``n_iter`` defaults to ``2*(h+w)``,
    ample for clutter/pillars; the result is connected for ANY ``n_iter`` (too
    small just keeps a smaller blob — never disconnected)."""
    h, w = wall.shape
    free = ~wall
    cr, cc = h // 2, w // 2
    free = free.at[cr, cc].set(True)                 # guarantee a free seed
    seed = jnp.zeros((h, w), bool).at[cr, cc].set(True)
    reach = _bfs_reach(seed, free, n_iter or 2 * (h + w))
    return ~reach                                    # everything unreached → wall


def _scatter(key, h, w, n):
    n = min(int(n), h * w)
    idx = jax.random.choice(key, h * w, shape=(n,), replace=False)
    return jnp.zeros((h * w,), bool).at[idx].set(True).reshape(h, w)


@dataclass(frozen=True)
class ConnectedClutter:
    """``n_obstacles`` random wall cells, then flood-fill → connected free space.
    A well-posed version of :class:`zymera.worldgen.RandomWalls` for high density."""

    n_obstacles: int
    fill_iters: int = 0                              # 0 → auto 2*(h+w)

    def walls(self, key, h, w):
        wall = _scatter(key, h, w, self.n_obstacles)
        return connected_fill(wall, self.fill_iters or None)


@dataclass(frozen=True)
class Pillars:
    """A regular lattice of ``size``×``size`` obstacle blocks every ``spacing``
    cells (parking-garage / forest of columns); corridors of width
    ``spacing-size`` run between them. Connected by construction; flood-fill
    trims any boundary scraps. Density ≈ ``(size/spacing)**2``."""

    spacing: int = 4
    size: int = 2
    jitter: bool = True

    def walls(self, key, h, w):
        off = (jax.random.randint(key, (2,), 0, max(self.spacing, 1))
               if self.jitter else jnp.zeros((2,), jnp.int32))
        rb = ((jnp.arange(h) + off[0]) % self.spacing) < self.size       # (H,)
        cb = ((jnp.arange(w) + off[1]) % self.spacing) < self.size       # (W,)
        wall = rb[:, None] & cb[None, :]
        return connected_fill(wall, 2 * (h + w))


@dataclass(frozen=True)
class MixedCluttRooms:
    """Vertical rooms (wide doors) PLUS scattered clutter, flood-filled →
    structured corridors *and* unstructured obstacles in one connected map."""

    rooms: int = 3
    n_obstacles: int = 40
    door_w: int = 2

    def walls(self, key, h, w):
        from zymera.worldgen import Rooms
        k1, k2 = jax.random.split(key)
        rwall = Rooms(rooms=self.rooms, door_w=self.door_w).walls(k1, h, w)
        cwall = _scatter(k2, h, w, self.n_obstacles)
        return connected_fill(rwall | cwall, 3 * (h + w))


@dataclass(frozen=True)
class RandomCrowded:
    """Per-reset MIXTURE — draws one member style each reset key (``lax.switch``,
    so only the drawn style computes). One policy thus trains across clutter +
    pillars + mixed in a single run → generalises over crowded-map DIVERSITY
    instead of overfitting one obstacle style. Members must be frozen/hashable."""

    members: tuple

    def walls(self, key, h, w):
        k0, k1 = jax.random.split(key)
        i = jax.random.randint(k0, (), 0, len(self.members))
        branches = [lambda k, m=m: m.walls(k, h, w) for m in self.members]
        return jax.lax.switch(i, branches, k1)


def default_crowded_mix(n_obstacles, pillar_spacing, pillar_size, rooms):
    """The standard training mixture: random clutter, pillar lattice, rooms+clutter."""
    return RandomCrowded((
        ConnectedClutter(n_obstacles=int(n_obstacles)),
        Pillars(spacing=int(pillar_spacing), size=int(pillar_size)),
        MixedCluttRooms(rooms=int(rooms), n_obstacles=int(n_obstacles) // 2),
    ))


# --- self-test: connectivity guarantee + density -----------------------------
if __name__ == "__main__":
    import numpy as np

    def n_components(wall):
        """Count 4-connected free components (host-side, exact)."""
        h, w = wall.shape
        free = ~np.asarray(wall)
        seen = np.zeros_like(free)
        comps = 0
        for i in range(h):
            for j in range(w):
                if free[i, j] and not seen[i, j]:
                    comps += 1
                    stack = [(i, j)]
                    seen[i, j] = True
                    while stack:
                        r, c = stack.pop()
                        for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                            nr, nc = r + dr, c + dc
                            if 0 <= nr < h and 0 <= nc < w and free[nr, nc] and not seen[nr, nc]:
                                seen[nr, nc] = True
                                stack.append((nr, nc))
        return comps

    terrains = {
        "clutter@22%(24)": (ConnectedClutter(n_obstacles=130), 24),
        "clutter@27%(32)": (ConnectedClutter(n_obstacles=280), 32),
        "pillars 4/2(24)": (Pillars(spacing=4, size=2), 24),
        "pillars 5/3(32)": (Pillars(spacing=5, size=3), 32),
        "mixed r3+clut(24)": (MixedCluttRooms(rooms=3, n_obstacles=50), 24),
        "mixed r4+clut(32)": (MixedCluttRooms(rooms=4, n_obstacles=90), 32),
    }
    print(f"{'terrain':20s} {'grid':6s} {'free%':>6s} {'comps':>6s}  ok")
    all_ok = True
    for name, (t, g) in terrains.items():
        for s in range(3):
            wall = np.asarray(jax.jit(lambda k: t.walls(k, g, g))(jax.random.PRNGKey(s)))
            free_frac = 100.0 * (~wall).mean()
            comps = n_components(wall)
            ok = comps == 1 and free_frac > 25.0
            all_ok &= ok
            if s == 0:
                print(f"{name:20s} {g}x{g:<2d} {free_frac:6.1f} {comps:6d}  {'OK' if ok else 'FAIL'}")
            elif comps != 1:
                print(f"   seed{s}: comps={comps} FAIL")
    print("\nALL CONNECTED + non-trivial free space" if all_ok else "\nSOME FAILED")
