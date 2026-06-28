"""Tests for the connectivity-safe crowded terrains (ctde_v0.terrains).

The contract: every generator yields a wall mask whose FREE space is a single
4-connected component (coverage well-posed, spawn always has free cells), and
build_env wires each one so reset places agents on free cells and a step runs.

    JAX_PLATFORMS=cpu \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m pytest \
        ctde_v0/tests/test_terrains.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))  # .../SharedExploration
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from ctde_v0 import env_utils, terrains  # noqa: E402
from ctde_v0.config import CTDEConfig, World  # noqa: E402


def _n_free_components(wall: np.ndarray) -> int:
    """Exact host-side count of 4-connected free components."""
    h, w = wall.shape
    free = ~wall
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


_TERRAINS = [
    ("clutter-light", terrains.ConnectedClutter(n_obstacles=60), 24),
    ("clutter-heavy", terrains.ConnectedClutter(n_obstacles=280), 32),
    ("pillars-4-2", terrains.Pillars(spacing=4, size=2), 24),
    ("pillars-5-3", terrains.Pillars(spacing=5, size=3), 32),
    ("mixed-r3", terrains.MixedCluttRooms(rooms=3, n_obstacles=50), 24),
    ("mixed-r4", terrains.MixedCluttRooms(rooms=4, n_obstacles=90), 32),
]


@pytest.mark.parametrize("name,terr,g", _TERRAINS, ids=[t[0] for t in _TERRAINS])
def test_free_space_single_component(name, terr, g):
    """Free space is ONE connected component, non-trivial, across reset keys."""
    f = jax.jit(lambda k: terr.walls(k, g, g))
    for s in range(5):
        wall = np.asarray(f(jax.random.PRNGKey(s)))
        assert wall.shape == (g, g)
        assert _n_free_components(wall) == 1, f"{name} seed {s}: not connected"
        assert (~wall).mean() > 0.25, f"{name} seed {s}: too little free space"


@pytest.mark.parametrize("terrain,kw", [
    ("clutter", dict(n_obstacles=120)),
    ("pillars", dict(pillar_spacing=4, pillar_size=2)),
    ("mixed", dict(rooms=3, n_obstacles=60)),
    ("crowded_mix", dict(n_obstacles=120, rooms=3)),
])
def test_build_env_crowded_reset_and_step(terrain, kw):
    """build_env wires the terrain; agents spawn on free cells and a step runs."""
    cfg = CTDEConfig(
        world=World(grid=20, n_agents=6, comm_r=4, horizon=5, terrain=terrain, **kw),
    )
    env = env_utils.build_env(cfg)
    obs, st = env.reset(jax.random.PRNGKey(0))
    wall = np.asarray(st.wall)
    pos = np.asarray(st.body.position)
    assert not wall[pos[:, 0], pos[:, 1]].any(), "an agent spawned inside a wall"
    assert _n_free_components(wall) == 1, "env wall mask free space not connected"
    # one step must not raise
    obs2, st2, *_ = env.step(st, jnp.zeros((6,), jnp.int32), jax.random.PRNGKey(1))
    assert st2.body.position.shape == (6, 2)
