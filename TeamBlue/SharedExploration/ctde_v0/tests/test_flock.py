"""Unit tests for the flock (connectivity-repair) skill — scripted + learned.

CPU-only by construction. Run from the SharedExploration working dir:
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m pytest \
        ctde_v0/tests/test_flock.py -q
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from ctde_v0.flock import FlockHead, scripted_flock_logits  # noqa: E402
from ctde_v0.nets import _compass_unit_dirs  # noqa: E402


def _offset_toward(K, direction):
    """Index of the compass offset whose unit direction best aligns with ``direction``
    (a (2,) unit row/col vector). Uses ``_compass_unit_dirs`` so the test never
    hardcodes the compass ordering."""
    dirs = _compass_unit_dirs(K)                                   # (K,2) unit dirs
    return int(jnp.argmax(dirs @ jnp.asarray(direction, dtype=jnp.float32)))


# ---- (a) two agents at the edge of comm range repair toward each other ------

def test_scripted_edge_of_range_points_at_partner():
    """2 agents at Chebyshev distance == comm_r (5): agent 0's argmax offset is the
    one pointing toward agent 1 (+col / East), agent 1's toward agent 0 (-col / West)."""
    K, comm_r = 9, 5
    position = jnp.array([[5, 5], [5, 10]], dtype=jnp.int32)        # cheby == 5 == comm_r
    logits = scripted_flock_logits(position, K, comm_r, sharp=2.0)
    assert logits.shape == (2, K)
    assert jnp.all(jnp.isfinite(logits))

    east = _offset_toward(K, (0.0, 1.0))                           # +col: agent0 -> agent1
    west = _offset_toward(K, (0.0, -1.0))                          # -col: agent1 -> agent0
    assert int(jnp.argmax(logits[0])) == east
    assert int(jnp.argmax(logits[1])) == west


# ---- (b) an isolated agent gets an all-zero (no-preference) row -------------

def test_scripted_isolated_agent_is_zero():
    """An agent whose nearest other agent is farther than comm_r away gets an all-zero
    logit row (no preference); the in-range pair still expresses a preference."""
    K, comm_r = 9, 5
    # agents 0,1 are a connected pair (dist 2); agent 2 sits 20 cells away (> comm_r).
    position = jnp.array([[5, 5], [5, 7], [5, 30]], dtype=jnp.int32)
    logits = scripted_flock_logits(position, K, comm_r, sharp=2.0)
    assert logits.shape == (3, K)
    # the isolated agent (row 2) -> all zeros.
    assert jnp.allclose(logits[2], 0.0)
    # the connected pair still has a non-trivial preference (sanity: not all zero).
    assert not jnp.allclose(logits[0], 0.0)
    assert not jnp.allclose(logits[1], 0.0)


# ---- (c) the learned FlockHead returns the right shape and is finite --------

def test_flock_head_shape_and_finite():
    """``FlockHead(width, K)(z)`` maps a per-agent belief (N,W) to (N,K) finite logits."""
    K, width, N = 9, 16, 4
    head = FlockHead(width=width, K=K, key=jax.random.PRNGKey(0))
    z = jax.random.normal(jax.random.PRNGKey(1), (N, width))
    out = head(z)
    assert out.shape == (N, K)
    assert jnp.all(jnp.isfinite(out))
