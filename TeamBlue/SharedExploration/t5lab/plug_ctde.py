"""ctde_v0 nav-field adapter.

Turns MVProp's per-agent value field into an ``(N,H,W)`` DISTANCE-like field matching the
contract of ctde_v0's ``controller.nav_distance_field`` (goal = minimum, descend to it), so
the existing reactive descent controller reuses unchanged. MVProp produces a VALUE field
(goal = maximum, ascend), so the adapter negates it.

Reads only MVProp (jax/equinox) — no ctde_v0 import — so it smoke-tests locally.
"""
import jax
import jax.numpy as jnp


def _one_agent_field(mvprop, goal_rc, blocked_hw):
    H, W = blocked_hw.shape
    goal_onehot = jnp.zeros((H, W)).at[goal_rc[0], goal_rc[1]].set(1.0)
    x = jnp.stack([blocked_hw.astype(jnp.float32), goal_onehot])   # (2,H,W)
    return -mvprop.field(x)                                        # value -> distance-like


def mvprop_field(mvprop, goal_cells, blocked):
    """(N,H,W) distance-like field. ``goal_cells`` (N,2) int; ``blocked`` (N,H,W) bool."""
    return jax.vmap(lambda g, b: _one_agent_field(mvprop, g, b))(goal_cells, blocked)
