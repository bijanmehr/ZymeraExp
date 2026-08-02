import jax
import jax.numpy as jnp
import equinox as eqx
from t5lab.mvprop import propagate, MVProp


def test_propagate_floods_from_goal_open():
    """On open ground, value is maximal at the goal and decreases with distance."""
    H = W = 9
    r = jnp.zeros((H, W)).at[4, 8].set(1.0)          # goal reward at (4,8)
    w = jnp.ones((H, W))                              # open terrain
    V = propagate(r, w, K=32)
    assert int(jnp.argmax(V)) == 4 * W + 8           # goal is the global max
    assert V[4, 7] > V[4, 0]                          # one step from goal > eight steps


def test_propagate_blocked_by_wall():
    """A full wall (no gap) blocks propagation: the cut-off side stays ~0."""
    H = W = 7
    r = jnp.zeros((H, W)).at[3, 6].set(1.0)          # goal on the right
    w = jnp.ones((H, W)).at[:, 3].set(0.0)           # full vertical wall at column 3, NO gap
    V = propagate(r, w, K=64)
    assert V[3, 6] == V.max()                        # goal is the max
    assert V[3, 0] < 1e-6                             # left side is cut off -> ~0
    assert V[3, 5] > 0.1                              # right side (with goal) has value


def test_mvprop_module_shapes_and_grad():
    """Module builds, is size-invariant (same params on 16 and 32), and is differentiable."""
    m = MVProp(in_ch=2, K=16, key=jax.random.PRNGKey(0))
    for H in (16, 32):
        x = jnp.zeros((2, H, H)).at[1, H // 2, H - 1].set(1.0)   # ch1 = goal one-hot
        V = m.field(x)
        assert V.shape == (H, H)
        logits = m.move_logits(x, jnp.array([H // 2, 0]))
        assert logits.shape == (5,)

    def loss(mod):
        x = jnp.zeros((2, 16, 16)).at[1, 8, 15].set(1.0)
        return mod.move_logits(x, jnp.array([8, 0])).sum()

    g = eqx.filter_grad(loss)(m)
    leaves = [l for l in jax.tree_util.tree_leaves(g) if hasattr(l, "shape")]
    assert leaves and all(jnp.isfinite(l).all() for l in leaves)
