"""MVProp — learned max-propagation navigation planner (Value/Max-Propagation Networks,
Nardelli 2018). A per-cell value field floods outward from the goal, discounted per step and
blocked by walls; the value at the agent's move-neighbours are the move scores
(differentiable -> trainable). Drop-in for ctde_v0's classical ``nav_distance_field``.

Root-cause note (probe_mvprop.py, 2026-08-02). A from-scratch MVProp whose reward map ``r`` AND
passability ``w`` are both free conv outputs does NOT train under RL: at init ``w = sigmoid(0)
~ 0.5`` so the flood decays ~0.5 per step and dies within ~5 cells; the field is ~0 and flat
where agents actually are, the move scores are identical, and the gradient through K max-prop
sweeps (max-sparsity x w^K decay) vanishes -> stuck. Two structural fixes make it trainable:

  * ANCHOR the reward to the goal.  ``r`` is the goal one-hot channel (a unit spike AT the goal),
    not a free conv — the flood always originates strongly at the goal.
  * FLOOD FAR AT INIT.  ``w = sigmoid(gain_conv(x) + w_init_bias)`` with a large positive
    ``w_init_bias`` so free cells start at ``w ~ 0.9`` and the flood reaches the whole map; the
    only thing left to LEARN is to drop ``w -> 0`` at walls so the field routes around them.

With those, the value field at init is already a usable ``gamma^dist`` gradient, and
distillation toward the classical wavefront (controller.nav_distance_field) drives it to route.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

# ACTION_DELTAS order (matches zymera.env, N_ACTIONS=5): 0=stay 1=N 2=E 3=S 4=W
_DELTAS = jnp.array([[0, 0], [-1, 0], [0, 1], [1, 0], [0, -1]])


def _max_4nbr(V):
    """(H,W) elementwise max over the 4 von-Neumann neighbours (out-of-grid = -inf).

    Every real cell of a grid with H,W >= 2 has at least one in-grid neighbour, so the
    returned max is finite everywhere — no -inf ever enters the propagated field.
    """
    up = jnp.pad(V[1:, :], ((0, 1), (0, 0)), constant_values=-jnp.inf)
    down = jnp.pad(V[:-1, :], ((1, 0), (0, 0)), constant_values=-jnp.inf)
    left = jnp.pad(V[:, 1:], ((0, 0), (0, 1)), constant_values=-jnp.inf)
    right = jnp.pad(V[:, :-1], ((0, 0), (1, 0)), constant_values=-jnp.inf)
    return jnp.maximum(jnp.maximum(up, down), jnp.maximum(left, right))


def propagate(r, w, K, gamma=0.9):
    """Max-propagation value field.

    ``V0 = r``; ``V_{k+1}(x) = max( r(x), gamma * w(x) * max_4nbr(V_k)(x) )`` for K sweeps.
    With ``r`` = goal reward (positive at the goal, ~0 elsewhere) and ``w`` = passability
    (0 at walls, 1 on free cells), value floods from the goal, decays by ``gamma`` per step,
    and does NOT cross walls (w=0 blocks the product). Reachable free cells get ``gamma^d``
    times the goal reward; walls and cut-off cells stay ~0. Higher V = closer to the goal.
    """
    def step(V, _):
        return jnp.maximum(r, gamma * w * _max_4nbr(V)), None
    V, _ = jax.lax.scan(step, r, None, length=K)
    return V


class MVProp(eqx.Module):
    """Learned MVProp planner. A conv turns the map channels into per-cell passability
    ``w in (0,1)``; the reward map ``r`` is the goal one-hot channel (anchored spike);
    ``propagate`` floods the value field; ``move_scores`` reads V at the agent's neighbours.

    Input ``x`` is ``(C,H,W)``; channel ``goal_ch`` (default the LAST channel) is the goal
    one-hot. All other channels feed the passability conv.
    """
    gain_conv: eqx.nn.Conv2d
    K: int = eqx.field(static=True)
    gamma: float = eqx.field(static=True)
    goal_ch: int = eqx.field(static=True)

    def __init__(self, in_ch, K, *, key, gamma=0.9, goal_ch=-1, w_init_bias=2.0):
        conv = eqx.nn.Conv2d(in_ch, 1, 3, padding=1, key=key)
        # Flood-far-at-init: bias the passability logit up so sigmoid(bias) ~ 0.88 on free
        # cells before any training (the probe's dead flood came from bias 0 -> w~0.5).
        conv = eqx.tree_at(lambda c: c.bias, conv,
                           conv.bias * 0.0 + jnp.asarray(w_init_bias, conv.bias.dtype))
        self.gain_conv = conv
        self.K = int(K)
        self.gamma = float(gamma)
        self.goal_ch = int(goal_ch)

    def field(self, x):
        """(C,H,W) input -> (H,W) value field. r = goal one-hot; w = sigmoid(conv(x))."""
        r = x[self.goal_ch]                              # (H,W) anchored goal spike
        w = jax.nn.sigmoid(self.gain_conv(x)[0])         # (H,W) learned passability in (0,1)
        return propagate(r, w, self.K, self.gamma)

    def move_scores(self, x, rc):
        """(C,H,W) input, (2,) agent cell -> (5,) neighbour values in ACTION_DELTAS order.
        Higher = better move (ascends the value field toward the goal)."""
        V = self.field(x)
        H, W = V.shape
        cells = jnp.clip(rc[None, :] + _DELTAS, 0, jnp.array([H - 1, W - 1]))   # (5,2)
        return V[cells[:, 0], cells[:, 1]]               # (5,)

    # back-compat alias (older callers used ``move_logits``)
    move_logits = move_scores
