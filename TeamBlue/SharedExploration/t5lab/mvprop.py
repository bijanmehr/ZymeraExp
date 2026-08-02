"""MVProp — learned max-propagation navigation planner (Value/Max-Propagation Networks,
Nardelli 2018). A per-cell value field floods outward from the goal, discounted per step and
blocked by walls; the value at the agent's 5 move-neighbours are the move logits
(differentiable → RL-trainable). Drop-in for ctde_v0's classical ``nav_distance_field``.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

# ACTION_DELTAS order: 0=stay 1=up 2=right 3=down 4=left
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


def propagate(r, w, K, gamma=0.99):
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
    """Learned MVProp. Two 3x3 convs turn the input map (channels e.g. [blocked, goal_onehot])
    into a per-cell reward ``r`` and a propagation gain ``w in (0,1)``; ``propagate`` floods the
    value field; ``move_logits`` reads V at the agent's 5 move-neighbours."""
    reward_conv: eqx.nn.Conv2d
    gain_conv: eqx.nn.Conv2d
    K: int = eqx.field(static=True)
    gamma: float = eqx.field(static=True)

    def __init__(self, in_ch, K, *, key, gamma=0.99):
        kr, kg = jax.random.split(key)
        self.reward_conv = eqx.nn.Conv2d(in_ch, 1, 3, padding=1, key=kr)
        self.gain_conv = eqx.nn.Conv2d(in_ch, 1, 3, padding=1, key=kg)
        self.K = int(K)
        self.gamma = float(gamma)

    def field(self, x):
        """(C,H,W) input -> (H,W) value field."""
        r = self.reward_conv(x)[0]                     # (H,W)
        w = jax.nn.sigmoid(self.gain_conv(x)[0])       # (H,W) in (0,1)
        return propagate(r, w, self.K, self.gamma)

    def move_logits(self, x, rc):
        """(C,H,W) input, (2,) agent cell -> (5,) move logits in ACTION_DELTAS order."""
        V = self.field(x)
        H, W = V.shape
        cells = jnp.clip(rc[None, :] + _DELTAS, 0, jnp.array([H - 1, W - 1]))  # (5,2)
        return V[cells[:, 0], cells[:, 1]]             # (5,) higher = better move
