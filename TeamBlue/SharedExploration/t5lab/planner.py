"""MVPropPlanner — the learned L1 controller plugged into ctde_v0 in place of ``greedy_move``.

This is the ONLY new piece in the "exact previous architecture + planner" deal: an opaque,
FROZEN (distilled) module with a single ``move`` method whose signature and semantics mirror
``controller.navfield_move`` — descend a per-agent value field toward the chosen goal, one
env-valid step, with the hard collision veto and a STAY fallback. The classical wavefront's
``argmin distance`` becomes MVProp's ``argmax value`` (higher value = closer to goal); nothing
else changes. The RL loss never touches this module (the move is a deterministic readout during
collection, not part of loss_fn), so its distilled weights stay frozen throughout training.
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from zymera.env import ActionId
from ctde_v0.controller import occupied_cell_mask
from t5lab.mvprop import MVProp

_STAY = int(ActionId.STAY)


def mvprop_input(blocked_i, goal_i):
    """(H,W) known-walls + (2,) goal cell -> (2,H,W) MVProp input [blocked, goal_onehot].

    The SINGLE source of truth for the channel layout, shared by the planner and the
    distillation trainer so training and inference see byte-identical inputs.
    """
    h, w = blocked_i.shape
    goal_oh = jnp.zeros((h, w)).at[goal_i[0], goal_i[1]].set(1.0)
    return jnp.stack([blocked_i.astype(jnp.float32), goal_oh])          # (2,H,W)


class MVPropPlanner(eqx.Module):
    """Wraps a (distilled) :class:`MVProp` net as a drop-in L1 controller."""
    net: MVProp

    def fields(self, blocked, goal):
        """(N,H,W) known-walls + (N,2) goals -> (N,H,W) per-agent value fields."""
        return jax.vmap(lambda b, g: self.net.field(mvprop_input(b, g)))(blocked, goal)

    def move(self, pos, goal, blocked, valid_targets, action_valid, forbid_collision=True):
        """(N,) int32 move — MVProp value-field descent toward ``goal`` (mirror of
        :func:`controller.navfield_move`). ``pos`` (N,2), ``goal`` (N,2), ``blocked`` (N,H,W)
        per-agent known walls, ``valid_targets`` (N,A,2), ``action_valid`` (N,A).

        For each agent: flood a value field to its goal over its own known-walls map, then
        among env-VALID actions take the one whose committed cell has the HIGHEST value
        (steepest ascent = closest to goal). The hard collision veto removes moves onto a
        currently-occupied cell; STAY is always selectable and is the fallback when no valid
        move strictly beats staying — so the emitted move is always env-valid, exactly like
        the greedy / navfield controllers."""
        n = pos.shape[0]
        V = self.fields(blocked, goal)                                 # (N,H,W)

        def gather_agent(i):
            return V[i][valid_targets[i, :, 0], valid_targets[i, :, 1]]  # (A,) value if taken

        s = jax.vmap(gather_agent)(jnp.arange(n))                      # (N,A) higher = better
        s = jnp.where(action_valid, s, -jnp.inf)                       # forbid invalid actions
        if forbid_collision:
            s = jnp.where(occupied_cell_mask(pos, valid_targets), -jnp.inf, s)  # reactive veto
        best = jnp.argmax(s, axis=-1).astype(jnp.int32)                # steepest ascent
        s_best = jnp.take_along_axis(s, best[:, None], axis=-1)[:, 0]
        s_stay = s[:, _STAY]
        return jnp.where(s_best > s_stay, best, jnp.int32(_STAY))       # STAY unless a move helps


def load_planner(path, in_ch=2, K=32, gamma=0.9):
    """Deserialise a distilled MVPropPlanner from ``path`` (an eqx tree of a fresh template)."""
    template = MVPropPlanner(MVProp(in_ch=in_ch, K=K, key=jax.random.PRNGKey(0), gamma=gamma))
    return eqx.tree_deserialise_leaves(path, template)
