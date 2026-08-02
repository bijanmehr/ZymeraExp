"""T5Actor — isolated composition of ctde_v0's Backbone + a goal-region head + the learned
MVProp planner. Zero ctde_v0 edits: ``Backbone`` is imported read-only.

Two-level PG policy (both trained end-to-end):
  L3 goal:  z = Backbone(obs, adj); goal_logits = goal_head(z)  -> sample a goal offset.
  L1 move:  MVProp floods a value field over the belief occupancy toward the chosen goal;
            move_logits = field value at the agent's 5 move-neighbours -> sample the move.
The move being a *sampled* readout of the MVProp field is what makes MVProp RL-trainable
(a hard argmax descent would give it no gradient).
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from ctde_v0.nets import Backbone           # READ-ONLY import
from t5lab.mvprop import MVProp


class T5Actor(eqx.Module):
    backbone: Backbone
    goal_head: eqx.nn.Linear                 # W -> K_goal (goal-offset logits)
    mvprop: MVProp                           # learned L1 planner (in_ch=2: [blocked, goal])
    value_head: eqx.nn.Linear                # W -> 1 (per-agent value; team = mean, DTE)
    aux_head: eqx.nn.Linear                  # W -> 1 (actor-side λ̂₂, per the ledger)
    width: int = eqx.field(static=True)

    def __init__(self, in_ch, K_goal, K_prop, backbone_cfg, *, key, dropout=0.0):
        kb, kg, km, kv, ka = jax.random.split(key, 5)
        # Mirror ctde_v0 Actor's Backbone construction (nets.py:873-878).
        self.backbone = Backbone(
            in_ch, backbone_cfg.width, backbone_cfg.depth, backbone_cfg.mp_rounds,
            backbone_cfg.agg, backbone_cfg.heads, backbone_cfg.norm, dropout, key=kb,
            message_content=getattr(backbone_cfg, "message_content", "learned"),
            position_ground=getattr(backbone_cfg, "position_ground", False),
        )
        W = backbone_cfg.width
        self.goal_head = eqx.nn.Linear(W, K_goal, key=kg)
        self.mvprop = MVProp(in_ch=2, K=K_prop, key=km)
        self.value_head = eqx.nn.Linear(W, 1, key=kv)
        self.aux_head = eqx.nn.Linear(W, 1, key=ka)
        self.width = int(W)

    def belief(self, obs, adj_off, *, key=None, inference=False):
        """(C,H,W)-per-agent obs + (N,N) adjacency -> (N,W) belief."""
        return self.backbone(obs, adj_off, dist=None, key=key, inference=inference)

    def heads(self, z):
        """(N,W) belief -> (goal_logits (N,K_goal), value (N,), lambda2_hat (N,))."""
        goal_logits = jax.vmap(self.goal_head)(z)
        value = jax.vmap(self.value_head)(z)[:, 0]
        lambda2_hat = jax.vmap(self.aux_head)(z)[:, 0]
        return goal_logits, value, lambda2_hat

    def move_logits(self, positions, goal_cells, blocked):
        """(N,2) agent cells, (N,2) goal cells, (N,H,W) belief-occupancy -> (N,5) move logits.
        Higher = better move (ascends the MVProp value field toward the goal)."""
        def one(pos, goal, blk):
            H, W = blk.shape
            goal_onehot = jnp.zeros((H, W)).at[goal[0], goal[1]].set(1.0)
            x = jnp.stack([blk.astype(jnp.float32), goal_onehot])   # (2,H,W)
            return self.mvprop.move_logits(x, pos)                  # (5,)
        return jax.vmap(one)(positions, goal_cells, blocked)        # (N,5)
