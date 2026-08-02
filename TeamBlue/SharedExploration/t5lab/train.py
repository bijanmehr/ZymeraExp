"""Isolated minimal MAPPO train step + loop for Phase 1a.

DTE (team value = mean of the per-agent value head), two-action PG (goal offset + MVProp move),
GAE + clipped-PG + value MSE + entropy, AdamW. 16 rollouts, never reduced. Reuses ctde_v0's
_gae / make_optimizer / make_stencil / goal_targets read-only. No dual (free-space Phase 1a;
the λ₂ guardrail is dormant), no selector/roles/difference-credit.

BALTHAR-VERIFY throughout: PPO broadcast bookkeeping (per-agent ratio × team advantage),
optimizer construction, and cfg field names are inferred from ctde_v0 and need one run to confirm.
"""
import jax
import jax.numpy as jnp
import equinox as eqx
import optax

from ctde_v0.ppo import _gae, make_stencil, make_optimizer   # READ-ONLY
from ctde_v0.controller import goal_targets                  # READ-ONLY
from t5lab.rollout import rollout


def _advantages(traj, gamma, lam):
    """(B,T) advantage + return from the team reward/value via ctde_v0._gae."""
    adv, ret = jax.vmap(lambda r, v, vl: _gae(r, v, vl, gamma, lam))(
        traj["rew_team"], traj["v_team"], traj["v_last"])
    return adv, ret


def _eval_logp_ent_val(actor, step, stencil, key):
    """Recompute (goal_logp, move_logp, team_value, entropy) for ONE timestep's stored data
    under the current params. step = dict of that timestep's (obs, adj, pos, goal_idx, move, blocked)."""
    obs, adj, pos = step["obs"], step["adj"], step["pos"]
    z = actor.belief(obs, adj, key=key, inference=False)
    goal_logits, value, _l2 = actor.heads(z)
    glp = jnp.take_along_axis(jax.nn.log_softmax(goal_logits, -1), step["goal_idx"][:, None], -1)[:, 0]
    h, w = obs.shape[-2], obs.shape[-1]
    gcells = goal_targets(pos, stencil, h, w)
    goal = gcells[jnp.arange(pos.shape[0]), step["goal_idx"]]
    mlogits = actor.move_logits(pos, goal, step["blocked"])
    mlp = jnp.take_along_axis(jax.nn.log_softmax(mlogits, -1), step["move"][:, None], -1)[:, 0]
    ent = (-(jax.nn.softmax(mlogits, -1) * jax.nn.log_softmax(mlogits, -1)).sum(-1)
           - (jax.nn.softmax(goal_logits, -1) * jax.nn.log_softmax(goal_logits, -1)).sum(-1))
    return glp, mlp, value.mean(), ent.mean()


def _ppo_loss(actor, batch, adv, ret, old_logp, stencil, clip, vf, ent_c, key):
    """Clipped-PG + value MSE + entropy over a (B,T) minibatch. old_logp = per-agent goal+move
    logp summed over agents, (B,T). BALTHAR-VERIFY the per-agent broadcast."""
    B, T = adv.shape

    def one_step(step, k):
        glp, mlp, vteam, ent = _eval_logp_ent_val(actor, step, stencil, k)
        return (glp + mlp).sum(), vteam, ent          # sum agents' joint logp -> ()

    ks = jax.random.split(key, B * T).reshape(B, T, 2)
    # vmap over B and T; `batch` is a pytree with leading (B,T)
    new_logp, vteam, ent = jax.vmap(jax.vmap(one_step))(batch, ks)   # (B,T) each
    ratio = jnp.exp(new_logp - old_logp)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
    vloss = vf * ((vteam - ret) ** 2).mean()
    ent_loss = -ent_c * ent.mean()
    return pg + vloss + ent_loss, dict(pg=pg, vloss=vloss, ent=ent.mean())


def train_step(env, actor, opt, opt_state, cfg, key, stencil):
    """One PPO iteration: 16 rollouts -> GAE -> 6 epochs x 4 minibatches clipped-PG."""
    t = cfg.trainer
    B = int(getattr(t, "rollouts_per_iter", 16))     # 16, never reduced
    rk, tk = jax.random.split(key)
    seeds = jax.random.split(rk, B)
    traj = jax.vmap(lambda s: rollout(env, actor, cfg, s, stencil))(seeds)   # (B,T,...)
    adv, ret = _advantages(traj, t.gamma, t.gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    old_logp = (traj["goal_logp"] + traj["move_logp"]).sum(-1)               # (B,T) joint over agents

    grad_fn = eqx.filter_value_and_grad(_ppo_loss, has_aux=True)
    for _ep in range(int(getattr(t, "ppo_epochs", 6))):
        tk, mbk = jax.random.split(tk)
        # 4 minibatches over the B seeds
        for mb in jnp.array_split(jax.random.permutation(mbk, B), 4):
            sub = jax.tree_util.tree_map(lambda x: x[mb], traj)
            (loss, logs), grad = grad_fn(actor, sub, adv[mb], ret[mb], old_logp[mb],
                                         stencil, t.clip, t.vf_coef, t.ent_coef, tk)
            updates, opt_state = opt.update(grad, opt_state, eqx.filter(actor, eqx.is_array))
            actor = eqx.apply_updates(actor, updates)
    metrics = dict(ret=ret.mean(), **{k: v for k, v in logs.items()})
    return actor, opt_state, metrics


def train(env, actor, cfg, key):
    """Full loop over cfg.iters. Returns (actor, history)."""
    opt = make_optimizer(cfg)                        # BALTHAR-VERIFY: signature (cfg) -> optax GradientTransformation
    opt_state = opt.init(eqx.filter(actor, eqx.is_array))
    stencil = make_stencil(cfg)
    hist = []
    k = key
    for it in range(int(cfg.iters)):
        k, sk = jax.random.split(k)
        actor, opt_state, m = eqx.filter_jit(train_step)(env, actor, opt, opt_state, cfg, sk, stencil)
        hist.append({kk: float(v) for kk, v in m.items()})
    return actor, hist
