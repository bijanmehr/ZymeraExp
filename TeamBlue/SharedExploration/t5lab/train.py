"""Isolated minimal MAPPO train step + loop for Phase 1a.

DTE (team value = mean of the per-agent value head), two-action PG (goal offset + MVProp move),
GAE + clipped-PG + value MSE + entropy, AdamW. 16 rollouts, never reduced. Reuses ctde_v0's
_gae / make_optimizer / make_stencil / goal_targets read-only. No dual/selector/roles.

Verified config paths: rollouts_per_iter/iters top-level; gamma/gae_lambda/clip/ppo_epochs/
minibatches on cfg.trainer; vf_coef on cfg.loss; entropy on cfg.regularization; optax pattern
matches ctde_v0 (params = eqx.filter(actor, is_array); opt.update(grads, opt_state, params)).
"""
import jax
import jax.numpy as jnp
import equinox as eqx

from ctde_v0.ppo import _gae, make_stencil, make_optimizer   # READ-ONLY
from ctde_v0.controller import goal_targets                  # READ-ONLY
from t5lab.rollout import rollout


def _advantages(traj, gamma, lam):
    adv, ret = jax.vmap(lambda r, v, vl: _gae(r, v, vl, gamma, lam))(
        traj["rew_team"], traj["v_team"], traj["v_last"])
    return adv, ret                                          # (B,T) each


def _step_logp_val_ent(actor, step, stencil, key, controller="mvprop"):
    """Recompute (goal_logp+move_logp summed over agents, team_value, entropy) for ONE timestep."""
    obs, adj, pos = step["obs"], step["adj"], step["pos"]
    n = pos.shape[0]
    z = actor.belief(obs, adj, key=key, inference=False)
    goal_logits, value, _l2 = actor.heads(z)
    glp = jnp.take_along_axis(jax.nn.log_softmax(goal_logits, -1), step["goal_idx"][:, None], -1)[:, 0]
    ent_g = -(jax.nn.softmax(goal_logits, -1) * jax.nn.log_softmax(goal_logits, -1)).sum(-1).mean()
    if controller == "greedy":
        mlp = jnp.zeros(n)                       # greedy move is deterministic -> no move PG action
        ent_m = 0.0
    else:
        h, w = obs.shape[-2], obs.shape[-1]
        goal = goal_targets(pos, stencil, h, w)[jnp.arange(n), step["goal_idx"]]
        mlogits = actor.move_logits(pos, goal, step["blocked"])
        mlp = jnp.take_along_axis(jax.nn.log_softmax(mlogits, -1), step["move"][:, None], -1)[:, 0]
        ent_m = -(jax.nn.softmax(mlogits, -1) * jax.nn.log_softmax(mlogits, -1)).sum(-1).mean()
    return (glp + mlp).sum(), value.mean(), ent_g + ent_m            # (), (), ()


def _ppo_loss(actor, batch, adv, ret, old_logp, stencil, clip, vf, ent_c, key, controller="mvprop"):
    B, T = adv.shape
    ks = jax.random.split(key, B * T).reshape(B, T, 2)
    # vmap only the per-(B,T) fields the loss recompute needs — NOT the whole traj (v_last is
    # scalar-per-rollout (B,), which the inner T-vmap can't map).
    steps = {k: batch[k] for k in ("obs", "adj", "pos", "goal_idx", "move", "blocked")}
    new_logp, vteam, ent = jax.vmap(jax.vmap(
        lambda s, k: _step_logp_val_ent(actor, s, stencil, k, controller)))(steps, ks)   # (B,T) each
    ratio = jnp.exp(new_logp - old_logp)
    pg = -jnp.minimum(ratio * adv, jnp.clip(ratio, 1 - clip, 1 + clip) * adv).mean()
    vloss = vf * ((vteam - ret) ** 2).mean()
    ent_loss = -ent_c * ent.mean()
    return pg + vloss + ent_loss, dict(pg=pg, vloss=vloss, ent=ent.mean())


def train_step(env, actor, opt, opt_state, cfg, key, stencil, controller="mvprop"):
    """One PPO iteration: 16 rollouts -> GAE -> ppo_epochs x minibatches clipped-PG."""
    t = cfg.trainer
    B = int(cfg.rollouts_per_iter)                           # 16, never reduced
    rk, tk = jax.random.split(key)
    seeds = jax.random.split(rk, B)
    traj = jax.vmap(lambda s: rollout(env, actor, cfg, s, stencil, controller))(seeds)   # (B,T,...)
    adv, ret = _advantages(traj, t.gamma, t.gae_lambda)
    adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    old_logp = (traj["goal_logp"] + traj["move_logp"]).sum(-1)               # (B,T) joint over agents

    grad_fn = eqx.filter_value_and_grad(_ppo_loss, has_aux=True)
    logs = {}
    for _ep in range(int(t.ppo_epochs)):
        tk, pk, lk = jax.random.split(tk, 3)
        for mb in jnp.array_split(jax.random.permutation(pk, B), int(t.minibatches)):
            sub = jax.tree_util.tree_map(lambda x: x[mb], traj)
            (loss, logs), grad = grad_fn(
                actor, sub, adv[mb], ret[mb], old_logp[mb], stencil,
                t.clip, cfg.loss.vf_coef, cfg.regularization.entropy_coef, lk, controller)
            params = eqx.filter(actor, eqx.is_array)
            updates, opt_state = opt.update(grad, opt_state, params)
            actor = eqx.apply_updates(actor, updates)
    metrics = dict(ret=ret.mean(), **logs)
    return actor, opt_state, metrics


def train(env, actor, cfg, key, controller="mvprop"):
    """Full loop over cfg.iters. Returns (actor, history)."""
    opt = make_optimizer(cfg)
    opt_state = opt.init(eqx.filter(actor, eqx.is_array))
    stencil = make_stencil(cfg)
    hist = []
    k = key
    step = eqx.filter_jit(train_step)
    for it in range(int(cfg.iters)):
        k, sk = jax.random.split(k)
        actor, opt_state, m = step(env, actor, opt, opt_state, cfg, sk, stencil, controller)
        row = {kk: float(v) for kk, v in m.items()}
        hist.append(row)
        if it % 10 == 0 or it == int(cfg.iters) - 1:
            print(f"[it {it}] ret={row['ret']:.2f} pg={row['pg']:.3f} "
                  f"vloss={row['vloss']:.1f} ent={row['ent']:.2f}", flush=True)
    return actor, hist
