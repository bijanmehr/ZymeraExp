"""MAPPO-CTDE trainer for the grounded v0 agent.

PPO optimizes the **GOAL policy** (the L3 goal head), NOT raw moves. Each step:

  backbone(obs, kb_adj) -> belief z_i -> goal_logits (N,K)
    └─ [mechanism] action_mask: mask candidate goals whose greedy first move
       would drop true λ₂ below the floor (forbid-disconnect); soft_lambda: no
       mask, a λ-penalty enters the reward instead.
    └─ sample goal index g_i ~ masked-softmax(goal_logits)  (PPO action)
    └─ goal cell = pos + stencil[g_i]
    └─ L1 greedy controller -> env move (only valid moves, STAY fallback)
    └─ env.step(move) -> reward terms, true λ₂, coverage

GAE runs on the centralized critic (team reward + team value = CTDE). Total loss:

  total = PPO(goal) + vf_coef*value + aux_beta*aux(λ̂₂, λ₂_true)
          + degree_reg*Var_batch(mean-degree) - entropy_coef*H(goal)

The aux head is supervised against the simulator's true λ₂ (mse | huber knob);
the degree regularizer (SizeShiftReg-style) penalizes the across-batch variance
of the per-node aggregated degree statistic to protect GNN size-transfer.

All JAX/Equinox: ``eqx.filter_jit`` rollouts + update; ``optax`` AdamW
(decoupled weight_decay) + global-norm clip. CPU-friendly (``JAX_PLATFORMS=cpu``).

Shapes (T horizon, B rollouts, N agents, K candidate goals):
  obs       (B,T,N,C,H,W)   central (B,T,Cg,H,W)
  goal      (B,T,N)          goal index sampled (PPO action)
  goal_logp (B,T,N)          masked log-prob of the sampled goal
  goal_mask (B,T,N,K)        the safe-goal mask used at sample time (replayed)
  rew_agent (B,T,N)          composed reward    rew_team (B,T) mean-over-agents
  v_team    (B,T)            centralized critic value
  true_l2   (B,T)            true Fiedler value (aux target, broadcast to agents)
  l2_hat    (B,T,N)          per-agent local λ̂₂ estimate (head output)
  degree    (B,T,N)          per-node comm degree (degree regularizer input)
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from . import controller as ctrl
from . import env_utils as _eu
from .config import CTDEConfig
from .nets import Actor, Critic

EPS = 1e-8
_NEG = -1e9   # masked-goal logit


# =============================================================================
# Goal selection + mechanism (per step, pure)
# =============================================================================


def _goal_mask(env, state, cfg: CTDEConfig, stencil):
    """(N,K) bool safe-goal mask for the active mission-safety mechanism.

    action_mask -> :func:`controller.safe_goal_mask` (forbid-disconnect, with a
    guaranteed safe 'here' candidate). soft_lambda / anything else -> all True
    (no masking; the penalty lives in the reward).
    """
    ms = cfg.mission_safety
    pos = state.body.position
    h, w = state.wall.shape
    if ms.mechanism == "action_mask":
        goal_cells = ctrl.goal_targets(pos, stencil, h, w)             # (N,K,2)
        valid_targets = env.dynamics.targets(state)                    # (N,A,2)
        action_valid = env.action_mask(state)                         # (N,A)
        return ctrl.safe_goal_mask(
            pos, goal_cells, valid_targets, action_valid,
            cfg.world.comm_r, cfg.connectivity.lambda2_sharp, ms.min_lambda2,
        )
    # soft_lambda (or any non-masking mechanism): nothing forbidden.
    n, K = pos.shape[0], stencil.shape[0]
    return jnp.ones((n, K), dtype=bool)


ROLE_EXPLORER = 0   # role index: explorer = the frontier/goal behaviour
ROLE_RELAY = 1      # role index: relay = the local λ̂₂-anchor (hold the bridge)


def _goal_to_move(env, state, goal_idx, stencil, role_idx=None, cfg: CTDEConfig = None):
    """((N,) move, (N,) bool was-valid) from chosen goals (+optional roles) via the
    L1 controllers. ``was_valid[i]`` audits the emitted move is env-valid.

    ``role_idx`` None -> every agent runs the explorer (greedy-toward-goal) move
    (v0 behaviour, default-unchanged). Otherwise role 0 (explorer) keeps the greedy
    goal move and role 1 (relay) takes the local-connectivity-anchor move
    (:func:`controller.relay_move`); both go through valid-moves-only controllers,
    selected per agent by role.
    """
    pos = state.body.position
    h, w = state.wall.shape
    goal_cells = ctrl.goal_targets(pos, stencil, h, w)                 # (N,K,2)
    n = pos.shape[0]
    goal = goal_cells[jnp.arange(n), goal_idx]                        # (N,2)
    valid_targets = env.dynamics.targets(state)                       # (N,A,2)
    action_valid = env.action_mask(state)                            # (N,A)
    expl_move = ctrl.greedy_move(pos, goal, valid_targets, action_valid)   # (N,)
    if role_idx is not None:
        relay = ctrl.relay_move(pos, valid_targets, action_valid,
                                cfg.world.comm_r, cfg.connectivity.lambda2_sharp)  # (N,)
        move = jnp.where(role_idx == ROLE_RELAY, relay, expl_move)    # (N,) per-role
    else:
        move = expl_move
    was_valid = jnp.take_along_axis(action_valid, move[:, None], axis=1)[:, 0]
    return move, was_valid


# =============================================================================
# Rollout (one batched scan; vmap over B seeds)
# =============================================================================


def _single_rollout(env, actor, critic, cfg: CTDEConfig, stencil, key):
    """Collect ONE episode under the current (actor, critic). Pure (vmap over key)."""
    reset_key, scan_key = jax.random.split(key)
    obs0, state0 = env.reset(reset_key)

    use_roles = cfg.role_picker == "expl_relay"

    def body(carry, _):
        state, obs, k = carry
        k, ak, rk, sk = jax.random.split(k, 4)

        adj_off = _eu.kb_adjacency(state.body.position, cfg)          # (N,N) KB graph
        goal_logits, role_logits, value_agent, l2_hat, _z = actor(
            obs, adj_off, inference=True
        )
        central = env.central_obs(state)                              # (Cg,H,W)
        v_team = critic(central, inference=True)                      # ()

        gmask = _goal_mask(env, state, cfg, stencil)                  # (N,K) bool
        masked_logits = jnp.where(gmask, goal_logits, _NEG)
        goal = jax.random.categorical(ak, masked_logits, axis=-1)     # (N,)
        logp = jnp.take_along_axis(
            jax.nn.log_softmax(masked_logits, axis=-1), goal[:, None], axis=-1
        )[:, 0]                                                       # (N,)

        # role picker (L3): sample a role per agent when enabled, else all-explorer.
        n = state.n_agents
        if use_roles:
            role = jax.random.categorical(rk, role_logits, axis=-1)   # (N,)
            role_logp = jnp.take_along_axis(
                jax.nn.log_softmax(role_logits, axis=-1), role[:, None], axis=-1
            )[:, 0]                                                   # (N,)
            role_idx = role
        else:
            role = jnp.zeros((n,), jnp.int32)                         # explorer
            role_logp = jnp.zeros((n,), jnp.float32)
            role_idx = None                                          # v0 routing

        move, move_valid = _goal_to_move(env, state, goal, stencil, role_idx, cfg)
        obs_next, state_next, _rew, done, info = env.step(state, move, sk)

        # soft-λ penalty (shared scalar shortfall below the floor); 0 unless soft.
        l2_true = _eu.true_lambda2(state_next.body.position, cfg)     # ()
        if cfg.mission_safety.mechanism == "soft_lambda":
            short = jax.nn.relu(cfg.mission_safety.min_lambda2 - l2_true)
            l2_penalty = jnp.broadcast_to(short, (n,))
        else:
            l2_penalty = None

        rew_agent = _eu.compose_reward(info["reward_terms"], state_next, cfg, l2_penalty)
        rew_team = rew_agent.mean()                                   # () centralized target
        cov = _eu.coverage_fraction_free(state_next, cfg)            # ()
        degree = _eu.degree_stats(state.body.position, cfg)         # (N,)

        per_step = {
            "obs": obs, "adj": adj_off, "central": central,
            "goal": goal, "goal_logp": logp, "goal_mask": gmask,
            "role": role, "role_logp": role_logp,
            "v_team": v_team, "l2_hat": l2_hat,
            "rew_agent": rew_agent, "rew_team": rew_team,
            "true_l2": l2_true, "coverage": cov, "degree": degree,
            "move": move, "move_valid": move_valid,
            "done": done.any().astype(jnp.float32),
        }
        return (state_next, obs_next, k), per_step

    (state_T, _obs_T, _), traj = jax.lax.scan(
        body, (state0, obs0, scan_key), xs=None, length=cfg.world.horizon
    )
    central_T = env.central_obs(state_T)
    traj["v_last"] = critic(central_T, inference=True)               # () GAE bootstrap
    return traj


def collect(env, actor, critic, cfg: CTDEConfig, stencil, key):
    """Vmap ``_single_rollout`` over B seeds -> batched trajectory (leading B,T)."""
    keys = jax.random.split(key, cfg.rollouts_per_iter)
    return jax.vmap(lambda k: _single_rollout(env, actor, critic, cfg, stencil, k))(keys)


# =============================================================================
# GAE (centralized critic; team reward + team value)
# =============================================================================


def _gae(rew, val, v_last, gamma, lam):
    def step(carry, x):
        gae, next_v = carry
        r, v = x
        delta = r + gamma * next_v - v
        gae = delta + gamma * lam * gae
        return (gae, v), gae

    (_, _), adv = jax.lax.scan(step, (jnp.zeros(()), v_last), (rew, val), reverse=True)
    return adv, adv + val


def compute_advantages(traj, cfg: CTDEConfig):
    t = cfg.trainer
    adv, ret = jax.vmap(lambda r, v, vl: _gae(r, v, vl, t.gamma, t.gae_lambda))(
        traj["rew_team"], traj["v_team"], traj["v_last"]
    )
    return adv, ret


# =============================================================================
# Loss
# =============================================================================


def _flatten_BT(x):
    return x.reshape((-1,) + x.shape[2:])


def _aux_loss(l2_hat, true_l2, cfg: CTDEConfig):
    """Per-agent aux loss vs the true λ₂ (broadcast). mse | huber knob."""
    err = l2_hat - true_l2[:, None]                                  # (M,N)
    if cfg.loss.aux_loss == "huber":
        d = cfg.loss.huber_delta
        a = jnp.abs(err)
        return jnp.mean(jnp.where(a <= d, 0.5 * err ** 2, d * (a - 0.5 * d)))
    return jnp.mean(err ** 2)                                        # mse (default)


def _clipped_pg(logp, old_logp, adv_norm, clip):
    """Clipped PPO surrogate for a (M,N) log-prob head against (M,1) adv."""
    ratio = jnp.exp(logp - old_logp)
    unclipped = ratio * adv_norm
    clipped = jnp.clip(ratio, 1.0 - clip, 1.0 + clip) * adv_norm
    return -jnp.minimum(unclipped, clipped).mean()


def loss_fn(actor, critic, batch, cfg: CTDEConfig, key):
    """Total loss = PPO(goal) [+ PPO(role)] + vf*value + beta*aux + degreeReg
    - ent*(goal entropy [+ role entropy]). The role terms are added ONLY when
    ``role_picker == 'expl_relay'`` (off -> identical to v0)."""
    obs = batch["obs"]                 # (M,N,C,H,W)
    adj = batch["adj"]                 # (M,N,N) KB graph
    central = batch["central"]         # (M,Cg,H,W)
    goal = batch["goal"]               # (M,N) sampled goal index
    old_logp = batch["goal_logp"]      # (M,N)
    gmask = batch["goal_mask"]         # (M,N,K)
    role = batch["role"]               # (M,N) sampled role index
    old_role_logp = batch["role_logp"] # (M,N)
    adv = batch["adv"]                 # (M,) team advantage
    ret = batch["ret"]                 # (M,) team return
    true_l2 = batch["true_l2"]         # (M,) aux target
    degree = batch["degree"]           # (M,N) per-node comm degree

    use_roles = cfg.role_picker == "expl_relay"
    M = obs.shape[0]
    akeys = jax.random.split(key, M)
    # actor forward over the minibatch (dropout active in training).
    def fwd(o, a, kk):
        return actor(o, a, key=kk, inference=False)
    goal_logits, role_logits, _v_agent, l2_hat, _z = jax.vmap(fwd)(obs, adj, akeys)
    # goal_logits (M,N,K); apply the SAME mask used at sample time.
    masked = jnp.where(gmask, goal_logits, _NEG)
    logp_all = jax.nn.log_softmax(masked, axis=-1)                  # (M,N,K)
    logp = jnp.take_along_axis(logp_all, goal[..., None], axis=-1)[..., 0]  # (M,N)
    probs = jnp.exp(logp_all)
    entropy = -(jnp.where(gmask, probs * logp_all, 0.0)).sum(-1)    # (M,N)

    adv_b = jax.lax.stop_gradient(adv)[:, None]                     # (M,1)
    adv_norm = (adv_b - adv_b.mean()) / (adv_b.std() + EPS)
    clip = cfg.loss.ppo_clip
    goal_pg = _clipped_pg(logp, old_logp, adv_norm, clip)

    # role policy head (shares the team advantage; trained only when enabled).
    if use_roles:
        role_logp_all = jax.nn.log_softmax(role_logits, axis=-1)    # (M,N,R)
        role_logp = jnp.take_along_axis(
            role_logp_all, role[..., None], axis=-1)[..., 0]        # (M,N)
        role_probs = jnp.exp(role_logp_all)
        role_entropy = -(role_probs * role_logp_all).sum(-1)        # (M,N)
        role_pg = _clipped_pg(role_logp, old_role_logp, adv_norm, clip)
        policy_loss = goal_pg + role_pg
        role_ent = role_entropy.mean()
    else:
        policy_loss = goal_pg
        role_pg = jnp.zeros(())
        role_ent = jnp.zeros(())

    v_pred = jax.vmap(lambda c: critic(c, inference=False))(central)  # (M,)
    value_loss = jnp.mean((v_pred - jax.lax.stop_gradient(ret)) ** 2)

    aux_loss = _aux_loss(l2_hat, true_l2, cfg)

    # SizeShiftReg-style degree regularizer: variance across the batch of the
    # per-sample mean comm-degree (penalize drift of local structure).
    mean_deg = degree.mean(-1)                                      # (M,)
    degree_reg = jnp.var(mean_deg)

    ent = entropy.mean()
    reg = cfg.regularization
    total = (
        policy_loss
        + cfg.loss.vf_coef * value_loss
        + cfg.loss.aux_beta * aux_loss
        + reg.degree_reg * degree_reg
        - reg.entropy_coef * (ent + role_ent)
    )
    metrics = {
        "policy_loss": policy_loss, "goal_pg": goal_pg, "role_pg": role_pg,
        "value_loss": value_loss, "aux_loss": aux_loss,
        "entropy": ent, "role_entropy": role_ent, "degree_reg": degree_reg,
    }
    return total, metrics


# =============================================================================
# Trainer
# =============================================================================


class TrainState(eqx.Module):
    actor: Actor
    critic: Critic
    opt_state: optax.OptState


def make_optimizer(cfg: CTDEConfig):
    t = cfg.trainer
    return optax.chain(
        optax.clip_by_global_norm(t.max_grad_norm),
        optax.adamw(t.lr, weight_decay=cfg.regularization.weight_decay),
    )


def make_stencil(cfg: CTDEConfig):
    return ctrl.goal_stencil(cfg.action_head.K, cfg.action_head.stride)


def init_state(env, cfg: CTDEConfig, key) -> TrainState:
    ka, kc = jax.random.split(key)
    in_ch = env.obs.obs_channels
    cg = env.obs.central_channels
    actor = Actor(in_ch, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=cfg.regularization.dropout, key=ka)
    critic = Critic(cg, cfg.backbone.width, cfg.backbone.depth, cfg.backbone.norm,
                    cfg.regularization.dropout, key=kc)
    opt = make_optimizer(cfg)
    params = (eqx.filter(actor, eqx.is_array), eqx.filter(critic, eqx.is_array))
    return TrainState(actor=actor, critic=critic, opt_state=opt.init(params))


def _update_epoch(carry, flat, perm, key, cfg: CTDEConfig, opt):
    """One PPO epoch over ``flat`` (M,...), split into ``minibatches`` chunks.

    ``carry`` = ``(actor, critic, opt_state)``; ``key`` seeds per-minibatch
    dropout-loss keys (one ``split`` per minibatch index, so it's deterministic
    under ``lax.scan``)."""
    M = perm.shape[0]
    nmb = cfg.trainer.minibatches
    mb = M // nmb
    mb_keys = jax.random.split(key, nmb)

    def one_minibatch(c, i):
        actor, critic, opt_state = c
        lk = mb_keys[i]
        idx = jax.lax.dynamic_slice_in_dim(perm, i * mb, mb)
        batch = {kk: v[idx] for kk, v in flat.items()}

        def lf(modules):
            a, cc = modules
            return loss_fn(a, cc, batch, cfg, lk)

        (_loss, metrics), grads = eqx.filter_value_and_grad(lf, has_aux=True)(
            (actor, critic)
        )
        params = (eqx.filter(actor, eqx.is_array), eqx.filter(critic, eqx.is_array))
        updates, opt_state = opt.update(grads, opt_state, params)
        actor, critic = eqx.apply_updates((actor, critic), updates)
        return (actor, critic, opt_state), metrics

    carry, metrics = jax.lax.scan(one_minibatch, carry, jnp.arange(nmb))
    return carry, metrics


def train_step(env, state: TrainState, cfg: CTDEConfig, key, opt, stencil):
    """One PPO iteration: collect -> GAE -> ppo_epochs of minibatch updates."""
    ck, pk = jax.random.split(key)
    traj = collect(env, state.actor, state.critic, cfg, stencil, ck)
    adv, ret = compute_advantages(traj, cfg)

    # The KB adjacency the actor consumed at rollout time is stored in the traj
    # ("adj"), so the loss replays the actor on exactly the same comm graph.
    flat = {
        "obs": _flatten_BT(traj["obs"]),
        "adj": _flatten_BT(traj["adj"]),
        "central": _flatten_BT(traj["central"]),
        "goal": _flatten_BT(traj["goal"]),
        "goal_logp": _flatten_BT(traj["goal_logp"]),
        "goal_mask": _flatten_BT(traj["goal_mask"]),
        "role": _flatten_BT(traj["role"]),
        "role_logp": _flatten_BT(traj["role_logp"]),
        "adv": _flatten_BT(adv),
        "ret": _flatten_BT(ret),
        "true_l2": _flatten_BT(traj["true_l2"]),
        "degree": _flatten_BT(traj["degree"]),
    }
    M = flat["obs"].shape[0]

    def one_epoch(carry, ek):
        pkey, lkey = jax.random.split(ek)
        perm = jax.random.permutation(pkey, M)
        carry, metrics = _update_epoch(carry, flat, perm, lkey, cfg, opt)
        return carry, metrics

    epoch_keys = jax.random.split(pk, cfg.trainer.ppo_epochs)
    carry0 = (state.actor, state.critic, state.opt_state)
    carry, epoch_metrics = jax.lax.scan(one_epoch, carry0, epoch_keys)
    actor, critic, opt_state = carry
    state = TrainState(actor=actor, critic=critic, opt_state=opt_state)
    last_metrics = jax.tree_util.tree_map(lambda x: x[-1], epoch_metrics)

    # ---- iteration diagnostics (on the freshly collected traj) ----
    ep_reward = traj["rew_team"].sum(axis=1).mean()
    coverage_pct = traj["coverage"][:, -1].mean()
    connectivity_pct = (traj["true_l2"] > cfg.connectivity.threshold).mean()
    mean_lambda2 = traj["true_l2"].mean()

    # aux λ₂ accuracy = 1 - median rel-err (l2_hat vs true_l2) over CONNECTED steps
    l2_true_bt = traj["true_l2"]                       # (B,T)
    l2_hat_bt = traj["l2_hat"].mean(axis=-1)           # (B,T) mean over agents
    connected = l2_true_bt > cfg.connectivity.threshold
    rel = jnp.abs(l2_hat_bt - l2_true_bt) / jnp.maximum(jnp.abs(l2_true_bt), EPS)
    rel_connected = jnp.where(connected, rel, jnp.nan)
    median_rel = jnp.nanmedian(rel_connected)
    aux_acc = jnp.clip(1.0 - median_rel, 0.0, 1.0)

    # controller validity: fraction of emitted moves that were in the env action
    # mask (the greedy controller's guarantee; should be exactly 1.0).
    valid_frac = traj["move_valid"].astype(jnp.float32).mean()

    # role split: fraction of agent-steps assigned EXPLORER vs RELAY. When
    # role_picker is off every agent is an explorer (frac=1.0 by construction).
    explorer_frac = (traj["role"] == ROLE_EXPLORER).astype(jnp.float32).mean()

    logs = {
        "ep_reward": ep_reward,
        "coverage_pct": coverage_pct,
        "connectivity_pct": connectivity_pct,
        "mean_lambda2": mean_lambda2,
        "aux_loss": jnp.mean(last_metrics["aux_loss"]),
        "aux_acc": aux_acc,
        "median_rel_l2": median_rel,
        "policy_loss": jnp.mean(last_metrics["policy_loss"]),
        "value_loss": jnp.mean(last_metrics["value_loss"]),
        "entropy": jnp.mean(last_metrics["entropy"]),
        "role_entropy": jnp.mean(last_metrics["role_entropy"]),
        "degree_reg": jnp.mean(last_metrics["degree_reg"]),
        "ctrl_valid_frac": valid_frac,
        "explorer_frac": explorer_frac,
        "relay_frac": 1.0 - explorer_frac,
    }
    return state, logs


def train(env, cfg: CTDEConfig, *, key=None, log_fn=None):
    """Full training loop over ``cfg.iters`` PPO iterations.

    ``log_fn(it, host_logs)`` is called each iteration. Returns (TrainState, history)."""
    if key is None:
        key = jax.random.PRNGKey(cfg.seed)
    opt = make_optimizer(cfg)
    stencil = make_stencil(cfg)
    state = init_state(env, cfg, key)

    @eqx.filter_jit
    def jitted_step(state, k):
        return train_step(env, state, cfg, k, opt, stencil)

    history = []
    k = key
    for it in range(cfg.iters):
        k, sk = jax.random.split(k)
        state, logs = jitted_step(state, sk)
        host_logs = {kk: float(np.asarray(v)) for kk, v in logs.items()}
        history.append(host_logs)
        if log_fn is not None:
            log_fn(it, host_logs)
    return state, history
