"""Smoke tests for the grounded CTDE v0 agent (LPAC + goal head + aux-λ₂).

CPU-only by construction. Run:
    JAX_PLATFORMS=cpu \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m pytest \
        ctde_v0/tests/test_smoke.py -q
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_HERE = os.path.dirname(os.path.abspath(__file__))
_PKG_PARENT = os.path.dirname(os.path.dirname(_HERE))  # .../SharedExploration
if _PKG_PARENT not in sys.path:
    sys.path.insert(0, _PKG_PARENT)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from ctde_v0 import controller as ctrl  # noqa: E402
from ctde_v0 import env_utils, ppo  # noqa: E402
from ctde_v0.config import (  # noqa: E402
    Backbone, CTDEConfig, MissionSafety, Trainer, World, from_dict,
)
from ctde_v0.nets import Actor, Critic  # noqa: E402


def _tiny_cfg(**over) -> CTDEConfig:
    """A tiny, fast config: 8x8 / 3 agents / 6-step horizon."""
    base = CTDEConfig(
        world=World(grid=8, n_agents=3, comm_r=3, horizon=6),
        backbone=Backbone(width=16, depth=2, mp_rounds=2),
        trainer=Trainer(minibatches=2, ppo_epochs=2),
        iters=1, rollouts_per_iter=2, seed=0,
    )
    if over:
        import dataclasses
        base = dataclasses.replace(base, **over)
    return base


# ---- config schema ----------------------------------------------------------

def test_config_roundtrip():
    cfg = _tiny_cfg()
    d = cfg.to_dict()
    assert d["backbone"]["agg"] == "max"
    assert d["action_head"]["kind"] == "goal_pointer"
    assert d["mission_safety"]["mechanism"] == "action_mask"
    # I1c: the new conn_signal axis round-trips and defaults to the I1b behaviour.
    assert d["mission_safety"]["conn_signal"] == "global_lambda2"
    assert d["mission_safety"]["degree_target"] == 1.0
    cfg2 = from_dict(d)
    assert cfg2.to_dict() == d


def test_conn_signal_axis_roundtrip():
    """I1c: a non-default conn_signal + degree_target round-trips through the tree."""
    cfg = _tiny_cfg(mission_safety=MissionSafety(
        mechanism="lagrangian", conn_signal="local_edge_margin", degree_target=2.0))
    d = cfg.to_dict()
    assert d["mission_safety"]["conn_signal"] == "local_edge_margin"
    assert d["mission_safety"]["degree_target"] == 2.0
    assert from_dict(d).to_dict() == d


# ---- env / shapes -----------------------------------------------------------

def test_env_shapes():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    assert env.n_agents == cfg.world.n_agents
    assert env.n_actions == 5
    obs, state = env.reset(jax.random.PRNGKey(0))
    assert obs.shape == (cfg.world.n_agents, env.obs.obs_channels, 8, 8)
    assert env.central_obs(state).shape == (env.obs.central_channels, 8, 8)
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    assert adj.shape == (cfg.world.n_agents, cfg.world.n_agents)
    assert not bool(jnp.diag(adj).any())  # diagonal cleared


# ---- backbone + heads -------------------------------------------------------

def test_actor_backbone_forward():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K,
                  backbone_cfg=cfg.backbone, dropout=0.0, key=jax.random.PRNGKey(2))
    goal_logits, role_logits, value, l2_hat, z = actor(obs, adj, inference=True)
    assert goal_logits.shape == (cfg.world.n_agents, cfg.action_head.K)
    assert role_logits.shape == (cfg.world.n_agents, actor.n_roles)
    assert value.shape == (cfg.world.n_agents,)
    assert l2_hat.shape == (cfg.world.n_agents,)
    assert z.shape == (cfg.world.n_agents, cfg.backbone.width)


def test_aggregators_all_run():
    """mean / max / multihead must all produce finite beliefs."""
    cfg0 = _tiny_cfg()
    env = env_utils.build_env(cfg0)
    obs, state = env.reset(jax.random.PRNGKey(7))
    adj = env_utils.kb_adjacency(state.body.position, cfg0)
    import dataclasses
    for agg in ("mean", "max", "multihead"):
        bb = dataclasses.replace(cfg0.backbone, agg=agg)
        actor = Actor(env.obs.obs_channels, cfg0.action_head.K,
                      backbone_cfg=bb, dropout=0.0, key=jax.random.PRNGKey(3))
        _, _, _, _, z = actor(obs, adj, inference=True)
        assert jnp.isfinite(z).all(), agg


# ---- controller: only valid moves ------------------------------------------

def test_controller_emits_only_valid_moves():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(4))
    stencil = ppo.make_stencil(cfg)
    pos = state.body.position
    h, w = state.wall.shape
    goal_cells = ctrl.goal_targets(pos, stencil, h, w)
    valid_targets = env.dynamics.targets(state)
    action_valid = env.action_mask(state)
    # try every candidate as the chosen goal for every agent
    n, K = pos.shape
    for k in range(K):
        goal = goal_cells[:, k]
        move = ctrl.greedy_move(pos, goal, valid_targets, action_valid)
        ok = jnp.take_along_axis(action_valid, move[:, None], axis=1)[:, 0]
        assert bool(ok.all()), (k, move)


def test_collision_mask_never_blocks_stay_and_stays_valid():
    """I1b: the hard collision mask never flags STAY, and a forbid_collision
    greedy move is still always env-valid (every candidate goal, every agent)."""
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(11))
    stencil = ppo.make_stencil(cfg)
    pos = state.body.position
    h, w = state.wall.shape
    goal_cells = ctrl.goal_targets(pos, stencil, h, w)
    valid_targets = env.dynamics.targets(state)
    action_valid = env.action_mask(state)
    blocked = ctrl.occupied_cell_mask(pos, valid_targets)            # (N,A) bool
    assert blocked.shape == action_valid.shape
    assert not bool(blocked[:, int(ctrl.ActionId.STAY)].any())       # STAY never blocked
    n, K = pos.shape
    for k in range(K):
        move = ctrl.greedy_move(pos, goal_cells[:, k], valid_targets, action_valid,
                                forbid_collision=True)
        ok = jnp.take_along_axis(action_valid, move[:, None], axis=1)[:, 0]
        assert bool(ok.all()), (k, move)


def test_collision_mask_forbids_occupied_cells():
    """I1b: two ADJACENT agents — each one's step ONTO the other's current cell is
    flagged (and ONLY that move), STAY never is. Self-consistent synthetic layout:
    targets are built the SAME way the env builds them (pos + ACTION_DELTAS), so
    ``valid_targets`` matches ``pos``."""
    from zymera.env import ACTION_DELTAS, ActionId
    # agent0 at (2,2); agent1 east neighbour at (2,3); agent2 far away.
    pos = jnp.array([[2, 2], [2, 3], [7, 7]], dtype=jnp.int32)
    deltas = jnp.asarray(ACTION_DELTAS, dtype=jnp.int32)            # (A,2)
    valid_targets = pos[:, None, :] + deltas[None, :, :]            # (N,A,2)
    east, west = 2, 4                                               # ACTION_DELTAS order
    blocked = ctrl.occupied_cell_mask(pos, valid_targets)          # (N,A)
    assert bool(blocked[0, east])      # agent0 stepping EAST lands on agent1's cell
    assert bool(blocked[1, west])      # agent1 stepping WEST lands on agent0's cell
    assert not bool(blocked[:, int(ActionId.STAY)].any())          # STAY never blocked
    assert not bool(blocked[2].any())  # the far agent has nothing blocked
    # exactly the two colliding moves are flagged (no over-masking).
    assert int(blocked.sum()) == 2


def test_safe_goal_mask_keeps_a_candidate():
    cfg = _tiny_cfg(mission_safety=MissionSafety(mechanism="action_mask",
                                                 min_lambda2=0.5))
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(8))
    stencil = ppo.make_stencil(cfg)
    pos = state.body.position
    h, w = state.wall.shape
    goal_cells = ctrl.goal_targets(pos, stencil, h, w)
    safe = ctrl.safe_goal_mask(
        pos, goal_cells, env.dynamics.targets(state), env.action_mask(state),
        cfg.world.comm_r, cfg.connectivity.lambda2_sharp, cfg.mission_safety.min_lambda2,
    )
    assert safe.shape == (pos.shape[0], stencil.shape[0])
    assert bool(safe.any(-1).all())          # every agent has >=1 safe candidate
    assert bool(safe[:, 0].all())            # 'here' is always safe


# ---- reward + λ₂ ------------------------------------------------------------

def test_local_edge_margin_edge_vs_interior():
    """I1c: ``local_edge_margin`` is a PER-AGENT 'you're at the edge of comms range'
    signal — an agent whose nearest link sits at ≈ comm_r gets a clearly POSITIVE
    margin while an agent comfortably surrounded by in-range teammates gets ≈0.

    Hand-built 3-agent layout (comm_r=5, sharp=2, degree_target=1.0):
      a0(0,0), a1(0,1) form a tight interior pair (dist 1);
      a2(0,6) is the EDGE agent — its nearest neighbour (a1) is exactly at comm_r=5
      and a0 is just past range (dist 6), so its soft-degree falls below the target.
    a1 is the clean interior agent (two in-range neighbours -> soft-degree >> 1)."""
    cfg = _tiny_cfg(world=World(grid=10, n_agents=3, comm_r=5, horizon=6))
    assert cfg.world.comm_r == 5 and cfg.connectivity.lambda2_sharp == 2.0
    assert cfg.mission_safety.degree_target == 1.0
    pos = jnp.array([[0, 0], [0, 1], [0, 6]], dtype=jnp.int32)
    p = env_utils.local_edge_margin(pos, cfg)
    assert p.shape == (3,) and p.dtype == jnp.float32
    assert bool((p >= 0).all())                       # a relu shortfall, never negative
    assert float(p[2]) > 0.1, p                        # EDGE agent: clearly penalized
    assert float(p[1]) < 1e-4, p                       # INTERIOR agent: ≈0
    # the edge agent is the MOST penalized of the three (it is the one stretching).
    assert int(jnp.argmax(p)) == 2, p
    # sanity vs the relay tool's soft degree it is built from: p_i == relu(target-deg).
    deg = ctrl._local_conn_score(pos, cfg.world.comm_r, cfg.connectivity.lambda2_sharp)
    expected = jax.nn.relu(cfg.mission_safety.degree_target - deg)
    assert bool(jnp.allclose(p, expected)), (p, expected)


def test_local_edge_margin_interior_all_zero():
    """I1c: with every agent packed well inside comm range (soft-degree >> target),
    the per-agent margin is ≈0 everywhere (no spurious penalty in the interior)."""
    cfg = _tiny_cfg(world=World(grid=10, n_agents=3, comm_r=5, horizon=6))
    pos = jnp.array([[2, 2], [2, 3], [3, 2]], dtype=jnp.int32)   # all mutually adjacent
    p = env_utils.local_edge_margin(pos, cfg)
    assert bool((p < 1e-4).all()), p


def test_reward_and_lambda2():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(3))
    move = jnp.zeros((cfg.world.n_agents,), jnp.int32)
    _, state2, _, _, info = env.step(state, move, jax.random.PRNGKey(4))
    rew = env_utils.compose_reward(info["reward_terms"], state2, cfg)
    assert rew.shape == (cfg.world.n_agents,)
    l2 = env_utils.true_lambda2(state2.body.position, cfg)
    assert l2.shape == () and float(l2) >= 0.0


# ---- connectivity-FLOOR barrier ("Hyper-Singularity") -----------------------

def _barrier_cfg(**reward_over):
    """A tiny cfg whose Reward block carries explicit barrier knobs (a=2, M=5)."""
    from ctde_v0.config import Reward
    base = dict(barrier_weight=1.0, barrier_a=2.0, barrier_M=5.0,
                barrier_p=2.0, barrier_cap=50.0)
    base.update(reward_over)
    return _tiny_cfg(world=World(grid=10, n_agents=2, comm_r=5, horizon=6),
                     reward=Reward(**base))


def test_barrier_shape_zero_rise_cap_and_finite():
    """(a) SHAPE: f==0 for x<=a, strictly increasing for a<x<M, saturates at cap as
    x->M, f==cap for x>=M (incl. x==M exactly AND x==M+1), and NO nan/inf anywhere."""
    cfg = _barrier_cfg()                                   # a=2, M=5, cap=50
    a, M, cap = cfg.barrier_a, cfg.barrier_M, cfg.reward.barrier_cap
    assert (a, M, cap) == (2.0, 5.0, 50.0)

    # sweep x over [0, M+1] on a fine grid (single-agent nearest-nbr distance).
    xs = jnp.linspace(0.0, M + 1.0, 64)

    def f_of(x):                                           # pure single-agent eval
        pos = jnp.array([[0.0, 0.0], [0.0, x]], dtype=jnp.float32)
        return env_utils.connectivity_barrier(pos, cfg)[0]
    fs = jax.vmap(f_of)(xs)

    assert bool(jnp.isfinite(fs).all()), fs                # no nan/inf for ANY x
    assert bool((fs >= 0.0).all()) and bool((fs <= cap).all())  # in [0, cap]
    # silent below the launch point.
    below = xs <= a
    assert bool((fs[below] == 0.0).all()), fs[below]
    # non-decreasing across the whole sweep (monotone wall, never dips).
    assert bool((jnp.diff(fs) >= -1e-5).all()), fs
    # STRICTLY increasing on the rising sub-cap portion of (a, M): take the samples in
    # (a, M) that have NOT yet saturated to the ceiling — they must be strictly rising.
    mid = (xs > a) & (xs < M) & (fs < cap - 1e-3)
    fmid = fs[mid]
    assert fmid.shape[0] >= 3
    assert bool((jnp.diff(fmid) > 0).all()), fmid          # strictly rising pre-cap
    # saturates toward cap as x -> M (just inside the wall is already at the ceiling).
    assert float(f_of(M - 0.05)) > 0.9 * cap
    # at/after the wall -> exactly cap (x == M, x == M+1, and a far isolate).
    assert float(f_of(M)) == cap
    assert float(f_of(M + 1.0)) == cap
    iso = jnp.array([[0.0, 0.0], [0.0, M + 50.0]], dtype=jnp.float32)
    assert float(env_utils.connectivity_barrier(iso, cfg)[0]) == cap


def test_barrier_per_agent_layout():
    """(b) PER-AGENT: a hand-built layout — an agent with a neighbour inside a -> 0;
    an agent whose nearest neighbour is just inside M -> large (near cap); an isolated
    agent (nearest > M) -> exactly cap. Each agent reads its OWN nearest-nbr distance.

    Layout (Chebyshev, a=2, M=5, cap=50): a0(0,0)&a1(0,1) are a tight pair (dist 1 < a
    -> both 0); a2(0,5.9) sits a hair inside the wall — its NEAREST neighbour is a1 at
    dist 4.9, in (a,M) and right against M -> the pole pushes it to the ceiling; a3 is
    far away (isolated, nearest > M -> exactly cap). Float positions let 'just inside M'
    be unambiguous while the metric stays the same Chebyshev the comm graph uses."""
    cfg = _barrier_cfg()                                   # a=2, M=5, cap=50
    cap = cfg.reward.barrier_cap
    # a0,a1 tight pair; a2 a hair inside the wall (nearest = a1 at dist 4.9); a3 isolated.
    pos = jnp.array([[0.0, 0.0], [0.0, 1.0], [0.0, 5.9], [0.0, 50.0]], dtype=jnp.float32)
    f = env_utils.connectivity_barrier(pos, cfg)
    assert f.shape == (4,) and f.dtype == jnp.float32
    assert bool(jnp.isfinite(f).all())
    # a0 & a1: nearest neighbour (each other) at dist 1 < a -> exactly 0 (safe interior).
    assert float(f[0]) == 0.0 and float(f[1]) == 0.0
    # a2: nearest at dist 4.9, just inside (a=2, M=5) and against the wall -> ~cap.
    assert 0.0 < float(f[2]) <= cap and float(f[2]) > 0.5 * cap, f
    # a3: nearest neighbour (a2 at dist ~45) > M -> exactly the ceiling.
    assert float(f[3]) == cap, f
    # the isolated agent is (tied-)most penalized; the near-wall agent far above the pair.
    assert float(f[3]) >= float(f[2]) and float(f[2]) > float(f[0])


def test_barrier_default_off_is_byte_unchanged():
    """(c) REGRESSION: barrier_weight==0 (the DEFAULT) -> compose_reward is byte-
    identical to the pre-barrier reward (the term contributes EXACTLY nothing — the
    no-op gate lives in compose_reward, which skips the subtraction entirely)."""
    cfg = _tiny_cfg()                                      # default Reward: barrier_weight 0
    assert cfg.reward.barrier_weight == 0.0
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(3))
    move = jnp.zeros((cfg.world.n_agents,), jnp.int32)
    _, state2, _, _, info = env.step(state, move, jax.random.PRNGKey(4))

    # reference reward WITHOUT the barrier term (re-derive the pre-barrier sum exactly).
    r = cfg.reward
    rt = info["reward_terms"]
    ref = (r.w_coverage * rt["coverage"]
           + r.w_connectivity * rt["connectivity"]
           + r.w_collision * rt["collision"]).astype(jnp.float32)
    got = env_utils.compose_reward(rt, state2, cfg)
    assert bool(jnp.array_equal(got, ref)), (got, ref)     # byte-identical with weight 0
    # the function never returns nan/inf even at the default (k=0) on a live layout.
    f0 = env_utils.connectivity_barrier(state2.body.position, cfg)
    assert bool(jnp.isfinite(f0).all()), f0
    # with k=0 the SMOOTH region contributes exactly 0 (the cap-at-wall is unconditional,
    # but it never reaches the reward because compose_reward skips the term at weight 0).
    pos_safe = jnp.array([[0, 0], [0, 1]], dtype=jnp.int32)   # tight pair, x=1 < a
    assert bool((env_utils.connectivity_barrier(pos_safe, cfg) == 0.0).all())


def test_barrier_config_roundtrip():
    """The barrier knobs round-trip through to_dict/from_dict and the comm_r-derived
    defaults resolve (None -> comm_r*0.6 for a, comm_r for M)."""
    from ctde_v0.config import Reward
    # defaults: a/M unset -> resolved off comm_r (=5 here -> a=3.0, M=5.0).
    cfg = _tiny_cfg(world=World(grid=10, n_agents=2, comm_r=5, horizon=6))
    assert cfg.reward.barrier_a is None and cfg.reward.barrier_M is None
    assert cfg.barrier_a == 3.0 and cfg.barrier_M == 5.0
    # explicit knobs survive a JSON-shaped round trip.
    cfg2 = _tiny_cfg(reward=Reward(barrier_weight=2.5, barrier_a=1.0, barrier_M=4.0,
                                   barrier_p=3.0, barrier_cap=25.0))
    d = cfg2.to_dict()
    assert d["reward"]["barrier_weight"] == 2.5
    assert d["reward"]["barrier_a"] == 1.0 and d["reward"]["barrier_M"] == 4.0
    assert d["reward"]["barrier_p"] == 3.0 and d["reward"]["barrier_cap"] == 25.0
    assert from_dict(d).to_dict() == d


def test_barrier_composes_in_rollout():
    """The barrier COMPOSES with a mechanism in a live rollout: weight>0 runs a full
    PPO iteration (finite reward, controller still 100% valid) — it is an ADDITIVE term,
    not a replacement for the mission-safety mechanism."""
    cfg = _tiny_cfg(reward=_barrier_cfg().reward,          # barrier on (weight 1, a2/M5)
                    mission_safety=MissionSafety(mechanism="soft_lambda"))
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    _, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"])
    assert float(logs["ctrl_valid_frac"]) == 1.0


# ---- one full PPO iteration -------------------------------------------------

def test_one_train_step_runs():
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    new_state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    for k in ("ep_reward", "coverage_pct", "connectivity_pct", "mean_lambda2",
              "aux_loss", "aux_acc", "ctrl_valid_frac", "degree_reg"):
        assert k in logs and jnp.isfinite(logs[k]), (k, logs.get(k))
    # the controller guarantee: 100% valid moves in the rollout.
    assert float(logs["ctrl_valid_frac"]) == 1.0
    import equinox as eqx
    p0 = jax.tree_util.tree_leaves(eqx.filter(state.actor, eqx.is_array))
    p1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(p0, p1)), "actor unchanged"


def test_soft_lambda_mechanism_runs():
    cfg = _tiny_cfg(mission_safety=MissionSafety(mechanism="soft_lambda"))
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    _, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"])
    assert float(logs["ctrl_valid_frac"]) == 1.0


def test_collision_mask_train_step_runs():
    """I1b: collision_mask='on' runs a PPO iteration and the controller still emits
    only env-valid moves over the whole rollout (the STAY-always-valid guarantee)."""
    import dataclasses
    cfg = dataclasses.replace(_tiny_cfg(), collision_mask="on")
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    _, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"])
    assert float(logs["ctrl_valid_frac"]) == 1.0          # only valid moves emitted


def _violating_threshold(cfg, env, stencil, key):
    """A constraint floor τ guaranteed to make v = relu(τ − mean λ₂) > 0 on the
    tiny env (measure the rollout's mean λ₂ and sit comfortably above it), so the
    dual mechanisms see a real violation deterministically (no flake)."""
    st = ppo.init_state(env, cfg, key)
    traj = ppo.collect(env, st.actor, st.critic, cfg, stencil, key, jnp.float32(0.0))
    return float(traj["true_l2"].mean()) + 0.5


def test_lagrangian_mechanism_moves_lambda():
    """I1b: the 'lagrangian' dual ascends λ from its init on a real violation, and
    the dual variable is carried functionally in the train state across iters."""
    base = _tiny_cfg()
    env = env_utils.build_env(base)
    opt = ppo.make_optimizer(base)
    stencil = ppo.make_stencil(base)
    tau = _violating_threshold(base, env, stencil, jax.random.PRNGKey(0))
    cfg = _tiny_cfg(mission_safety=MissionSafety(
        mechanism="lagrangian", constraint_threshold=tau, lambda_lr=0.05,
        lambda_init=0.0))
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    assert float(state.dual.lam) == 0.0                    # starts at lambda_init
    lams = [float(state.dual.lam)]
    k = jax.random.PRNGKey(6)
    for _ in range(3):
        k, sk = jax.random.split(k)
        state, logs = ppo.train_step(env, state, cfg, sk, opt, stencil)
        lams.append(float(state.dual.lam))
        assert jnp.isfinite(logs["dual_lambda"])
        assert float(logs["ctrl_valid_frac"]) == 1.0
    assert max(lams) > lams[0], lams                       # λ changed from its init
    assert float(state.dual.lam) >= 0.0                    # dual stays non-negative


def test_lagrangian_local_edge_margin_moves_lambda():
    """I1c: (mechanism=lagrangian × conn_signal=local_edge_margin) runs a few PPO
    iters on 10×10/2 and the dual λ MOVES from its init. With 2 agents and
    degree_target=2.0 the per-agent soft-degree (≤1 neighbour each) can never reach
    the target, so the aggregate margin violation v=mean_i p_i is positive every
    step and the dual ascends deterministically (no flake)."""
    cfg = _tiny_cfg(
        world=World(grid=10, n_agents=2, comm_r=5, horizon=6),
        mission_safety=MissionSafety(
            mechanism="lagrangian", conn_signal="local_edge_margin",
            degree_target=2.0, lambda_lr=0.05, lambda_init=0.0),
    )
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    assert float(state.dual.lam) == 0.0                    # starts at lambda_init
    lams = [float(state.dual.lam)]
    k = jax.random.PRNGKey(6)
    for _ in range(3):
        k, sk = jax.random.split(k)
        state, logs = ppo.train_step(env, state, cfg, sk, opt, stencil)
        lams.append(float(state.dual.lam))
        assert jnp.isfinite(logs["dual_lambda"])
        assert float(logs["dual_violation"]) > 0.0         # local margin really fires
        assert float(logs["ctrl_valid_frac"]) == 1.0
    assert max(lams) > lams[0], lams                       # λ moved from its init


def test_pid_lagrangian_mechanism_updates_lambda():
    """I1b: the 'pid_lagrangian' dual updates λ via the PID controller and carries
    the integral / prev-error in the train state (both evolve across iters)."""
    base = _tiny_cfg()
    env = env_utils.build_env(base)
    opt = ppo.make_optimizer(base)
    stencil = ppo.make_stencil(base)
    tau = _violating_threshold(base, env, stencil, jax.random.PRNGKey(0))
    cfg = _tiny_cfg(mission_safety=MissionSafety(
        mechanism="pid_lagrangian", constraint_threshold=tau,
        pid_kp=1.0, pid_ki=0.01, pid_kd=0.1, lambda_init=0.0))
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    assert float(state.dual.integral) == 0.0               # PID state starts empty
    lams, integ = [float(state.dual.lam)], [float(state.dual.integral)]
    k = jax.random.PRNGKey(6)
    for _ in range(3):
        k, sk = jax.random.split(k)
        state, logs = ppo.train_step(env, state, cfg, sk, opt, stencil)
        lams.append(float(state.dual.lam))
        integ.append(float(state.dual.integral))
        assert jnp.isfinite(logs["dual_lambda"])
        assert float(logs["ctrl_valid_frac"]) == 1.0
    assert max(lams) > 0.0, lams                           # λ took a non-zero value
    assert integ[-1] > integ[0], integ                     # PID integral advanced


def test_adaptive_mechanisms_threaded_through_train():
    """I1b: ``ppo.train`` carries the dual through the jitted loop and logs λ each
    iteration for both adaptive mechanisms (history is self-describing)."""
    for mech in ("lagrangian", "pid_lagrangian"):
        base = _tiny_cfg()
        env = env_utils.build_env(base)
        stencil = ppo.make_stencil(base)
        tau = _violating_threshold(base, env, stencil, jax.random.PRNGKey(0))
        cfg = _tiny_cfg(iters=2, mission_safety=MissionSafety(
            mechanism=mech, constraint_threshold=tau))
        _, history = ppo.train(env, cfg)
        assert len(history) == 2
        assert all("dual_lambda" in h and jnp.isfinite(h["dual_lambda"])
                   for h in history), mech


def test_global_lambda2_default_is_byte_unchanged():
    """I1c regression guard: conn_signal='global_lambda2' (the DEFAULT) reproduces
    the I1b rollout EXACTLY. For each penalty mechanism the rollout under an explicit
    global_lambda2 and under the default MissionSafety produce bit-identical reward,
    moves and true λ₂ — and the global path's trajectory carries NO 'margin_step'
    key (the local-only diagnostic), so the global trace pytree is untouched."""
    import dataclasses
    for mech in ("soft_lambda", "lagrangian"):
        default_ms = MissionSafety(mechanism=mech)              # conn_signal defaults
        assert default_ms.conn_signal == "global_lambda2"
        explicit_ms = dataclasses.replace(default_ms, conn_signal="global_lambda2")
        cfg_def = _tiny_cfg(mission_safety=default_ms)
        cfg_exp = _tiny_cfg(mission_safety=explicit_ms)
        env = env_utils.build_env(cfg_def)
        stencil = ppo.make_stencil(cfg_def)
        key = jax.random.PRNGKey(6)
        lam0 = jnp.float32(0.3)                                 # a non-trivial dual λ
        st = ppo.init_state(env, cfg_def, jax.random.PRNGKey(5))
        t_def = ppo.collect(env, st.actor, st.critic, cfg_def, stencil, key, lam0)
        t_exp = ppo.collect(env, st.actor, st.critic, cfg_exp, stencil, key, lam0)
        for k in ("rew_agent", "move", "true_l2"):
            assert bool(jnp.array_equal(t_def[k], t_exp[k])), (mech, k)
        assert "margin_step" not in t_def, mech               # global trace untouched


def test_action_mask_and_soft_lambda_unchanged_by_dual():
    """I1b regression guard: the dual variable is INERT for action_mask and
    soft_lambda — λ never leaves its init and dual_update returns it unchanged,
    so the published I1 mechanisms are byte-unchanged in behaviour."""
    for mech in ("action_mask", "soft_lambda"):
        cfg = _tiny_cfg(mission_safety=MissionSafety(mechanism=mech))
        env = env_utils.build_env(cfg)
        opt = ppo.make_optimizer(cfg)
        stencil = ppo.make_stencil(cfg)
        state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
        for _ in range(2):
            state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6),
                                         opt, stencil)
            assert float(state.dual.lam) == cfg.mission_safety.lambda_init, mech
            assert float(logs["dual_lambda"]) == cfg.mission_safety.lambda_init, mech


def test_short_train_loop():
    cfg = _tiny_cfg(iters=2)
    env = env_utils.build_env(cfg)
    _, history = ppo.train(env, cfg)
    assert len(history) == 2
    assert all(jnp.isfinite(h["aux_loss"]) for h in history)


if __name__ == "__main__":
    test_config_roundtrip()
    test_conn_signal_axis_roundtrip()
    test_env_shapes()
    test_actor_backbone_forward()
    test_aggregators_all_run()
    test_controller_emits_only_valid_moves()
    test_collision_mask_never_blocks_stay_and_stays_valid()
    test_collision_mask_forbids_occupied_cells()
    test_safe_goal_mask_keeps_a_candidate()
    test_local_edge_margin_edge_vs_interior()
    test_local_edge_margin_interior_all_zero()
    test_reward_and_lambda2()
    test_barrier_shape_zero_rise_cap_and_finite()
    test_barrier_per_agent_layout()
    test_barrier_default_off_is_byte_unchanged()
    test_barrier_config_roundtrip()
    test_barrier_composes_in_rollout()
    test_one_train_step_runs()
    test_soft_lambda_mechanism_runs()
    test_collision_mask_train_step_runs()
    test_lagrangian_mechanism_moves_lambda()
    test_lagrangian_local_edge_margin_moves_lambda()
    test_pid_lagrangian_mechanism_updates_lambda()
    test_adaptive_mechanisms_threaded_through_train()
    test_global_lambda2_default_is_byte_unchanged()
    test_action_mask_and_soft_lambda_unchanged_by_dual()
    test_short_train_loop()
    print("all smoke tests passed")
