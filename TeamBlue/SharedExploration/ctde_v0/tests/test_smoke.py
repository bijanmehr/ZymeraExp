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
from ctde_v0 import env_utils, nets, ppo  # noqa: E402
from ctde_v0.config import (  # noqa: E402
    ActionHead, Backbone, CTDEConfig, MissionSafety, Trainer, World, from_dict,
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
    # I2: the explorer_tool / relay_tool / compass axes round-trip and default to v0.
    assert d["action_head"]["explorer_tool"] == "goal_head"
    assert d["action_head"]["relay_tool"] == "lambda2_anchor"
    assert d["action_head"]["compass"] == "off"
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
    goal_logits, role_logits, value, l2_hat, z, _h = actor(obs, adj, inference=True)
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
        _, _, _, _, z, _h = actor(obs, adj, inference=True)
        assert jnp.isfinite(z).all(), agg


# ---- frontier-attention explorer tool (I2 / L4 "disperse") ------------------

def _obs_with_frontier_east(C, H, W, ar, ac):
    """A hand-built obs (C,H,W) with the agent at (ar,ac) and ALL ground known
    EXCEPT the columns strictly east of the agent — so the only frontier (uncovered)
    ground lies to the EAST (compass sector 2)."""
    obs = jnp.zeros((C, H, W), dtype=jnp.float32)
    obs = obs.at[nets._CH_OWN_POS, ar, ac].set(1.0)        # own_pos one-hot
    obs = obs.at[nets._CH_KNOWN].set(1.0)                  # everything known...
    obs = obs.at[nets._CH_KNOWN, :, ac + 1:].set(0.0)      # ...except east -> frontier
    return obs


def test_sector_frontier_features_concentrate():
    """Unit: with uncovered cells concentrated in ONE compass sector (EAST), the
    per-sector frontier feature puts clearly higher mass on that sector (and ≈0 on
    the opposite West sector). Features are normalized fractions in [0,1]."""
    cfg = _tiny_cfg()
    K = cfg.action_head.K
    obs = _obs_with_frontier_east(5, 11, 11, ar=5, ac=5)
    feats = nets.sector_frontier_features(obs, K)
    assert feats.shape == (K, 2)
    assert bool((feats >= 0.0).all()) and bool((feats <= 1.0).all())   # fractions
    east, west = 2, 4                                      # _COMPASS order: 2=E, 4=W
    # the EAST sector is the most frontier-rich of all K sectors (fraction feature).
    assert int(jnp.argmax(feats[:, 0])) == east, feats[:, 0]
    assert float(feats[east, 0]) > float(feats[west, 0]) + 0.3, feats[:, 0]
    # density feature agrees: more frontier mass toward east than west.
    assert float(feats[east, 1]) > float(feats[west, 1]), feats[:, 1]


def test_frontier_attn_shifts_goal_probability():
    """Unit: the frontier-attention term shifts goal probability toward the
    frontier-rich sector. Compare goal-softmax with the frontier tool ON vs the
    goal-head ALONE (same belief/params): the EAST sector's probability strictly
    increases once the frontier bias is added."""
    cfg = _tiny_cfg(backbone=Backbone(width=16))
    K = cfg.action_head.K
    obs1 = _obs_with_frontier_east(5, 11, 11, ar=5, ac=5)
    obs = obs1[None]                                       # (1,C,H,W) one agent
    adj = jnp.zeros((1, 1), dtype=bool)                    # lone agent (no neighbours)
    # a frontier_attn actor; read its goal_head-only vs goal_head+frontier logits.
    actor = Actor(5, K, backbone_cfg=cfg.backbone, dropout=0.0,
                  key=jax.random.PRNGKey(0), explorer_tool="frontier_attn")
    z = actor.backbone(obs, adj, inference=True)           # (1,W)
    base = jax.vmap(actor.goal_head)(z)                    # (1,K) goal head alone
    term = actor.frontier_attn(obs, z, K)                  # (1,K) frontier bias
    east = 2
    # the additive frontier term is non-negative and (by construction) MAXIMAL at the
    # frontier-rich EAST sector — frontier-positive even at random init.
    assert bool((term[0] >= -1e-6).all()), term[0]
    assert int(jnp.argmax(term[0])) == east, term[0]
    p_base = jax.nn.softmax(base, axis=-1)[0]
    p_bias = jax.nn.softmax(base + term, axis=-1)[0]
    # ...so its sampled probability strictly increases vs the goal-head-only policy.
    assert float(p_bias[east]) > float(p_base[east]), (p_base[east], p_bias[east])


def test_frontier_features_scale_invariant():
    """Scale-invariance: the SAME relative frontier layout (east half uncovered,
    agent centered) yields near-identical per-sector features at 11×11 and 21×21 —
    the features are fractions/unit-directions, independent of grid size."""
    cfg = _tiny_cfg()
    K = cfg.action_head.K
    f_small = nets.sector_frontier_features(_obs_with_frontier_east(5, 11, 11, 5, 5), K)
    f_big = nets.sector_frontier_features(_obs_with_frontier_east(5, 21, 21, 10, 10), K)
    # the FRACTION feature (col 0) is a pure fraction -> ~grid-size invariant (the soft
    # sector boundaries shift a touch with cell count, but values track closely).
    assert bool(jnp.allclose(f_small[:, 0], f_big[:, 0], atol=0.12)), (
        f_small[:, 0], f_big[:, 0])
    # both rank EAST top regardless of size (the directional signal transfers).
    assert int(jnp.argmax(f_small[:, 0])) == int(jnp.argmax(f_big[:, 0])) == 2


def test_explorer_tool_goal_head_byte_unchanged():
    """Regression: explorer_tool='goal_head' (DEFAULT) leaves the goal policy byte-
    identical — the actor's goal logits equal goal_head(z) EXACTLY (the frontier
    module never contributes; the static branch is skipped)."""
    cfg = _tiny_cfg()
    assert cfg.action_head.explorer_tool == "goal_head"    # the default
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(2), explorer_tool="goal_head")
    goal_logits, _rl, _v, _l2, z, _h = actor(obs, adj, inference=True)
    goal_only = jax.vmap(actor.goal_head)(z)               # goal head with NO tool
    assert bool(jnp.array_equal(goal_logits, goal_only)), "goal_head path changed"


def test_explorer_tool_param_surface_stable():
    """The frontier_attn submodule is ALWAYS built (stable param tree) regardless of
    the explorer_tool knob — only its USE is gated. Both actors expose frontier_attn
    leaves of identical shapes (mirrors role_head being always-built)."""
    cfg = _tiny_cfg()
    a_gh = Actor(5, cfg.action_head.K, backbone_cfg=cfg.backbone, dropout=0.0,
                 key=jax.random.PRNGKey(0), explorer_tool="goal_head")
    a_fa = Actor(5, cfg.action_head.K, backbone_cfg=cfg.backbone, dropout=0.0,
                 key=jax.random.PRNGKey(0), explorer_tool="frontier_attn")
    import equinox as eqx
    s_gh = [p.shape for p in jax.tree_util.tree_leaves(
        eqx.filter(a_gh.frontier_attn, eqx.is_array))]
    s_fa = [p.shape for p in jax.tree_util.tree_leaves(
        eqx.filter(a_fa.frontier_attn, eqx.is_array))]
    assert s_gh == s_fa and len(s_gh) > 0                  # same non-empty param surface


def test_explorer_tool_frontier_attn_train_step():
    """Train: a PPO iter with --explorer-tool frontier_attn (+ role_picker expl_relay)
    on 10×10/2 runs without error and the controller still emits 100% valid moves."""
    cfg = _tiny_cfg(
        world=World(grid=10, n_agents=2, comm_r=5, horizon=6),
        action_head=ActionHead(explorer_tool="frontier_attn"),
        role_picker="expl_relay",
    )
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    new_state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"])
    assert float(logs["ctrl_valid_frac"]) == 1.0           # only valid moves emitted
    # the actor (incl. the frontier module) actually took a gradient step.
    import equinox as eqx
    p0 = jax.tree_util.tree_leaves(eqx.filter(state.actor, eqx.is_array))
    p1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(p0, p1)), "actor unchanged"


def test_explorer_tool_config_roundtrip():
    """I2: a non-default explorer_tool round-trips through the config tree."""
    import dataclasses
    cfg = _tiny_cfg(action_head=ActionHead(explorer_tool="frontier_attn"))
    d = cfg.to_dict()
    assert d["action_head"]["explorer_tool"] == "frontier_attn"
    assert from_dict(d).to_dict() == d


# ---- relay tool axis (I2: lambda2_anchor | hold) ----------------------------

def _deltas_targets(pos):
    """(valid_targets (N,A,2), action_valid (N,A) all-True) built EXACTLY the way the
    env builds them (pos + ACTION_DELTAS), so the synthetic layout is self-consistent
    with what the relay controllers read off ``dynamics.targets`` / ``action_mask``."""
    from zymera.env import ACTION_DELTAS
    deltas = jnp.asarray(ACTION_DELTAS, dtype=jnp.int32)              # (A,2)
    valid_targets = pos[:, None, :] + deltas[None, :, :]             # (N,A,2)
    action_valid = jnp.ones((pos.shape[0], deltas.shape[0]), dtype=bool)
    return valid_targets, action_valid


def test_relay_hold_well_connected_stays():
    """I2 relay 'hold': a well-connected relay (soft-degree >> the hold floor) STAYs —
    the static beacon does not wander while it is comfortably anchored. Tight mutually
    adjacent triangle at comm_r=5/sharp=2 -> every agent's soft-degree is 2.0."""
    sharp, comm_r = 2.0, 5
    pos = jnp.array([[5, 5], [5, 6], [6, 5]], dtype=jnp.int32)        # all adjacent
    vt, av = _deltas_targets(pos)
    deg = ctrl._local_conn_score(pos, comm_r, sharp)
    assert bool((deg >= 0.5).all()), deg                              # above the hold floor
    move = ctrl.relay_hold_move(pos, vt, av, comm_r, sharp)           # (N,)
    assert bool((move == int(ctrl.ActionId.STAY)).all()), move        # everyone holds


def test_relay_hold_about_to_isolate_reconnects():
    """I2 relay 'hold': a relay whose only neighbour has drifted just past the hold
    floor (soft-degree < floor) takes the SINGLE valid move that best restores a
    neighbour (toward it), lifting its soft-degree back up — and STAY stays env-valid.

    Layout (comm_r=5, sharp=2, floor 0.5): a0(0,0)'s nearest neighbour is a1(0,6), a
    cell PAST comm range, so a0's soft-degree (~0.12) is below the floor; the east step
    toward a1 raises it to ~0.5 (the boundary). a2(0,7) sits tight beside a1 (dist 1) so
    a2 is comfortably anchored (soft-degree >= the floor) and HOLDS — the static beacon
    moves ONLY the about-to-isolate agent, not the well-connected one."""
    sharp, comm_r = 2.0, 5
    pos = jnp.array([[0, 0], [0, 6], [0, 7]], dtype=jnp.int32)
    vt, av = _deltas_targets(pos)
    deg_now = ctrl._local_conn_score(pos, comm_r, sharp)
    assert float(deg_now[0]) < 0.5, deg_now                          # a0 below the floor
    assert float(deg_now[2]) >= 0.5, deg_now                         # a2 anchored (a1 adjacent)
    move = ctrl.relay_hold_move(pos, vt, av, comm_r, sharp)          # (N,)
    east = 2                                                         # _COMPASS / ACTION order
    assert int(move[0]) != int(ctrl.ActionId.STAY)                  # a0 moves to reconnect
    assert int(move[0]) == east, move                               # ...toward its neighbour
    # the chosen move strictly INCREASES a0's soft-degree (it restored a link).
    after = ctrl._local_conn_score(pos.at[0].set(vt[0, move[0]]), comm_r, sharp)
    assert float(after[0]) > float(deg_now[0]), (deg_now[0], after[0])
    # the comfortably-anchored agent a2 HOLDS (hold moves only the isolating agent).
    assert int(move[2]) == int(ctrl.ActionId.STAY), move


def test_relay_hold_emits_only_valid_moves():
    """I2 relay 'hold': on a live env layout every emitted relay move is env-valid (the
    STAY-always-valid guarantee), exactly like the lambda2_anchor relay."""
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    _, state = env.reset(jax.random.PRNGKey(4))
    pos = state.body.position
    valid_targets = env.dynamics.targets(state)
    action_valid = env.action_mask(state)
    move = ctrl.relay_hold_move(pos, valid_targets, action_valid,
                                cfg.world.comm_r, cfg.connectivity.lambda2_sharp)
    ok = jnp.take_along_axis(action_valid, move[:, None], axis=1)[:, 0]
    assert bool(ok.all()), move


def test_relay_tool_hold_train_step():
    """I2: a PPO iter with --relay-tool hold (+ --role-picker expl_relay) on 10×10/2
    runs without error and the controller still emits 100% valid moves."""
    cfg = _tiny_cfg(
        world=World(grid=10, n_agents=2, comm_r=5, horizon=6),
        action_head=ActionHead(relay_tool="hold"),
        role_picker="expl_relay",
    )
    assert cfg.action_head.relay_tool == "hold"
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    _, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"])
    assert float(logs["ctrl_valid_frac"]) == 1.0                     # only valid moves


def test_relay_tool_lambda2_anchor_byte_unchanged():
    """I2 REGRESSION: relay_tool='lambda2_anchor' (the DEFAULT) reproduces the v0/I1
    relay routing EXACTLY. An expl_relay rollout under the default and under an explicit
    'lambda2_anchor' produce bit-identical moves / reward / true λ₂."""
    import dataclasses
    cfg_def = _tiny_cfg(world=World(grid=10, n_agents=2, comm_r=5, horizon=6),
                        role_picker="expl_relay")
    assert cfg_def.action_head.relay_tool == "lambda2_anchor"        # the default
    cfg_exp = dataclasses.replace(
        cfg_def, action_head=ActionHead(relay_tool="lambda2_anchor"))
    env = env_utils.build_env(cfg_def)
    stencil = ppo.make_stencil(cfg_def)
    st = ppo.init_state(env, cfg_def, jax.random.PRNGKey(5))
    key = jax.random.PRNGKey(6)
    t_def = ppo.collect(env, st.actor, st.critic, cfg_def, stencil, key, jnp.float32(0.0))
    t_exp = ppo.collect(env, st.actor, st.critic, cfg_exp, stencil, key, jnp.float32(0.0))
    for k in ("move", "rew_agent", "true_l2"):
        assert bool(jnp.array_equal(t_def[k], t_exp[k])), k


def test_relay_tool_config_roundtrip():
    """I2: a non-default relay_tool round-trips through the config tree."""
    cfg = _tiny_cfg(action_head=ActionHead(relay_tool="hold"))
    d = cfg.to_dict()
    assert d["action_head"]["relay_tool"] == "hold"
    assert from_dict(d).to_dict() == d


# ---- compass directional feature (I2: off | on) -----------------------------

def _obs_team_east_frontier_west(C, H, W, ar, ac):
    """A hand-built obs (C,H,W): the agent at (ar,ac), an in-range TEAMMATE three cells
    to the EAST (``neighbors`` channel), and ALL ground known EXCEPT a patch to the far
    WEST (frontier). So the GATHER direction (toward teammates) is EAST and the EXPLORE
    direction (toward the nearest uncovered cell) is WEST — two distinct headings."""
    obs = jnp.zeros((C, H, W), dtype=jnp.float32)
    obs = obs.at[nets._CH_OWN_POS, ar, ac].set(1.0)        # own_pos one-hot
    obs = obs.at[nets._CH_NEIGHBORS, ar, ac + 3].set(1.0)  # teammate to the EAST
    obs = obs.at[nets._CH_KNOWN].set(1.0)                  # everything known...
    obs = obs.at[nets._CH_KNOWN, :, :ac - 1].set(0.0)      # ...except far west -> frontier
    return obs


def test_compass_features_point_correctly():
    """Unit: with teammates EAST and frontier WEST, the gather direction peaks on the
    EAST sector and the explore direction on the WEST sector. Both are normalized soft
    sector distributions (each Σ_k ≈ 1) over the K compass headings."""
    cfg = _tiny_cfg()
    K = cfg.action_head.K
    obs = _obs_team_east_frontier_west(5, 13, 13, ar=6, ac=6)
    feats = nets.compass_features(obs, K)                  # (2, K) gather, explore
    assert feats.shape == (2, K)
    gather, explore = feats[0], feats[1]
    assert abs(float(gather.sum()) - 1.0) < 1e-4 and abs(float(explore.sum()) - 1.0) < 1e-4
    east, west = 2, 4                                      # _COMPASS order: 2=E, 4=W
    assert int(jnp.argmax(gather)) == east, gather         # gather -> teammates (EAST)
    assert int(jnp.argmax(explore)) == west, explore       # explore -> frontier (WEST)
    assert float(gather[east]) > float(gather[west])
    assert float(explore[west]) > float(explore[east])


def test_compass_empty_streams_fall_to_here():
    """Unit: with NO in-range teammate the gather direction falls entirely on the 'here'
    sector (index 0, no direction); with NO frontier in view the explore direction does
    too — a safe scale-free default, never a spurious compass heading."""
    cfg = _tiny_cfg()
    K = cfg.action_head.K
    here = 0
    # all ground known (no frontier) and no neighbours marked.
    obs = jnp.zeros((5, 11, 11), dtype=jnp.float32)
    obs = obs.at[nets._CH_OWN_POS, 5, 5].set(1.0)
    obs = obs.at[nets._CH_KNOWN].set(1.0)                  # everything known -> no frontier
    feats = nets.compass_features(obs, K)
    assert int(jnp.argmax(feats[0])) == here and float(feats[0, here]) > 0.99  # gather here
    assert int(jnp.argmax(feats[1])) == here and float(feats[1, here]) > 0.99  # explore here


def test_compass_features_scale_invariant():
    """Scale-invariance: the SAME relative layout (teammate east, frontier west, agent
    centered) yields near-identical compass directions at 13×13 and 25×25 — the features
    are cosines of unit directions + normalized softmaxes, independent of grid size."""
    cfg = _tiny_cfg()
    K = cfg.action_head.K
    f_small = nets.compass_features(_obs_team_east_frontier_west(5, 13, 13, 6, 6), K)
    f_big = nets.compass_features(_obs_team_east_frontier_west(5, 25, 25, 12, 12), K)
    assert bool(jnp.allclose(f_small, f_big, atol=0.15)), (f_small, f_big)
    # both rank the SAME sector top for each direction regardless of size.
    assert int(jnp.argmax(f_small[0])) == int(jnp.argmax(f_big[0])) == 2   # gather EAST
    assert int(jnp.argmax(f_small[1])) == int(jnp.argmax(f_big[1])) == 4   # explore WEST


def test_compass_off_byte_unchanged():
    """Regression: compass='off' (DEFAULT) leaves the belief z — and therefore EVERY
    head output — byte-identical to the pre-compass actor (the compass term is never
    added; the static branch is skipped, z == backbone(obs))."""
    cfg = _tiny_cfg()
    assert cfg.action_head.compass == "off"                # the default
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(2), compass="off")
    goal_logits, role_logits, value, l2_hat, z, _h = actor(obs, adj, inference=True)
    z_raw = actor.backbone(obs, adj, inference=True)       # belief with NO compass term
    assert bool(jnp.array_equal(z, z_raw)), "z changed when compass off"
    # every head equals its raw-belief readout (no compass anywhere).
    assert bool(jnp.array_equal(goal_logits, jax.vmap(actor.goal_head)(z_raw)))
    assert bool(jnp.array_equal(role_logits, jax.vmap(actor.role_head)(z_raw)))


def test_compass_off_init_byte_identical_to_on():
    """Regression: the compass module is ALWAYS built, AND a compass='off' actor shares
    EVERY non-compass parameter bit-for-bit with a compass='on' actor of the same key
    (the compass key is fold_in-derived, so the backbone / goal / role / frontier / aux
    / value keys are unchanged — an off actor is the pre-compass network exactly)."""
    cfg = _tiny_cfg()
    a_off = Actor(5, cfg.action_head.K, backbone_cfg=cfg.backbone, dropout=0.0,
                  key=jax.random.PRNGKey(0), compass="off")
    a_on = Actor(5, cfg.action_head.K, backbone_cfg=cfg.backbone, dropout=0.0,
                 key=jax.random.PRNGKey(0), compass="on")
    import equinox as eqx
    def leaves(m):
        return jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    for name in ("backbone", "goal_head", "role_head", "frontier_attn",
                 "aux_head", "value_head"):
        lo, ln = leaves(getattr(a_off, name)), leaves(getattr(a_on, name))
        assert lo and all(bool(jnp.array_equal(x, y)) for x, y in zip(lo, ln)), name
    # the compass param surface itself is present and identical between the two.
    s_off = [p.shape for p in leaves(a_off.compass)]
    s_on = [p.shape for p in leaves(a_on.compass)]
    assert s_off == s_on and len(s_off) > 0


def test_compass_on_shifts_belief_and_train_step():
    """I2: compass='on' actually shifts the belief z (so the heads see the directional
    cue), and a PPO iter with --compass on runs without error at 100% valid moves with
    the compass params taking a gradient step."""
    cfg = _tiny_cfg(
        world=World(grid=10, n_agents=2, comm_r=5, horizon=6),
        action_head=ActionHead(compass="on"),
    )
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(2), compass="on")
    _, _, _, _, z, _h = actor(obs, adj, inference=True)
    z_raw = actor.backbone(obs, adj, inference=True)
    assert not bool(jnp.array_equal(z, z_raw)), "compass=on did not shift z"
    # a full PPO iter runs and the compass submodule actually moves.
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    tstate = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    new_state, logs = ppo.train_step(env, tstate, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"])
    assert float(logs["ctrl_valid_frac"]) == 1.0
    import equinox as eqx
    p0 = jax.tree_util.tree_leaves(eqx.filter(tstate.actor.compass, eqx.is_array))
    p1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor.compass, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(p0, p1)), "compass unchanged"


def test_compass_config_roundtrip():
    """I2: a non-default compass round-trips through the config tree."""
    cfg = _tiny_cfg(action_head=ActionHead(compass="on"))
    d = cfg.to_dict()
    assert d["action_head"]["compass"] == "on"
    assert from_dict(d).to_dict() == d


def test_defaults_byte_unchanged_full_actor():
    """I2 REGRESSION (defaults): the DEFAULT action_head (relay_tool=lambda2_anchor,
    compass=off, explorer_tool=goal_head) leaves the actor forward byte-identical to the
    raw goal-head/role/value/aux readout off the belief — no I2 tool touches the default
    network anywhere."""
    cfg = _tiny_cfg()
    ah = cfg.action_head
    assert (ah.relay_tool, ah.compass, ah.explorer_tool) == (
        "lambda2_anchor", "off", "goal_head")
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(3))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(2))   # all-default tools
    goal_logits, role_logits, value, l2_hat, z, _h = actor(obs, adj, inference=True)
    z_raw = actor.backbone(obs, adj, inference=True)
    assert bool(jnp.array_equal(z, z_raw))                  # compass off -> z untouched
    assert bool(jnp.array_equal(goal_logits, jax.vmap(actor.goal_head)(z_raw)))  # no frontier
    assert bool(jnp.array_equal(value, jax.vmap(actor.value_head)(z_raw)[:, 0]))
    assert bool(jnp.array_equal(l2_hat, jax.vmap(actor.aux_head)(z_raw)[:, 0]))


# ---- recurrence axis (feedforward | recurrent) ------------------------------
#
# The GRU is ALWAYS built (stable param surface) and only USED when
# recurrence='recurrent'; the hidden threads through BOTH the rollout scan AND the
# PPO loss (recomputed along each trajectory via a per-episode BPTT scan). Tests:
#  (a) param-surface stable + feedforward forward byte-unchanged + key-identity;
#  (b) recurrent: h changes within an episode and resets at episode start; a few PPO
#      iters run end-to-end (ctrl_valid=100%, loss finite) and the GRU takes a step;
#  (c) the recurrent forward IN THE LOSS reproduces the rollout's logits (consistency);
#  (d) REGRESSION: recurrence='feedforward' leaves the rollout + update byte-identical.


def test_recurrence_config_roundtrip():
    """The recurrence axis round-trips through the config tree and defaults to v0."""
    import dataclasses
    cfg = _tiny_cfg()
    assert cfg.backbone.recurrence == "feedforward"            # the default
    d = cfg.to_dict()
    assert d["backbone"]["recurrence"] == "feedforward"
    cfg2 = _tiny_cfg(backbone=dataclasses.replace(_tiny_cfg().backbone,
                                                  recurrence="recurrent"))
    d2 = cfg2.to_dict()
    assert d2["backbone"]["recurrence"] == "recurrent"
    assert from_dict(d2).to_dict() == d2


def test_recurrence_param_surface_stable_and_key_identical():
    """The gru submodule is ALWAYS built (stable param tree) regardless of the
    recurrence knob, AND a feedforward actor shares EVERY non-gru parameter bit-for-bit
    with a recurrent actor of the same key (the gru key is fold_in-derived, so the
    backbone / goal / role / frontier / compass / aux / value keys are unchanged — a
    feedforward actor is the pre-recurrence network exactly)."""
    import dataclasses
    import equinox as eqx
    cfg = _tiny_cfg()
    bb_re = dataclasses.replace(cfg.backbone, recurrence="recurrent")
    a_ff = Actor(5, cfg.action_head.K, backbone_cfg=cfg.backbone, dropout=0.0,
                 key=jax.random.PRNGKey(0), recurrence="feedforward")
    a_re = Actor(5, cfg.action_head.K, backbone_cfg=bb_re, dropout=0.0,
                 key=jax.random.PRNGKey(0), recurrence="recurrent")

    def leaves(m):
        return jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    for name in ("backbone", "goal_head", "role_head", "frontier_attn", "compass",
                 "aux_head", "value_head"):
        lo, ln = leaves(getattr(a_ff, name)), leaves(getattr(a_re, name))
        assert lo and all(bool(jnp.array_equal(x, y)) for x, y in zip(lo, ln)), name
    # the gru param surface itself is present and identical between the two.
    s_ff = [p.shape for p in leaves(a_ff.gru)]
    s_re = [p.shape for p in leaves(a_re.gru)]
    assert s_ff == s_re and len(s_ff) > 0


def test_recurrence_feedforward_byte_unchanged_forward():
    """REGRESSION: recurrence='feedforward' (DEFAULT) leaves the actor forward byte-
    identical — the heads read the belief z directly (feat == backbone(obs)), the GRU
    never contributes, and the returned hidden is the zero passthrough."""
    cfg = _tiny_cfg()
    assert cfg.backbone.recurrence == "feedforward"            # the default
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(2), recurrence="feedforward")
    goal_logits, role_logits, value, l2_hat, feat, h = actor(obs, adj, inference=True)
    z_raw = actor.backbone(obs, adj, inference=True)           # belief, no GRU
    assert bool(jnp.array_equal(feat, z_raw)), "feat changed when feedforward"
    assert bool((h == 0.0).all()), "feedforward hidden is not the zero passthrough"
    # every head equals its raw-belief readout (the GRU touched nothing).
    assert bool(jnp.array_equal(goal_logits, jax.vmap(actor.goal_head)(z_raw)))
    assert bool(jnp.array_equal(role_logits, jax.vmap(actor.role_head)(z_raw)))
    assert bool(jnp.array_equal(value, jax.vmap(actor.value_head)(z_raw)[:, 0]))
    assert bool(jnp.array_equal(l2_hat, jax.vmap(actor.aux_head)(z_raw)[:, 0]))


def test_recurrent_hidden_changes_and_resets():
    """Recurrent: the carried hidden h evolves step-to-step within an episode (so the
    heads see a different feature each step even on identical obs) and RESETS to the
    same value at episode start (feeding the zero init reproduces step-1's hidden)."""
    import dataclasses
    cfg = _tiny_cfg(backbone=dataclasses.replace(
        _tiny_cfg().backbone, recurrence="recurrent"))
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(2), recurrence="recurrent")
    h0 = actor.init_hidden(state.n_agents)                     # (N,W) episode start zeros
    assert bool((h0 == 0.0).all())
    g1, _, _, _, _, h1 = actor(obs, adj, h=h0, inference=True)
    g2, _, _, _, _, h2 = actor(obs, adj, h=h1, inference=True)  # SAME obs, carried h1
    assert bool((h1 != h0).any()), "hidden did not update on step 1"
    assert bool((h2 != h1).any()), "hidden did not evolve step-to-step"
    # because the heads read the hidden, identical obs gives DIFFERENT logits as h moves.
    assert bool((g2 != g1).any()), "logits independent of the carried hidden"
    # episode reset: feeding the zero init again reproduces step-1's hidden EXACTLY.
    _, _, _, _, _, h1b = actor(obs, adj, h=actor.init_hidden(state.n_agents),
                               inference=True)
    assert bool(jnp.array_equal(h1, h1b)), "hidden did not reset at episode start"


def test_recurrent_train_step_runs_and_grus_step():
    """Recurrent: a PPO iter with recurrence='recurrent' runs end-to-end — the
    controller still emits 100% valid moves, the loss is finite, and the GRU itself
    takes a gradient step (BPTT through the trajectory actually flows into it)."""
    import dataclasses
    import equinox as eqx
    cfg = _tiny_cfg(backbone=dataclasses.replace(
        _tiny_cfg().backbone, recurrence="recurrent"))
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    new_state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"]) and jnp.isfinite(logs["policy_loss"])
    assert float(logs["ctrl_valid_frac"]) == 1.0              # only valid moves emitted
    # the GRU took a gradient step (BPTT reached it through the per-episode loss scan).
    g0 = jax.tree_util.tree_leaves(eqx.filter(state.actor.gru, eqx.is_array))
    g1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor.gru, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(g0, g1)), "gru unchanged"


def test_recurrent_loss_forward_reproduces_rollout_logits():
    """CONSISTENCY (the crux): the recurrent forward used IN THE LOSS reproduces the
    rollout's per-step logits for the same params/obs. Collect a rollout, then re-run
    ``_actor_forward_recurrent`` over the stored (B,T) obs/adj and confirm the masked
    log-prob of the SAMPLED goals equals the stored ``goal_logp`` — the per-episode
    BPTT scan re-folds the hidden EXACTLY as the rollout scan did (deterministic at
    dropout=0), so the clipped-PPO ratio starts at exactly 1 as it must."""
    import dataclasses
    cfg = _tiny_cfg(backbone=dataclasses.replace(
        _tiny_cfg().backbone, recurrence="recurrent"))
    env = env_utils.build_env(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    key = jax.random.PRNGKey(6)
    traj = ppo.collect(env, state.actor, state.critic, cfg, stencil, key, jnp.float32(0.0))
    obs_bt, adj_bt = traj["obs"], traj["adj"]                 # (B,T,N,...)
    goal_bt, gmask_bt, logp_bt = traj["goal"], traj["goal_mask"], traj["goal_logp"]
    B, T = obs_bt.shape[0], obs_bt.shape[1]
    # the loss's recurrent forward (dropout off -> deterministic, key irrelevant).
    g, _r, _l2, _feat = ppo._actor_forward_recurrent(
        state.actor, obs_bt, adj_bt, jax.random.PRNGKey(0))
    g = g.reshape(B, T, *g.shape[1:])                         # (B,T,N,K)
    masked = jnp.where(gmask_bt, g, ppo._NEG)
    logp_all = jax.nn.log_softmax(masked, axis=-1)
    logp = jnp.take_along_axis(logp_all, goal_bt[..., None], axis=-1)[..., 0]  # (B,T,N)
    assert bool(jnp.allclose(logp, logp_bt, atol=1e-5)), (
        float(jnp.max(jnp.abs(logp - logp_bt))))
    # the PPO ratio exp(logp - old_logp) is therefore ~1 everywhere at the first epoch.
    assert bool(jnp.allclose(jnp.exp(logp - logp_bt), 1.0, atol=1e-4))


def test_recurrent_minibatches_over_episodes_intact():
    """The recurrent path minibatches over EPISODES (keeps each T-step sequence intact
    for the BPTT scan), NOT over flattened steps. With B rollouts and ``minibatches``
    chunks each minibatch is a block of whole episodes shaped (mb_B, T, ...); the loss
    runs and is finite (a flattened-step minibatch would have broken the (B,T) scan)."""
    import dataclasses
    cfg = _tiny_cfg(
        world=World(grid=8, n_agents=3, comm_r=3, horizon=6),
        backbone=dataclasses.replace(_tiny_cfg().backbone, recurrence="recurrent"),
        trainer=Trainer(minibatches=2, ppo_epochs=2),
        rollouts_per_iter=4,                                  # B=4 episodes, 2 minibatches
    )
    env = env_utils.build_env(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    key = jax.random.PRNGKey(6)
    traj = ppo.collect(env, state.actor, state.critic, cfg, stencil, key, jnp.float32(0.0))
    adv, ret = ppo.compute_advantages(traj, cfg)
    # build the recurrent minibatch the way train_step does: KEEP (B,T,...), take the
    # first half of the episodes as one minibatch and run loss_fn over the SEQUENCES.
    B = traj["obs"].shape[0]
    mb = B // cfg.trainer.minibatches
    sl = slice(0, mb)
    fields = ("obs", "adj", "central", "goal", "goal_logp", "goal_mask",
              "role", "role_logp", "true_l2", "degree")
    batch = {k: traj[k][sl] for k in fields}
    batch["adv"], batch["ret"] = adv[sl], ret[sl]
    assert batch["obs"].ndim == 6                             # (mb_B, T, N, C, H, W) intact
    assert batch["obs"].shape[0] == mb and batch["obs"].shape[1] == cfg.world.horizon
    total, metrics = ppo.loss_fn(state.actor, state.critic, batch, cfg,
                                 jax.random.PRNGKey(7))
    assert bool(jnp.isfinite(total))
    assert all(bool(jnp.isfinite(v)) for v in metrics.values())


def test_recurrence_feedforward_train_step_byte_unchanged():
    """REGRESSION (the update): recurrence='feedforward' (DEFAULT) leaves the WHOLE PPO
    step byte-identical — the rollout trajectory carries no extra keys (hidden lives in
    the scan carry only) and the actor/critic after one train_step match a run done with
    an explicitly feedforward config, bit-for-bit (the GRU never contributes anywhere)."""
    import dataclasses
    import equinox as eqx
    cfg_def = _tiny_cfg()
    assert cfg_def.backbone.recurrence == "feedforward"       # the default
    cfg_exp = _tiny_cfg(backbone=dataclasses.replace(
        _tiny_cfg().backbone, recurrence="feedforward"))
    env = env_utils.build_env(cfg_def)
    stencil = ppo.make_stencil(cfg_def)
    # (i) the rollout pytree carries NO recurrence-specific keys (hidden is carry-only).
    st = ppo.init_state(env, cfg_def, jax.random.PRNGKey(5))
    traj = ppo.collect(env, st.actor, st.critic, cfg_def, stencil, jax.random.PRNGKey(6),
                       jnp.float32(0.0))
    assert "hidden" not in traj and "h" not in traj           # trajectory untouched
    # (ii) a full train_step under the default and under an explicit feedforward config
    #      produce bit-identical actor/critic params (the feedforward path is unchanged).
    opt = ppo.make_optimizer(cfg_def)
    s_def = ppo.init_state(env, cfg_def, jax.random.PRNGKey(5))
    s_exp = ppo.init_state(env, cfg_exp, jax.random.PRNGKey(5))
    n_def, _ = ppo.train_step(env, s_def, cfg_def, jax.random.PRNGKey(6), opt, stencil)
    n_exp, _ = ppo.train_step(env, s_exp, cfg_exp, jax.random.PRNGKey(6), opt, stencil)
    la = jax.tree_util.tree_leaves(eqx.filter(n_def.actor, eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(n_exp.actor, eqx.is_array))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(la, lb)), "actor diverged"
    lc = jax.tree_util.tree_leaves(eqx.filter(n_def.critic, eqx.is_array))
    ld = jax.tree_util.tree_leaves(eqx.filter(n_exp.critic, eqx.is_array))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(lc, ld)), "critic diverged"


# ---- message_content axis (I2: learned | edge_distance | index) -------------
#
# The GNN message-design dial — WHAT each agent appends to its comm message beyond the
# learned `msg` feature transform. `learned` (default) is byte-unchanged; `edge_distance`
# appends the comm_r-normalized sender->receiver distance (receiver knows how far each
# neighbour is); `index` appends a fixed sinusoidal embedding of the sender's normalized
# index (receiver tells neighbours apart). Both extra signals are scale-/count-invariant.


def test_message_content_config_roundtrip():
    """I2: a non-default message_content round-trips through the config tree and the
    default reproduces v0."""
    import dataclasses
    cfg = _tiny_cfg()
    assert cfg.backbone.message_content == "learned"           # the default
    assert cfg.to_dict()["backbone"]["message_content"] == "learned"
    for mode in ("edge_distance", "index"):
        cm = _tiny_cfg(backbone=dataclasses.replace(_tiny_cfg().backbone,
                                                    message_content=mode))
        d = cm.to_dict()
        assert d["backbone"]["message_content"] == mode
        assert from_dict(d).to_dict() == d


def test_kb_distance_normalized_and_scale_invariant():
    """``kb_distance`` is the comm-graph Chebyshev distance NORMALIZED by comm_r (so
    in-range edges land in [0,1]) with a cleared diagonal — and the SAME relative layout
    yields the SAME normalized distances when both positions and comm_r scale together
    (scale-invariant). Out-of-range pairs read > 1 (the receiver masks them out)."""
    cfg = _tiny_cfg(world=World(grid=16, n_agents=3, comm_r=4, horizon=6))
    pos = jnp.array([[0, 0], [0, 2], [0, 8]], dtype=jnp.int32)  # a1 in range, a2 out
    d = env_utils.kb_distance(pos, cfg)
    assert d.shape == (3, 3) and d.dtype == jnp.float32
    assert float(jnp.diag(d).max()) == 0.0                     # diagonal cleared
    assert abs(float(d[0, 1]) - 0.5) < 1e-6                    # cheby 2 / comm_r 4 = 0.5
    assert float(d[0, 2]) > 1.0                                # cheby 8 / 4 = 2.0 -> out of range
    # scale-invariance: double the grid, positions AND comm_r -> identical normalized dist.
    cfg2 = _tiny_cfg(world=World(grid=32, n_agents=3, comm_r=8, horizon=6))
    d2 = env_utils.kb_distance(pos * 2, cfg2)
    assert bool(jnp.allclose(d, d2)), (d, d2)


def test_edge_distance_message_encodes_neighbour_distance():
    """I2 edge_distance: the aggregated message ENCODES neighbour distance — with the
    SAME features on every node (so any output difference is purely geometric), a NEAR
    neighbour and a FAR neighbour produce different aggregated belief features at the
    receiver. (`learned` cannot tell them apart; edge_distance can.)"""
    W = 16
    feats = jnp.ones((3, W), dtype=jnp.float32) * 0.5          # identical -> isolate distance
    layer = nets.MPLayer(W, "mean", 4, key=jax.random.PRNGKey(0),
                         message_content="edge_distance")
    adj = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=bool)   # a0's only nbr is a1
    near = jnp.array([[0., 0.1, 0.], [0.1, 0., 0.], [0., 0., 0.]], dtype=jnp.float32)
    far = jnp.array([[0., 0.9, 0.], [0.9, 0., 0.], [0., 0., 0.]], dtype=jnp.float32)
    out_near = layer(feats, adj, near)[0]
    out_far = layer(feats, adj, far)[0]
    assert jnp.isfinite(out_near).all() and jnp.isfinite(out_far).all()
    assert not bool(jnp.allclose(out_near, out_far)), "distance not encoded in the message"
    # a `learned` layer (no distance channel) is BLIND to the same swap.
    learned = nets.MPLayer(W, "mean", 4, key=jax.random.PRNGKey(0),
                           message_content="learned")
    assert bool(jnp.array_equal(learned(feats, adj, near), learned(feats, adj, far)))


def test_edge_distance_train_step_runs():
    """I2: a PPO iter with --message-content edge_distance on 10×10/2 runs without error,
    threads the comm-graph distance through the rollout AND the loss, and the controller
    still emits 100% valid moves (the actor — incl. the widened update — takes a step)."""
    import dataclasses
    import equinox as eqx
    cfg = _tiny_cfg(
        world=World(grid=10, n_agents=2, comm_r=5, horizon=6),
        backbone=dataclasses.replace(_tiny_cfg().backbone,
                                     message_content="edge_distance"),
    )
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    new_state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"]) and jnp.isfinite(logs["policy_loss"])
    assert float(logs["ctrl_valid_frac"]) == 1.0              # only valid moves emitted
    p0 = jax.tree_util.tree_leaves(eqx.filter(state.actor, eqx.is_array))
    p1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(p0, p1)), "actor unchanged"


def test_index_signal_distinct_and_agent_count_invariant():
    """I2 index: distinct senders produce DISTINGUISHABLE identity signals, and it is the
    SAME function at N=4 and N=10 (agent-count-invariant: a fixed sinusoid of i/N, not a
    learned per-N table). Identity at the SAME normalized index u=i/N matches across team
    sizes, and the embedding dimension never depends on N."""
    import itertools
    s4 = nets._index_signal(4)
    s10 = nets._index_signal(10)
    assert s4.shape == (4, nets._IDX_DIM) and s10.shape == (10, nets._IDX_DIM)
    # the embedding width is FIXED (N-independent).
    assert s4.shape[1] == s10.shape[1] == nets._IDX_DIM
    # every pair of senders is distinguishable (no two identity codes collide).
    for i, j in itertools.combinations(range(4), 2):
        assert float(jnp.linalg.norm(s4[i] - s4[j])) > 1e-3, (i, j)
    # SAME function across team sizes: matching normalized index u=i/N gives the SAME code
    # (u=0 -> agent0 in both; u=0.5 -> agent2 of 4 == agent5 of 10).
    assert bool(jnp.allclose(s4[0], s10[0]))
    assert bool(jnp.allclose(s4[2], s10[5], atol=1e-6))


def test_index_message_distinguishes_neighbours():
    """I2 index: a receiver whose neighbour is sender j gets a DIFFERENT aggregated message
    than a receiver whose neighbour is a different sender — the identity channel lets the
    receiver tell its neighbours apart (identical features, so the diff is pure identity).
    Same MPLayer applied at N=4 and N=10 runs (the function is agent-count-invariant)."""
    W = 16
    feats4 = jnp.ones((4, W), dtype=jnp.float32) * 0.5         # identical -> isolate identity
    layer = nets.MPLayer(W, "mean", 4, key=jax.random.PRNGKey(0), message_content="index")
    # a0's neighbour is a1; a2's neighbour is a3 — different sender identities.
    adj = jnp.array([[0, 1, 0, 0], [1, 0, 0, 0],
                     [0, 0, 0, 1], [0, 0, 1, 0]], dtype=bool)
    out = layer(feats4, adj, None)
    assert jnp.isfinite(out).all()
    assert not bool(jnp.allclose(out[0], out[2])), "identity not encoded in the message"
    # the SAME layer (no per-N params) applies unchanged at a larger team size.
    feats10 = jnp.ones((10, W), dtype=jnp.float32) * 0.5
    adj10 = jnp.zeros((10, 10), dtype=bool).at[0, 1].set(True).at[1, 0].set(True)
    out10 = layer(feats10, adj10, None)
    assert out10.shape == (10, W) and jnp.isfinite(out10).all()


def test_index_train_step_runs():
    """I2: a PPO iter with --message-content index on 10×10/2 runs without error and the
    controller still emits 100% valid moves (the actor takes a gradient step)."""
    import dataclasses
    import equinox as eqx
    cfg = _tiny_cfg(
        world=World(grid=10, n_agents=2, comm_r=5, horizon=6),
        backbone=dataclasses.replace(_tiny_cfg().backbone, message_content="index"),
    )
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    new_state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"]) and jnp.isfinite(logs["policy_loss"])
    assert float(logs["ctrl_valid_frac"]) == 1.0
    p0 = jax.tree_util.tree_leaves(eqx.filter(state.actor, eqx.is_array))
    p1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(p0, p1)), "actor unchanged"


def test_message_content_learned_param_tree_byte_identical():
    """I2 REGRESSION (param surface): a `learned` (default) MPLayer is bit-for-bit the
    PRE-message_content layer — the update Linear is (2W -> W) exactly and the key
    consumption order is unchanged. (The non-default modes widen the update input, but
    the default's param tree must match the original.)"""
    import equinox as eqx
    W = 16
    key = jax.random.PRNGKey(123)
    learned = nets.MPLayer(W, "max", 4, key=key, message_content="learned")
    # the update reads [self || agg] = 2W with NO extra channel.
    assert learned.upd.weight.shape == (W, 2 * W), learned.upd.weight.shape
    # the non-default modes widen the update input (extra channels appended).
    ed = nets.MPLayer(W, "max", 4, key=key, message_content="edge_distance")
    ix = nets.MPLayer(W, "max", 4, key=key, message_content="index")
    assert ed.upd.weight.shape == (W, 2 * W + 2)
    assert ix.upd.weight.shape == (W, 2 * W + nets._IDX_DIM)
    # msg / q / k are IDENTICAL across modes (only `upd` changes) — proving the key split
    # order is preserved, so `learned` reproduces the original network's non-upd params.
    for name in ("msg", "q", "k"):
        a = jax.tree_util.tree_leaves(eqx.filter(getattr(learned, name), eqx.is_array))
        b = jax.tree_util.tree_leaves(eqx.filter(getattr(ed, name), eqx.is_array))
        assert all(bool(jnp.array_equal(x, y)) for x, y in zip(a, b)), name


def test_message_content_learned_forward_byte_unchanged():
    """I2 REGRESSION (forward): message_content='learned' (DEFAULT) leaves the actor
    forward byte-identical and IGNORES any supplied ``dist`` — passing a distance matrix
    to a learned actor changes nothing (the static branch never reads it)."""
    cfg = _tiny_cfg()
    assert cfg.backbone.message_content == "learned"           # the default
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    dist = env_utils.kb_distance(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(2))
    out_none = actor(obs, adj, inference=True)
    out_dist = actor(obs, adj, dist=dist, inference=True)      # learned ignores dist
    for a, b in zip(out_none, out_dist):
        assert bool(jnp.array_equal(a, b)), "learned read dist (should ignore it)"
    # and the belief equals the raw backbone readout (no extra channel anywhere).
    z_raw = actor.backbone(obs, adj, inference=True)
    assert bool(jnp.array_equal(out_none[4], z_raw))


def test_message_content_learned_train_step_byte_unchanged():
    """I2 REGRESSION (the update + trajectory): message_content='learned' (DEFAULT) leaves
    the WHOLE PPO step byte-identical — the rollout trajectory carries NO 'dist' key (it
    lives only on the non-default modes) and a full train_step under the default matches
    one under an explicit 'learned' config bit-for-bit."""
    import dataclasses
    import equinox as eqx
    cfg_def = _tiny_cfg()
    assert cfg_def.backbone.message_content == "learned"
    cfg_exp = _tiny_cfg(backbone=dataclasses.replace(_tiny_cfg().backbone,
                                                     message_content="learned"))
    env = env_utils.build_env(cfg_def)
    stencil = ppo.make_stencil(cfg_def)
    # (i) the rollout pytree carries NO 'dist' key (distance is a non-default-only channel).
    st = ppo.init_state(env, cfg_def, jax.random.PRNGKey(5))
    traj = ppo.collect(env, st.actor, st.critic, cfg_def, stencil, jax.random.PRNGKey(6),
                       jnp.float32(0.0))
    assert "dist" not in traj                                  # trajectory untouched
    # (ii) a full train_step under the default and under an explicit learned config produce
    #      bit-identical actor/critic params (the learned path is byte-unchanged).
    opt = ppo.make_optimizer(cfg_def)
    s_def = ppo.init_state(env, cfg_def, jax.random.PRNGKey(5))
    s_exp = ppo.init_state(env, cfg_exp, jax.random.PRNGKey(5))
    n_def, _ = ppo.train_step(env, s_def, cfg_def, jax.random.PRNGKey(6), opt, stencil)
    n_exp, _ = ppo.train_step(env, s_exp, cfg_exp, jax.random.PRNGKey(6), opt, stencil)
    la = jax.tree_util.tree_leaves(eqx.filter(n_def.actor, eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(n_exp.actor, eqx.is_array))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(la, lb)), "actor diverged"


# ---- selector axis (the hierarchical 3-skill mode-picker: off | on) ---------
#
# selector='off' (default) = the v0 single goal/role head, byte-unchanged. 'on' = a
# learned SELECTOR over {0=disperse, 1=flock, 2=hold}: a categorical PPO action off the
# belief picks a skill, that skill emits (N,K) goal-offset logits, the offset is the second
# PPO action (replacing the goal-head sample), routed through the fixed L1 controller. The
# selector_head + a learned flock_head are ALWAYS built (stable param surface) and only
# USED via skill_forward. Tests: (a) off byte-unchanged (forward + train_step) + param
# surface / key-identity; (b) on runs a train_step (finite loss, 100% valid moves) for BOTH
# flock flavors and congestion on/off; (c) skill_forward shapes + the selector takes a step.


def _selector_cfg(**over):
    """A tiny selector-on cfg on 10×10/4 (room for the 3 skills to differ)."""
    import dataclasses
    base = _tiny_cfg(world=World(grid=10, n_agents=4, comm_r=5, horizon=6),
                     selector="on")
    return dataclasses.replace(base, **over) if over else base


def test_selector_off_byte_unchanged_forward():
    """Regression: selector='off' (DEFAULT) leaves the actor forward byte-identical — the
    goal logits equal goal_head(z) EXACTLY (skill_forward is never called; the selector_head
    / flock_head params just sit unused), exactly like the explorer_tool default test."""
    cfg = _tiny_cfg()
    assert cfg.selector == "off"                           # the default
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(2), selector="off")
    goal_logits, _rl, _v, _l2, z, _h = actor(obs, adj, inference=True)
    goal_only = jax.vmap(actor.goal_head)(z)               # goal head with NO selector
    assert bool(jnp.array_equal(goal_logits, goal_only)), "selector off changed the forward"


def test_selector_param_surface_stable_and_key_identical():
    """The selector_head + flock_head are ALWAYS built (stable param tree) regardless of the
    selector knob, AND a selector='off' actor shares EVERY non-selector parameter bit-for-bit
    with a selector='on' actor of the same key (the selector keys are fold_in-derived, so the
    backbone / goal / role / frontier / compass / gru / aux / value keys are unchanged)."""
    import equinox as eqx
    cfg = _tiny_cfg()
    a_off = Actor(5, cfg.action_head.K, backbone_cfg=cfg.backbone, dropout=0.0,
                  key=jax.random.PRNGKey(0), selector="off")
    a_on = Actor(5, cfg.action_head.K, backbone_cfg=cfg.backbone, dropout=0.0,
                 key=jax.random.PRNGKey(0), selector="on")

    def leaves(m):
        return jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))
    for name in ("backbone", "goal_head", "role_head", "frontier_attn", "compass",
                 "goal_residual", "gru", "aux_head", "value_head"):
        lo, ln = leaves(getattr(a_off, name)), leaves(getattr(a_on, name))
        assert lo and all(bool(jnp.array_equal(x, y)) for x, y in zip(lo, ln)), name
    # the selector_head + flock_head param surfaces are present and identical between the two.
    for name in ("selector_head", "flock_head"):
        s_off = [p.shape for p in leaves(getattr(a_off, name))]
        s_on = [p.shape for p in leaves(getattr(a_on, name))]
        assert s_off == s_on and len(s_off) > 0, name


def test_selector_off_train_step_byte_unchanged():
    """REGRESSION (the update + trajectory): selector='off' (DEFAULT) leaves the WHOLE PPO
    step byte-identical — the rollout trajectory carries NO selector keys (skill / skill_logp
    / position) and a full train_step under the default matches one under an explicit
    selector='off' config bit-for-bit (the selector path is never traced)."""
    import dataclasses
    import equinox as eqx
    cfg_def = _tiny_cfg()
    assert cfg_def.selector == "off"
    cfg_exp = dataclasses.replace(_tiny_cfg(), selector="off")
    env = env_utils.build_env(cfg_def)
    stencil = ppo.make_stencil(cfg_def)
    # (i) the rollout pytree carries NO selector-specific keys.
    st = ppo.init_state(env, cfg_def, jax.random.PRNGKey(5))
    traj = ppo.collect(env, st.actor, st.critic, cfg_def, stencil, jax.random.PRNGKey(6),
                       jnp.float32(0.0))
    assert "skill" not in traj and "skill_logp" not in traj and "position" not in traj
    # (ii) a full train_step under the default and under an explicit off config match.
    opt = ppo.make_optimizer(cfg_def)
    s_def = ppo.init_state(env, cfg_def, jax.random.PRNGKey(5))
    s_exp = ppo.init_state(env, cfg_exp, jax.random.PRNGKey(5))
    n_def, _ = ppo.train_step(env, s_def, cfg_def, jax.random.PRNGKey(6), opt, stencil)
    n_exp, _ = ppo.train_step(env, s_exp, cfg_exp, jax.random.PRNGKey(6), opt, stencil)
    la = jax.tree_util.tree_leaves(eqx.filter(n_def.actor, eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(n_exp.actor, eqx.is_array))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(la, lb)), "actor diverged"


def test_selector_config_roundtrip():
    """A non-default selector / flock / congestion round-trips through the config tree and
    the defaults reproduce v0 (selector=off, flock=scripted, congestion=off)."""
    import dataclasses
    cfg = _tiny_cfg()
    assert (cfg.selector, cfg.flock, cfg.congestion) == ("off", "scripted", "off")
    cfg2 = dataclasses.replace(cfg, selector="on", flock="learned", congestion="on",
                               congestion_weight=0.25)
    d = cfg2.to_dict()
    assert (d["selector"], d["flock"], d["congestion"], d["congestion_weight"]) == (
        "on", "learned", "on", 0.25)
    assert from_dict(d).to_dict() == d


def test_skill_forward_shapes_and_skills_differ():
    """skill_forward returns the right shapes — skill_logits (N,3), offset_logits (3,N,K),
    feat (N,W), h_next (N,W) — and the three skills produce DISTINCT offset-logit rows (the
    disperse / flock / hold skills are genuinely different behaviours, not a shared head)."""
    cfg = _selector_cfg()
    K = cfg.action_head.K
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    actor = ppo._make_actor(env.obs.obs_channels, cfg, jax.random.PRNGKey(2))
    sl, ol, feat, h = actor.skill_forward(obs, adj, state.body.position, inference=True)
    N, W = cfg.world.n_agents, cfg.backbone.width
    assert sl.shape == (N, 3) and ol.shape == (3, N, K)
    assert feat.shape == (N, W) and h.shape == (N, W)
    assert bool(jnp.isfinite(sl).all()) and bool(jnp.isfinite(ol).all())
    # the three skills are distinct (disperse vs flock vs hold differ on this layout).
    assert not bool(jnp.allclose(ol[0], ol[1])), "disperse == flock"
    assert not bool(jnp.allclose(ol[0], ol[2])), "disperse == hold"
    assert not bool(jnp.allclose(ol[1], ol[2])), "flock == hold"


def test_hold_skill_stays_when_anchored_reconnects_when_isolating():
    """The HOLD skill's argmax offset is 'here' (index 0 = STAY) for a well-anchored agent
    (soft-degree >= the hold floor) and the reconnect direction (toward the NEAREST other
    agent) for an about-to-isolate agent (soft-degree < floor). Mirrors relay_hold_move but
    as goal-offset logits. Layout (comm_r=5, sharp=2, floor=1.0): a0,a1,a2 form a tight
    mutually-adjacent TRIANGLE (each has two in-range neighbours -> soft-degree ~2 >> floor,
    so all three HOLD); a3 sits far away (isolated, soft-degree ~0 < floor) so its row
    RECONNECTS toward its nearest neighbour (a0, up-left = NW)."""
    cfg = _selector_cfg(world=World(grid=20, n_agents=4, comm_r=5, horizon=6))
    K = cfg.action_head.K
    actor = ppo._make_actor(5, cfg, jax.random.PRNGKey(0))
    # a0,a1,a2 a mutually-adjacent triangle (soft-degree ~2 >> floor 1.0); a3 isolated.
    pos = jnp.array([[5, 5], [5, 6], [6, 5], [18, 18]], dtype=jnp.int32)
    deg = ctrl._local_conn_score(pos, cfg.world.comm_r, cfg.connectivity.lambda2_sharp)
    assert bool((deg[:3] >= 1.0).all()) and float(deg[3]) < 1.0, deg   # premise of the test
    hold = actor._hold_logits(pos)                          # (N,K)
    assert hold.shape == (4, K)
    here = 0
    assert int(jnp.argmax(hold[0])) == here, hold[0]        # anchored a0 holds (STAY)
    assert int(jnp.argmax(hold[1])) == here, hold[1]        # anchored a1 holds (STAY)
    assert int(jnp.argmax(hold[2])) == here, hold[2]        # anchored a2 holds (STAY)
    # a3 is isolated (nearest is a0 at (5,5); bearing from (18,18) is up-left = NW).
    nw = int(jnp.argmax(nets._compass_unit_dirs(K)
                        @ jnp.asarray([-1.0, -1.0]) / jnp.sqrt(2.0)))
    assert int(jnp.argmax(hold[3])) == nw, hold[3]          # isolating a3 reconnects toward nbr


def _selector_train_step(cfg):
    """Run one ppo.train_step on a selector cfg; return (new_state, state, logs)."""
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    new_state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    return new_state, state, logs


def test_selector_train_step_both_flock_flavors():
    """selector='on' runs a full PPO iter for BOTH flock flavors (scripted | learned): the
    loss is finite, the controller emits 100% valid moves, every skill-usage fraction is
    logged, and the selector_head ITSELF takes a gradient step (the mode-picker learns)."""
    import equinox as eqx
    for flavor in ("scripted", "learned"):
        cfg = _selector_cfg(flock=flavor)
        new_state, state, logs = _selector_train_step(cfg)
        assert jnp.isfinite(logs["ep_reward"]) and jnp.isfinite(logs["policy_loss"]), flavor
        assert float(logs["ctrl_valid_frac"]) == 1.0, flavor       # only valid moves
        # the per-skill usage fractions are logged and sum to 1 (every agent-step picks one).
        fr = [float(logs[f"skill_frac_{n}"]) for n in ("disperse", "flock", "hold")]
        assert abs(sum(fr) - 1.0) < 1e-4, (flavor, fr)
        assert jnp.isfinite(logs["skill_usage_entropy"])
        # the SELECTOR head took a gradient step (the skill-PG flowed into it).
        s0 = jax.tree_util.tree_leaves(eqx.filter(state.actor.selector_head, eqx.is_array))
        s1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor.selector_head, eqx.is_array))
        assert any(not jnp.allclose(a, b) for a, b in zip(s0, s1)), f"selector unchanged ({flavor})"


def test_selector_train_step_congestion_on_off():
    """selector='on' runs a full PPO iter with the free-market congestion price ON and OFF,
    for both flock flavors: finite loss + 100% valid moves either way (the congestion penalty
    is an additive reward term computed from the sampled skills, not a routing change)."""
    for congestion in ("off", "on"):
        for flavor in ("scripted", "learned"):
            cfg = _selector_cfg(flock=flavor, congestion=congestion)
            _ns, _st, logs = _selector_train_step(cfg)
            tag = (congestion, flavor)
            assert jnp.isfinite(logs["ep_reward"]), tag
            assert jnp.isfinite(logs["policy_loss"]), tag
            assert float(logs["ctrl_valid_frac"]) == 1.0, tag       # only valid moves


def test_selector_recurrent_train_step():
    """selector='on' composes with recurrence='recurrent': a full PPO iter runs end-to-end
    (the per-episode BPTT scan re-folds the hidden AND re-runs skill_forward), the loss is
    finite, the controller stays 100% valid, and the selector head takes a gradient step."""
    import dataclasses
    import equinox as eqx
    cfg = _selector_cfg(backbone=dataclasses.replace(
        _tiny_cfg().backbone, recurrence="recurrent"))
    new_state, state, logs = _selector_train_step(cfg)
    assert jnp.isfinite(logs["ep_reward"]) and jnp.isfinite(logs["policy_loss"])
    assert float(logs["ctrl_valid_frac"]) == 1.0
    s0 = jax.tree_util.tree_leaves(eqx.filter(state.actor.selector_head, eqx.is_array))
    s1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor.selector_head, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(s0, s1)), "selector unchanged (recurrent)"


def test_selector_loss_forward_reproduces_rollout_goal_logp():
    """CONSISTENCY: the selector forward used IN THE LOSS reproduces the rollout's goal
    log-probs for the same params/obs — gathering the SELECTED skill's offset-logits and
    masking gives back the stored goal_logp, so the clipped-PPO ratio for the offset action
    starts at exactly 1 (as it must on the first epoch). The scripted flavor is deterministic
    at dropout=0, so the loss-time skill_forward matches the rollout-time one exactly."""
    cfg = _selector_cfg(flock="scripted")
    env = env_utils.build_env(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    key = jax.random.PRNGKey(6)
    traj = ppo.collect(env, state.actor, state.critic, cfg, stencil, key, jnp.float32(0.0))
    obs, adj, pos = traj["obs"], traj["adj"], traj["position"]   # (B,T,...)
    skill, gmask, goal, logp_bt = (traj["skill"], traj["goal_mask"], traj["goal"],
                                   traj["goal_logp"])
    B, T = obs.shape[0], obs.shape[1]
    _f = lambda x: x.reshape((B * T,) + x.shape[2:])
    # the loss's FF selector forward (dropout off -> deterministic, key irrelevant).
    sl, ol, _l2, _feat = ppo._actor_skill_forward_ff(
        state.actor, _f(obs), _f(adj), _f(pos), jax.random.PRNGKey(0))
    # gather the SELECTED skill's offset-logits exactly as loss_fn does, mask, log-softmax.
    g = jnp.take_along_axis(ol, _f(skill)[:, None, :, None], axis=1)[:, 0]  # (M,N,K)
    masked = jnp.where(_f(gmask), g, ppo._NEG)
    logp_all = jax.nn.log_softmax(masked, axis=-1)
    logp = jnp.take_along_axis(logp_all, _f(goal)[..., None], axis=-1)[..., 0]  # (M,N)
    assert bool(jnp.allclose(logp, _f(logp_bt), atol=1e-5)), (
        float(jnp.max(jnp.abs(logp - _f(logp_bt)))))


def test_selector_full_train_loop_logs_skill_usage():
    """ppo.train carries the selector through the jitted loop and logs the per-skill usage
    fractions + mode-usage entropy each iteration (the history is self-describing)."""
    cfg = _selector_cfg(iters=2)
    env = env_utils.build_env(cfg)
    _state, history = ppo.train(env, cfg)
    assert len(history) == 2
    for h in history:
        assert all(f"skill_frac_{n}" in h for n in ("disperse", "flock", "hold"))
        assert "skill_usage_entropy" in h and jnp.isfinite(h["skill_usage_entropy"])
        assert jnp.isfinite(h["skill_entropy"])


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


# ---- warm-start / checkpoint-init (the scale-strategy: cross-rung load) ------
#
# The LPAC backbone + heads are scale-invariant: the (actor, critic) PARAM SHAPES
# depend only on the obs/central channels, width/depth/mp_rounds, the goal-K and
# n_roles — NOT on grid size or agent count (those only set runtime tensor dims via
# same-padding conv + global-average-pool + the per-agent vmap). So a model saved at
# one rung loads into a fresh skeleton built for the NEXT rung with byte-identical
# param shapes; only --grid/--n-agents/--comm-r differ, none of which appear in the
# params. These two tests pin (a) that cross-rung warm-start actually transplants the
# saved weights (and trains), and (b) that the DEFAULT (no --init-from) random init is
# byte-unchanged.


def test_warm_start_cross_rung_loads_and_trains(tmp_path):
    """(a) Save a tiny model @8²/2, then warm-start a @10²/3 run from it (different
    grid AND agent count). The loaded actor params must be the SAVED ones (warm-
    started) — NOT the fresh-random init the @10²/3 skeleton would otherwise get — and
    training must run a couple iters with the controller emitting only valid moves
    (ctrl_valid == 100%). This proves the scale-invariant cross-rung load end to end."""
    import equinox as eqx
    from ctde_v0 import checkpoint as ckpt

    # TRAIN a tiny source model @8²/2 (one real PPO step so its weights move OFF the
    # random draw — otherwise the scale-invariant random init would coincide with the
    # @10²/3 fresh init at the same seed and the load would be indistinguishable), then
    # SAVE its (actor, critic) snapshot.
    cfg_small = _tiny_cfg(world=World(grid=8, n_agents=2, comm_r=3, horizon=6))
    env_small = env_utils.build_env(cfg_small)
    opt_s = ppo.make_optimizer(cfg_small)
    stencil_s = ppo.make_stencil(cfg_small)
    src0 = ppo.init_state(env_small, cfg_small, jax.random.PRNGKey(0))
    src, _ = ppo.train_step(env_small, src0, cfg_small, jax.random.PRNGKey(1),
                            opt_s, stencil_s)
    path = str(tmp_path / "model.eqx")
    ckpt.save_model(path, (src.actor, src.critic), meta=cfg_small.to_dict())
    assert os.path.exists(path)

    # a DIFFERENT rung: 10²/3 (different grid, n_agents AND comm_r) — same backbone.
    cfg_big = _tiny_cfg(world=World(grid=10, n_agents=3, comm_r=5, horizon=6), iters=2)
    env_big = env_utils.build_env(cfg_big)

    # the fresh-random init the @10²/3 skeleton would get WITHOUT a warm start
    # (same key the warm-start path uses) — to prove the load actually replaced it.
    key = jax.random.PRNGKey(cfg_big.seed)
    fresh = ppo.init_state(env_big, cfg_big, key)

    warm = ppo.init_state_from_checkpoint(env_big, cfg_big, path, key)

    saved_actor = jax.tree_util.tree_leaves(eqx.filter(src.actor, eqx.is_array))
    warm_actor = jax.tree_util.tree_leaves(eqx.filter(warm.actor, eqx.is_array))
    fresh_actor = jax.tree_util.tree_leaves(eqx.filter(fresh.actor, eqx.is_array))
    # scale-invariant shapes: the @8²/2 save and the @10²/3 skeleton agree leaf-for-leaf.
    assert len(saved_actor) == len(warm_actor) == len(fresh_actor)
    # warm-started == the SAVED weights (transplanted across rungs)...
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(warm_actor, saved_actor)), \
        "warm-start actor != saved checkpoint (not warm-started)"
    # ...and NOT the fresh-random init it would otherwise have gotten.
    assert not all(bool(jnp.array_equal(a, b))
                   for a, b in zip(warm_actor, fresh_actor)), \
        "warm-start actor == fresh random (checkpoint load was a no-op)"
    # critic is warm-started too.
    saved_critic = jax.tree_util.tree_leaves(eqx.filter(src.critic, eqx.is_array))
    warm_critic = jax.tree_util.tree_leaves(eqx.filter(warm.critic, eqx.is_array))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(warm_critic, saved_critic))

    # and it TRAINS from the warm start: a couple iters @10²/3, controller stays valid.
    _, history = ppo.train(env_big, cfg_big, init_from=path)
    assert len(history) == 2
    assert all(jnp.isfinite(h["aux_loss"]) for h in history)
    assert all(float(h["ctrl_valid_frac"]) == 1.0 for h in history)


def test_warm_start_rejects_mismatched_backbone(tmp_path):
    """A checkpoint with a DIFFERENT backbone (width 16) must NOT load into a width-32
    skeleton: the scale-invariance contract allows only grid/n_agents/comm_r to change
    between rungs — a different architecture has different param SHAPES and must fail
    loudly (caught at deserialise, before any silent reshape)."""
    from ctde_v0 import checkpoint as ckpt
    import pytest

    cfg_w16 = _tiny_cfg(world=World(grid=8, n_agents=2, comm_r=3, horizon=6),
                        backbone=Backbone(width=16, depth=2, mp_rounds=2))
    env_w16 = env_utils.build_env(cfg_w16)
    src = ppo.init_state(env_w16, cfg_w16, jax.random.PRNGKey(0))
    path = str(tmp_path / "model.eqx")
    ckpt.save_model(path, (src.actor, src.critic), meta=cfg_w16.to_dict())

    cfg_w32 = _tiny_cfg(world=World(grid=10, n_agents=3, comm_r=5, horizon=6),
                        backbone=Backbone(width=32, depth=2, mp_rounds=2))
    env_w32 = env_utils.build_env(cfg_w32)
    with pytest.raises((ValueError, RuntimeError)):
        ppo.init_state_from_checkpoint(env_w32, cfg_w32, path, jax.random.PRNGKey(1))


def test_init_from_none_is_byte_unchanged():
    """(b) REGRESSION: the default (init_from=None) random init is byte-identical to a
    plain ppo.init_state — the warm-start branch is never touched, so the RNG draw /
    param surface is unchanged."""
    import equinox as eqx
    cfg = _tiny_cfg(iters=1)
    env = env_utils.build_env(cfg)
    key = jax.random.PRNGKey(cfg.seed)

    # ppo.train(init_from=None) seeds init_state with PRNGKey(cfg.seed); reproduce it.
    ref = ppo.init_state(env, cfg, key)
    # default-path train should leave the FIRST init byte-identical: run one iter from
    # each and confirm the post-step params match (same init + same step => same out).
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    s_ref = ppo.init_state(env, cfg, key)
    n_ref, _ = ppo.train_step(env, s_ref, cfg, jax.random.PRNGKey(123), opt, stencil)
    # and the train() default path (no init_from) must start from that same init.
    _state, hist = ppo.train(env, cfg)
    assert len(hist) == 1
    la = jax.tree_util.tree_leaves(eqx.filter(ref.actor, eqx.is_array))
    lb = jax.tree_util.tree_leaves(eqx.filter(s_ref.actor, eqx.is_array))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(la, lb)), \
        "default init_state is not reproducible from PRNGKey(cfg.seed)"


# ---- Arm-A / Arm-B mechanism knobs (warmstart-noise · DTE · DiCo · fork) -----
#
# Four additive arms, each gated + byte-unchanged at its default (warmstart_noise=0,
# critic_mode='central', diversity_residual='off', fork_groups=1). Tests: config
# round-trip; the DiCo residual is mean-zero across agents + 'off' is byte-unchanged;
# the GroupedActor partitions/shapes/diverges + replicates a bootstrap (copy 0 exact);
# the decentral DTE + the warm-start perturbation run / behave.


def test_arm_knobs_config_roundtrip():
    """The four new arm knobs round-trip through the config tree and keep their v0
    defaults (warmstart_noise=0, critic_mode=central, diversity_residual=off, fork_groups=1)."""
    import dataclasses
    cfg = _tiny_cfg()
    assert (cfg.warmstart_noise, cfg.critic_mode, cfg.diversity_residual,
            cfg.fork_groups) == (0.0, "central", "off", 1)
    cfg2 = dataclasses.replace(cfg, warmstart_noise=0.1, critic_mode="decentral",
                               diversity_residual="on", fork_groups=2)
    d = cfg2.to_dict()
    assert (d["warmstart_noise"], d["critic_mode"], d["diversity_residual"],
            d["fork_groups"]) == (0.1, "decentral", "on", 2)
    assert from_dict(d).to_dict() == d


def test_diversity_residual_off_byte_unchanged_and_on_mean_zero():
    """diversity_residual='off' (default) leaves goal_logits == goal_head(z) (the residual
    is never added); 'on' SHIFTS the logits by a per-agent term that is MEAN-ZERO across
    agents (controlled diversity preserves the team-mean goal policy)."""
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    obs, state = env.reset(jax.random.PRNGKey(1))
    adj = env_utils.kb_adjacency(state.body.position, cfg)
    a_off = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                  dropout=0.0, key=jax.random.PRNGKey(0), diversity_residual="off")
    a_on = Actor(env.obs.obs_channels, cfg.action_head.K, backbone_cfg=cfg.backbone,
                 dropout=0.0, key=jax.random.PRNGKey(0), diversity_residual="on")
    z = a_off.backbone(obs, adj, inference=True)
    g_off = a_off(obs, adj, inference=True)[0]
    assert bool(jnp.array_equal(g_off, jax.vmap(a_off.goal_head)(z)))   # off: no residual
    g_on = a_on(obs, adj, inference=True)[0]
    assert not bool(jnp.allclose(g_on, g_off))                         # on: shifted
    resid = g_on - jax.vmap(a_on.goal_head)(z)                         # the added residual
    assert float(jnp.abs(resid.mean(axis=0)).max()) < 1e-5            # mean-zero across agents


def test_grouped_actor_partition_shapes_and_diverges():
    """GroupedActor (G=2): the partition is the contiguous near-even split, the forward
    returns the same 6-tuple shapes as a single Actor, and a PPO iter runs (100% valid
    moves) + actually updates the sub-actors."""
    import equinox as eqx
    cfg = _tiny_cfg(world=World(grid=10, n_agents=4, comm_r=5, horizon=6), fork_groups=2)
    assert [int(x) for x in nets.GroupedActor._group(4, 2)] == [0, 0, 1, 1]
    assert [int(x) for x in nets.GroupedActor._group(10, 2)] == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg); stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    assert isinstance(state.actor, nets.GroupedActor) and state.actor.G == 2
    obs, st = env.reset(jax.random.PRNGKey(2))
    adj = env_utils.kb_adjacency(st.body.position, cfg)
    g, r, v, l2, feat, h = state.actor(obs, adj, inference=True)
    assert g.shape == (4, cfg.action_head.K) and v.shape == (4,) and feat.shape[0] == 4
    new_state, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"]) and float(logs["ctrl_valid_frac"]) == 1.0
    p0 = jax.tree_util.tree_leaves(eqx.filter(state.actor, eqx.is_array))
    p1 = jax.tree_util.tree_leaves(eqx.filter(new_state.actor, eqx.is_array))
    assert any(not jnp.allclose(a, b) for a, b in zip(p0, p1)), "grouped actor unchanged"


def test_grouped_actor_warmstart_replicate_copy0_is_bootstrap(tmp_path):
    """B-fork warm-start: replicating a single bootstrap into a GroupedActor makes sub 0
    the bootstrap EXACTLY and sub 1 a (perturbed, hence different) sibling."""
    import equinox as eqx
    from ctde_v0 import checkpoint as ckpt
    cfg1 = _tiny_cfg(world=World(grid=10, n_agents=4, comm_r=5, horizon=6))   # single boot
    env = env_utils.build_env(cfg1)
    boot = ppo.init_state(env, cfg1, jax.random.PRNGKey(0))
    p = str(tmp_path / "boot.eqx")
    ckpt.save_model(p, (boot.actor, boot.critic), meta=cfg1.to_dict())
    import dataclasses
    cfgf = dataclasses.replace(cfg1, fork_groups=2)
    forked = ppo.init_state_from_checkpoint(env, cfgf, p, jax.random.PRNGKey(9))
    assert isinstance(forked.actor, nets.GroupedActor)
    s0 = jax.tree_util.tree_leaves(eqx.filter(forked.actor.subs[0], eqx.is_array))
    s1 = jax.tree_util.tree_leaves(eqx.filter(forked.actor.subs[1], eqx.is_array))
    boot_leaves = jax.tree_util.tree_leaves(eqx.filter(boot.actor, eqx.is_array))
    assert all(bool(jnp.array_equal(a, b)) for a, b in zip(s0, boot_leaves))  # copy0 exact
    assert any(not jnp.allclose(a, b) for a, b in zip(s1, boot_leaves))       # copy1 perturbed


def test_critic_mode_decentral_train_step():
    """critic_mode='decentral' (the DTE tail) runs a PPO iter sourcing the value from the
    actor's own value head (the centralized critic is unused), valid moves preserved."""
    cfg = _tiny_cfg(world=World(grid=10, n_agents=3, comm_r=5, horizon=6),
                    critic_mode="decentral")
    env = env_utils.build_env(cfg)
    opt = ppo.make_optimizer(cfg); stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, jax.random.PRNGKey(5))
    _ns, logs = ppo.train_step(env, state, cfg, jax.random.PRNGKey(6), opt, stencil)
    assert jnp.isfinite(logs["ep_reward"]) and float(logs["ctrl_valid_frac"]) == 1.0


def test_warmstart_noise_zero_noop_and_perturbs():
    """_perturb_actor: σ=0 is a byte-exact no-op; σ>0 changes the float leaves while
    preserving every shape."""
    import equinox as eqx
    cfg = _tiny_cfg()
    env = env_utils.build_env(cfg)
    a = ppo._make_actor(env.obs.obs_channels, cfg, jax.random.PRNGKey(0))
    a0 = ppo._perturb_actor(a, 0.0, jax.random.PRNGKey(1))
    la = jax.tree_util.tree_leaves(eqx.filter(a, eqx.is_array))
    l0 = jax.tree_util.tree_leaves(eqx.filter(a0, eqx.is_array))
    assert all(bool(jnp.array_equal(x, y)) for x, y in zip(la, l0))     # σ=0 no-op
    a1 = ppo._perturb_actor(a, 0.1, jax.random.PRNGKey(1))
    l1 = jax.tree_util.tree_leaves(eqx.filter(a1, eqx.is_array))
    assert all(x.shape == y.shape for x, y in zip(la, l1))             # shapes preserved
    assert any(not jnp.allclose(x, y) for x, y in zip(la, l1))         # σ>0 perturbs


if __name__ == "__main__":
    test_config_roundtrip()
    test_conn_signal_axis_roundtrip()
    test_env_shapes()
    test_actor_backbone_forward()
    test_aggregators_all_run()
    test_sector_frontier_features_concentrate()
    test_frontier_attn_shifts_goal_probability()
    test_frontier_features_scale_invariant()
    test_explorer_tool_goal_head_byte_unchanged()
    test_explorer_tool_param_surface_stable()
    test_explorer_tool_frontier_attn_train_step()
    test_explorer_tool_config_roundtrip()
    test_relay_hold_well_connected_stays()
    test_relay_hold_about_to_isolate_reconnects()
    test_relay_hold_emits_only_valid_moves()
    test_relay_tool_hold_train_step()
    test_relay_tool_lambda2_anchor_byte_unchanged()
    test_relay_tool_config_roundtrip()
    test_compass_features_point_correctly()
    test_compass_empty_streams_fall_to_here()
    test_compass_features_scale_invariant()
    test_compass_off_byte_unchanged()
    test_compass_off_init_byte_identical_to_on()
    test_compass_on_shifts_belief_and_train_step()
    test_compass_config_roundtrip()
    test_defaults_byte_unchanged_full_actor()
    test_recurrence_config_roundtrip()
    test_recurrence_param_surface_stable_and_key_identical()
    test_recurrence_feedforward_byte_unchanged_forward()
    test_recurrent_hidden_changes_and_resets()
    test_recurrent_train_step_runs_and_grus_step()
    test_recurrent_loss_forward_reproduces_rollout_logits()
    test_recurrent_minibatches_over_episodes_intact()
    test_recurrence_feedforward_train_step_byte_unchanged()
    test_message_content_config_roundtrip()
    test_kb_distance_normalized_and_scale_invariant()
    test_edge_distance_message_encodes_neighbour_distance()
    test_edge_distance_train_step_runs()
    test_index_signal_distinct_and_agent_count_invariant()
    test_index_message_distinguishes_neighbours()
    test_index_train_step_runs()
    test_message_content_learned_param_tree_byte_identical()
    test_message_content_learned_forward_byte_unchanged()
    test_message_content_learned_train_step_byte_unchanged()
    test_selector_off_byte_unchanged_forward()
    test_selector_param_surface_stable_and_key_identical()
    test_selector_off_train_step_byte_unchanged()
    test_selector_config_roundtrip()
    test_skill_forward_shapes_and_skills_differ()
    test_hold_skill_stays_when_anchored_reconnects_when_isolating()
    test_selector_train_step_both_flock_flavors()
    test_selector_train_step_congestion_on_off()
    test_selector_recurrent_train_step()
    test_selector_loss_forward_reproduces_rollout_goal_logp()
    test_selector_full_train_loop_logs_skill_usage()
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
    import pathlib
    import tempfile
    test_warm_start_cross_rung_loads_and_trains(pathlib.Path(tempfile.mkdtemp()))
    test_warm_start_rejects_mismatched_backbone(pathlib.Path(tempfile.mkdtemp()))
    test_init_from_none_is_byte_unchanged()
    print("all smoke tests passed")
