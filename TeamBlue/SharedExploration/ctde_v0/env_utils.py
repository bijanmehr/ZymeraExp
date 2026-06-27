"""Env construction + reward composition + λ₂ oracle + KB-adjacency + metrics.

The lab/experiment boundary (agent_architecture.md): ``zymera`` runs the WORLD
only and is reward-agnostic; the experiment composes the scalar reward HERE from
the env's UNWEIGHTED per-term magnitudes in ``info["reward_terms"]``, re-weighting
coverage / connectivity / collision per the config (Reward block).

The auxiliary supervision target is the simulator's true Fiedler value
``_lambda2(world.body.position, comm_r, sharp)`` (missions_terms) — a scalar per
step, broadcast to all agents (one true λ₂ for the team). Grading uses the same
oracle: connectivity-% = fraction of steps with true λ₂ > τ.

This module also exposes the **comm-graph adjacency** the GNN-KB fuses over
(``kb_adjacency``: in-range neighbours at ``comm_r``, diagonal cleared) and the
**degree statistics** the SizeShiftReg-style regularizer penalizes.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import zymera
from zymera.missions_terms import _lambda2

from .config import CTDEConfig


# =============================================================================
# Env
# =============================================================================


def build_env(cfg: CTDEConfig):
    """Construct the comm-coverage env from the config's World block (recipe
    defaults supply the reward TERMS; we re-weight their magnitudes ourselves).

    When ``reward_anti_overlap == 'on'`` we append a **zero-weight** ``overlap``
    term (``same_step_overlap``) so the env populates ``info['reward_terms']
    ['overlap']`` — :func:`compose_reward` then re-weights it. Zero weight keeps the
    env's own (unused) scalar reward identical; we always compose the reward here.
    """
    w = cfg.world
    terms = None
    if cfg.reward_anti_overlap == "on":
        # default coverage/connectivity/collision terms + a 0-weight overlap probe.
        from zymera.missions_terms import DEFAULT_TERMS
        terms = list(DEFAULT_TERMS) + [("overlap", 0.0)]
    env = zymera.make(
        w.recipe,
        grid=w.grid,
        n_agents=w.n_agents,
        comm_r=w.comm_r,
        sense_r=w.sense_r,
        cover_r=w.cover_r,
        n_obstacles=w.n_obstacles,
        spawn_radius=w.spawn_radius,   # None -> scatter spawn inside the recipe
        max_steps=None,                # horizon controlled by the fixed-length scan
        terms=terms,                   # None -> recipe DEFAULT_TERMS (v0 unchanged)
    )
    comm_r = int(env.channel.topology.radius)
    assert comm_r == w.comm_r, (comm_r, w.comm_r)
    return env


# =============================================================================
# Reward composition (coverage + connectivity), reward engineering in-experiment
# =============================================================================


def compose_reward(reward_terms: dict, world, cfg: CTDEConfig,
                   lambda2_penalty: jax.Array | None = None) -> jax.Array:
    """(N,) scalar reward from the env's unweighted per-term (N,) magnitudes.

    base_i = w_cov*coverage_i + w_conn*connectivity_i + w_coll*collision_i
    (collision weight negative -> a penalty). When ``Reward.normalized`` is set,
    the coverage term is divided by the free-cell count (fractional coverage).
    When the soft-λ mechanism is active, ``lambda2_penalty`` (a shared scalar
    shortfall) is subtracted with weight ``Reward.soft_lambda_penalty``.
    """
    r = cfg.reward
    cov = reward_terms["coverage"]
    if r.normalized:
        free = jnp.maximum((~world.wall).sum().astype(jnp.float32), 1.0)
        cov = cov / free
    out = (
        r.w_coverage * cov
        + r.w_connectivity * reward_terms["connectivity"]
        + r.w_collision * reward_terms["collision"]
    )
    if lambda2_penalty is not None:
        out = out - r.soft_lambda_penalty * lambda2_penalty
    # Anti-overlap (Increment-1): penalize cells my footprint shares with a
    # teammate THIS step (same_step_overlap) -> rewards non-redundant coverage.
    # Only present when build_env appended the 0-weight 'overlap' probe term.
    if cfg.reward_anti_overlap == "on" and "overlap" in reward_terms:
        out = out - cfg.anti_overlap_weight * reward_terms["overlap"]
    return out.astype(jnp.float32)


# =============================================================================
# True λ₂ oracle (aux target + grader) and the KB comm-graph
# =============================================================================


def true_lambda2(position: jax.Array, cfg: CTDEConfig) -> jax.Array:
    """Scalar true Fiedler value of the soft comm-graph at ``position`` (N,2)."""
    return _lambda2(position, cfg.world.comm_r, cfg.connectivity.lambda2_sharp)


def kb_adjacency(position: jax.Array, cfg: CTDEConfig) -> jax.Array:
    """(N,N) bool — in-range neighbours at ``comm_r`` with the diagonal CLEARED.

    This is the comm graph the GNN-KB message-passing fuses over (the formalism's
    *bridge*). Chebyshev disk, derived from positions — matches the env's
    DiskTopology / the true-λ₂ soft graph support.
    """
    n = position.shape[0]
    d = jnp.max(jnp.abs(position[:, None, :] - position[None, :, :]), axis=-1)
    adj = d <= cfg.world.comm_r
    return adj & ~jnp.eye(n, dtype=bool)


def degree_stats(position: jax.Array, cfg: CTDEConfig) -> jax.Array:
    """(N,) float32 per-node in-range degree (neighbour count) over the comm
    graph — the per-node statistic the SizeShiftReg-style regularizer watches.
    """
    return kb_adjacency(position, cfg).sum(-1).astype(jnp.float32)


# =============================================================================
# Coverage metric (campaign definition)
# =============================================================================


def coverage_fraction_free(world, cfg: CTDEConfig) -> jax.Array:
    """Covered FREE cells / free cells (the campaign coverage-% definition).
    Falls back to all-cells when there are no walls (free == all)."""
    del cfg
    covered = world.covered                       # (H, W) bool
    free = ~world.wall                            # (H, W) bool
    num = (covered & free).sum().astype(jnp.float32)
    den = jnp.maximum(free.sum().astype(jnp.float32), 1.0)
    return num / den
