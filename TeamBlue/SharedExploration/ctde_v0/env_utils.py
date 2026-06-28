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

from . import controller as _ctrl
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
                   lambda2_penalty: jax.Array | None = None,
                   congestion_penalty: jax.Array | None = None) -> jax.Array:
    """(N,) scalar reward from the env's unweighted per-term (N,) magnitudes.

    base_i = w_cov*coverage_i + w_conn*connectivity_i + w_coll*collision_i
    (collision weight negative -> a penalty). When ``Reward.normalized`` is set,
    the coverage term is divided by the free-cell count (fractional coverage).
    When the soft-λ mechanism is active, ``lambda2_penalty`` (a shared scalar
    shortfall) is subtracted with weight ``Reward.soft_lambda_penalty``.

    When ``Reward.barrier_weight > 0`` the per-agent **connectivity-FLOOR barrier**
    (:func:`connectivity_barrier`, read off ``world.body.position`` — the SAME source
    true λ₂ / :func:`local_edge_margin` use) is SUBTRACTED. It composes with every
    other connectivity mechanism (it is NOT a replacement). At the default
    ``barrier_weight == 0`` the branch is skipped entirely, so the composed reward is
    byte-identical to the pre-barrier behaviour.
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
    # Free-market congestion price (selector + congestion on): a per-agent same-skill
    # crowding penalty computed in the rollout from the sampled skills (None otherwise,
    # so the branch is skipped and the reward is byte-unchanged).
    if congestion_penalty is not None:
        out = out - cfg.congestion_weight * congestion_penalty
    # Anti-overlap (Increment-1): penalize cells my footprint shares with a
    # teammate THIS step (same_step_overlap) -> rewards non-redundant coverage.
    # Only present when build_env appended the 0-weight 'overlap' probe term.
    if cfg.reward_anti_overlap == "on" and "overlap" in reward_terms:
        out = out - cfg.anti_overlap_weight * reward_terms["overlap"]
    # Connectivity-FLOOR barrier ("Hyper-Singularity"): a capped per-agent wall at the
    # disconnection edge, COMPOSED with whatever else is active. weight==0 -> skipped
    # entirely (no op added; out byte-identical). k=barrier_weight is inside the term.
    if r.barrier_weight > 0:
        out = out - connectivity_barrier(world.body.position, cfg)
    return out.astype(jnp.float32)


# =============================================================================
# True λ₂ oracle (aux target + grader) and the KB comm-graph
# =============================================================================


def true_lambda2(position: jax.Array, cfg: CTDEConfig) -> jax.Array:
    """Scalar true Fiedler value of the soft comm-graph at ``position`` (N,2)."""
    return _lambda2(position, cfg.world.comm_r, cfg.connectivity.lambda2_sharp)


def local_edge_margin(position: jax.Array, cfg: CTDEConfig) -> jax.Array:
    """(N,) float32 — the PER-AGENT "you're at the edge of comms range" signal.

    Where ``true_lambda2`` is a GLOBAL scalar (the team's λ₂ floor) broadcast
    identically to every agent — so no single agent knows IT is the one stretching
    the bridge — this is its LOCAL, per-agent counterpart: each agent reads only its
    OWN incident-edge mass and is penalized for the shortfall against a target
    degree. An agent comfortably surrounded by in-range teammates scores ≈0; one
    drifting toward the edge of its comm range (links approaching / crossing
    ``comm_r``) sees its penalty ramp up. It is ANTICIPATORY (the soft edge weight
    decays smoothly as a link nears ``comm_r``, so it fires BEFORE the link breaks)
    and partial-observability-native (computable from an agent's own neighbourhood).

    Construction — the SAME soft incident-edge mass the relay tool maximizes
    (:func:`controller._local_conn_score`, reused so the signal and the relay
    anchor agree):

      w_ij      = sigmoid(sharp · (comm_r − cheby_dist_ij))   for j ≠ i
      soft_deg_i = Σ_{j≠i} w_ij                               (soft neighbour count)
      p_i        = relu(degree_target − soft_deg_i)           (the shortfall)

    where ``sharp = cfg.connectivity.lambda2_sharp``, ``comm_r = cfg.world.comm_r``
    and ``degree_target = cfg.mission_safety.degree_target``. The result is the
    per-agent shortfall in exactly the soft-degree the relay maximizes — NOT
    averaged/broadcast: ``p_i`` is agent i's own margin, so the rollout can charge
    the stretching agent specifically. Pure JAX (vmap/scan/jit-safe).
    """
    soft_deg = _ctrl._local_conn_score(
        position, cfg.world.comm_r, cfg.connectivity.lambda2_sharp
    )                                                          # (N,) soft degree
    target = jnp.asarray(cfg.mission_safety.degree_target, dtype=jnp.float32)
    return jax.nn.relu(target - soft_deg).astype(jnp.float32)  # (N,) per-agent margin


def connectivity_barrier(position: jax.Array, cfg: CTDEConfig) -> jax.Array:
    """(N,) float32 — the per-agent **connectivity FLOOR barrier** ("Hyper-Singularity").

    A one-sided interior-point wall on each agent's NEAREST-NEIGHBOUR distance: it is
    EXACTLY 0 while the agent is comfortably linked (silent in the safe zone), rises
    smoothly as that agent's closest teammate drifts toward the edge of comm range,
    and saturates at a finite ``cap`` ("almost infinity") at / past the break point —
    so the rollout feels an explosive-but-finite push BEFORE the link snaps. It is a
    standalone, config-knobbed reward TERM that COMPOSES with every other connectivity
    mechanism (``conn_signal`` / ``mechanism``); it does NOT replace any of them. When
    ``Reward.barrier_weight == 0`` (the default) it is identically 0 and a no-op.

    The signal is the LOCAL Chebyshev nearest-neighbour distance — the SAME metric as
    the comm graph (:func:`controller._cheby` / :func:`kb_adjacency`), so the wall and
    the link agree on "range":

      x_i = min_{j != i} cheby_dist(i, j)         (self masked with +inf before the min)

    The barrier is the user's formula, made RL-safe (the literal pole at ``M`` is
    GUARDED so there is no inf/nan for ANY x, including ``x_i == M`` exactly and
    ``x_i > M`` / a lone agent's ``x_i = +inf``):

      raw(x)  = barrier_weight * relu(x - a)^2 / (M - x)^p     (0 for x<=a; pole at x=M)
      xc_i    = minimum(x_i, M - eps)             eps=1e-3, so (M - xc) in [eps, .] > 0
      f_i     = minimum( barrier_weight * relu(xc_i - a)^2 / (M - xc_i)^p , cap )
      f_i     = cap         where  x_i >= M        (link already broken / agent isolated)

    Here ``a = barrier_a`` (launch point), ``M = barrier_M`` (the wall / break range),
    ``p = barrier_p`` (explosion power), ``cap = barrier_cap`` (the finite ceiling) and
    ``barrier_weight`` IS the ``k`` of the formula (already folded in). The ``relu(·)^2``
    is the user's ``(x - a + |x - a|)^2 / 4`` written via the ReLU identity. Result is
    finite and in ``[0, cap]`` for every x. Pure JAX (vmap/scan/jit-safe).
    """
    r = cfg.reward
    k = jnp.asarray(r.barrier_weight, dtype=jnp.float32)
    a = jnp.asarray(cfg.barrier_a, dtype=jnp.float32)        # resolved (None -> comm_r*0.6)
    M = jnp.asarray(cfg.barrier_M, dtype=jnp.float32)        # resolved (None -> comm_r)
    p = jnp.asarray(r.barrier_p, dtype=jnp.float32)
    cap = jnp.asarray(r.barrier_cap, dtype=jnp.float32)
    eps = jnp.asarray(1e-3, dtype=jnp.float32)

    n = position.shape[0]
    # Chebyshev pairwise distance (same metric as the comm graph / controller._cheby).
    d = jnp.max(jnp.abs(position[:, None, :] - position[None, :, :]),
                axis=-1).astype(jnp.float32)                          # (N,N)
    # mask self with +inf so a lone agent yields x_i=+inf -> caught by the x>=M branch.
    # (jnp.where, NOT eye*inf: 0*inf would be NaN on the OFF-diagonal and poison the min.)
    d = jnp.where(jnp.eye(n, dtype=bool), jnp.inf, d)                # (N,N), diag +inf
    x = jnp.min(d, axis=-1)                                           # (N,) nearest-nbr dist

    # GUARD the pole: clamp x below M so the denominator (M - xc) >= eps > 0 (finite,
    # never 0/negative), THEN cap. x >= M (incl. x == M exactly and the +inf isolate)
    # is forced to the ceiling regardless of the clamped value.
    xc = jnp.minimum(x, M - eps)                                      # (N,) in (-inf, M-eps]
    raw = k * jax.nn.relu(xc - a) ** 2 / (M - xc) ** p                # (N,) finite
    f = jnp.minimum(raw, cap)                                         # (N,) in [0, cap]
    f = jnp.where(x >= M, cap, f)                                     # broken link -> cap
    return f.astype(jnp.float32)                                      # (N,)


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


def kb_distance(position: jax.Array, cfg: CTDEConfig) -> jax.Array:
    """(N,N) float32 — the comm-graph sender→receiver Chebyshev distance, NORMALIZED
    by ``comm_r`` (so in-range edges land in [0,1]) with the diagonal CLEARED to 0.

    This is the SAME Chebyshev distance ``kb_adjacency`` thresholds (the comm graph /
    DiskTopology metric), exposed for the GNN-KB ``message_content`` modes that append a
    per-edge geometry channel to each message (``nets.MPLayer`` / ``edge_distance``). It
    is normalized by ``comm_r`` (NOT raw cells) so a model trained @16²/4 reads the same
    edge geometry @32²/10 — SCALE-INVARIANT. The receiver multiplies it by the (boolean)
    adjacency, so out-of-range entries (> 1 here) never enter the aggregation.
    Pure JAX (vmap/scan/jit-safe).
    """
    n = position.shape[0]
    d = jnp.max(jnp.abs(position[:, None, :] - position[None, :, :]),
                axis=-1).astype(jnp.float32)                              # (N,N) cheby
    d = d / jnp.maximum(jnp.asarray(cfg.world.comm_r, jnp.float32), 1.0)  # normalize -> [0,1] in-range
    return jnp.where(jnp.eye(n, dtype=bool), 0.0, d).astype(jnp.float32)  # (N,N), diag 0


def degree_stats(position: jax.Array, cfg: CTDEConfig) -> jax.Array:
    """(N,) float32 per-node in-range degree (neighbour count) over the comm
    graph — the per-node statistic the SizeShiftReg-style regularizer watches.
    """
    return kb_adjacency(position, cfg).sum(-1).astype(jnp.float32)


def skill_congestion(skill_idx: jax.Array, position: jax.Array,
                     cfg: CTDEConfig) -> jax.Array:
    """(N,) float32 — the FREE-MARKET congestion price: for each agent, the number of its
    IN-RANGE neighbours that chose the SAME skill this step.

    Choosing a crowded skill (one many neighbours also picked) costs more, so the team
    spreads across the skill library instead of collapsing into one mode — the decentralized,
    learned-against anti-collapse force (the price is subtracted from the reward in
    :func:`compose_reward`, weighted by ``congestion_weight``). It is LOCAL (reads only the
    comm neighbourhood) and emergent — NOT a global auction. Pure JAX (vmap/scan/jit-safe):

      adj_ij   = in-range neighbour (Chebyshev ≤ comm_r, i ≠ j; ``kb_adjacency``)
      same_ij  = skill_i == skill_j
      price_i  = Σ_j adj_ij · same_ij                 (same-skill neighbour count)
    """
    adj = kb_adjacency(position, cfg)                                # (N,N) in-range, diag 0
    same = skill_idx[:, None] == skill_idx[None, :]                  # (N,N) same skill
    return (adj & same).sum(-1).astype(jnp.float32)                 # (N,) crowding price


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
