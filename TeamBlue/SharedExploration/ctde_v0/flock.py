"""Flock — the L4 "repair connectivity" skill (the complement of the disperse tool).

The goal head picks 1 of K compass waypoints off the belief ``z`` (``nets.Actor``);
``nets.FrontierAttn`` (the *disperse* skill) biases those K logits toward the most
frontier-rich sector so the swarm SPREADS. This module is its antagonist: a *flock*
(connectivity-repair) skill that biases the SAME K goal-offsets back toward the
teammate whose comm link is most at risk of dropping out — the directional pull that
keeps the bridge of the formalism alive while the explorer pushes outward. Two flavors
share the K-offset / soft-sector-cosine interface:

  * :func:`scripted_flock_logits` — a hand-derived, parameter-free heuristic: for each
    agent it finds its at-risk (farthest in-range) neighbour and rewards the compass
    offset whose direction best aligns with the unit bearing toward that neighbour. It
    is the flock analogue of ``controller.relay_move`` (hold the bridge) expressed as
    additive GOAL-offset logits rather than a 1-step move.
  * :class:`FlockHead` — a tiny LEARNED flock: a single ``Linear(width, K)`` off the
    per-agent belief ``z``. Reads z ALONE (no absolute coordinate) so it is
    scale-invariant by construction and transfers up the scale ladder.

Both produce a ``(N, K)`` term to ADD to the goal logits (so a zero row / a disabled
gate is exactly the unmodified goal policy). Everything is pure JAX (vmap/jit-safe);
no Python branching on traced values — directional decisions use :func:`jnp.where`.

Design discipline (mirrors ``nets.py``):
  * SCALE-INVARIANT: the scripted scorer uses UNIT bearing directions and the unit
    compass directions (:func:`nets._compass_unit_dirs`) ONLY — no absolute cell
    coordinate or grid-size magnitude survives, so the SAME relative layout yields the
    same logits at any H, W or team size. The learned head reads only the belief z.
  * SOFT-SECTOR COSINE -> SCORE: like ``sector_frontier_features`` / ``compass_features``
    the per-offset score is the cosine of the (unit) target bearing against each (unit)
    compass direction, scaled by ``sharp`` — the higher the alignment, the higher the
    additive logit, sharpened by ``sharp`` exactly as the soft-sector modules do.
"""
from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp

from .nets import _compass_unit_dirs


# =============================================================================
# Scripted flock — parameter-free connectivity-repair goal-offset scorer
# =============================================================================


def scripted_flock_logits(position: jax.Array, K: int, comm_r: float,
                          sharp: float = 2.0) -> jax.Array:
    """(N, K) float32 additive goal-offset logits biasing each agent's goal toward
    REPAIRING its weakest (most at-risk) in-range comm link — the scripted flock skill.

    For agent ``i`` the at-risk link is the in-range neighbour ``j`` (Chebyshev distance
    ``d_ij <= comm_r``, ``j != i``) with the LARGEST distance, i.e. the teammate closest
    to falling out of comm range at ``comm_r``. The unit bearing toward that neighbour is
    matched (cosine) against the K compass directions (:func:`nets._compass_unit_dirs`,
    index 0 = "here"/zero vector); the offset that best aligns gets the highest logit. The
    score is the cosine scaled by ``sharp`` (the same soft-sector cosine->score shape as
    ``nets.sector_frontier_features`` / ``compass_features``), so larger ``sharp`` peaks
    the preference harder on the repair direction.

    Logit row for agent i (with ``û_i`` = unit bearing toward its at-risk neighbour and
    ``dir_k`` the unit compass directions)::

        logit[i, k] = sharp · (û_i · dir_k)            (cosine alignment, sharpened)

    The "here" offset (``dir_0 = 0``) scores 0 — a neutral baseline the directional
    offsets are measured against, so heading toward the at-risk teammate is rewarded
    over staying put whenever any neighbour is at risk.

    Fallbacks:
      * neighbours present but none "at risk" — the farthest in-range neighbour is still
        the target (a benign no-op: a close, safely-anchored neighbour simply yields a
        weak pull), so no special-case is needed.
      * NO in-range neighbour (isolated agent) — the row is ALL ZEROS: no preference, so
        the flock term leaves an isolated agent's goal policy untouched (REQUIRED). An
        isolated agent has nothing to repair *toward* from local comm information; the
        explorer / compass handle re-gathering.

    SCALE-INVARIANT by construction: only the UNIT bearing toward the at-risk neighbour
    and the UNIT compass directions enter — no absolute coordinate or grid-size magnitude
    survives, so a model using this term transfers across the scale ladder. Pure JAX
    (vmap/jit-safe): the isolated-agent and self-edge cases are handled with
    :func:`jnp.where`, never a Python branch on a traced value.

    Args:
      position: (N, 2) int/float agent cells (rows, cols).
      K: number of compass goal-offsets (matches the goal head's K; index 0 = "here").
      comm_r: comm range — neighbours are in range iff Chebyshev distance ``<= comm_r``.
      sharp: cosine sharpening factor (higher = a peakier pull on the repair offset).

    Returns:
      (N, K) float32 additive offset-logits (the term to ADD to the goal logits).
    """
    pos = position.astype(jnp.float32)                              # (N,2)
    n = pos.shape[0]

    diff = pos[None, :, :] - pos[:, None, :]                        # (N,N,2) i->j displacement
    cheb = jnp.max(jnp.abs(diff), axis=-1)                          # (N,N) Chebyshev distance
    eye = jnp.eye(n, dtype=bool)                                    # (N,N) self mask
    in_range = (cheb <= jnp.asarray(comm_r, jnp.float32)) & (~eye)  # (N,N) in-range neighbours (j!=i)

    # at-risk neighbour = the in-range neighbour with the LARGEST distance (closest to
    # dropping out at comm_r). Non-neighbours are pushed to -inf so they never win the
    # argmax; an isolated row is all -inf -> argmax returns index 0 (guarded out below).
    masked_d = jnp.where(in_range, cheb, -jnp.inf)                 # (N,N)
    tgt = jnp.argmax(masked_d, axis=-1)                            # (N,) index of at-risk nbr
    has_nbr = in_range.any(axis=-1)                                # (N,) any in-range neighbour?

    # unit bearing toward each agent's at-risk neighbour (displacement / its norm). Only
    # the ANGLE matters (scale-free); a zero-length bearing (degenerate) yields a zero
    # vector via the max(norm, eps) guard and so scores the "here" offset.
    bearing = diff[jnp.arange(n), tgt]                             # (N,2) i->at-risk displacement
    norm = jnp.sqrt((bearing ** 2).sum(-1, keepdims=True))         # (N,1)
    unit = bearing / jnp.maximum(norm, 1e-6)                       # (N,2) unit bearing

    dirs = _compass_unit_dirs(K)                                    # (K,2) unit compass dirs (0->here)
    cos = unit @ dirs.T                                            # (N,K) cosine alignment per offset
    logits = jnp.asarray(sharp, jnp.float32) * cos                 # (N,K) sharpened cosine score

    # isolated agents (no in-range neighbour) express NO preference -> all-zero row.
    return jnp.where(has_nbr[:, None], logits, 0.0).astype(jnp.float32)  # (N,K)


# =============================================================================
# Learned flock — tiny belief-conditioned connectivity-repair head
# =============================================================================


class FlockHead(eqx.Module):
    """A LEARNED flock skill — a tiny belief-conditioned connectivity-repair head.

    Maps each agent's per-agent belief ``z_i`` (W,) to ``K`` additive goal-offset logits
    via a single ``Linear(width, K)``, vmapped over the team. The belief z is the
    post-message-passing KB state (``nets.Backbone``), so the repair direction is learned
    from the FUSED neighbourhood the comm graph exposes — the learned counterpart of the
    scripted scorer's "head toward the at-risk teammate", here discovered from reward
    rather than hand-derived.

    Drop-in alongside the disperse tool (``nets.FrontierAttn``): both emit a ``(N, K)``
    term to ADD to ``goal_head(z)``, so a zero contribution is exactly the unmodified
    goal policy and PPO keeps sampling a goal from a distribution. Kept deliberately TINY
    (one Linear) and SCALE-INVARIANT — it reads ONLY the belief z (a fixed width W, never
    an absolute coordinate or a team-size-dependent quantity), so a model trained @16²/4
    transfers up the scale ladder exactly like the other belief-only heads. Pure JAX
    (vmap/jit-safe).

    Construction mirrors the other tool modules (``__init__(self, width, K, *, key)``):
      * ``head`` — ``Linear(W -> K)`` the per-agent offset-logit map.
    """
    head: eqx.nn.Linear
    K: int = eqx.field(static=True)
    width: int = eqx.field(static=True)

    def __init__(self, width: int, K: int, *, key):
        self.head = eqx.nn.Linear(width, K, key=key)
        self.K = int(K)
        self.width = int(width)

    def __call__(self, z: jax.Array) -> jax.Array:
        """(N, K) additive goal-offset logits from the per-agent belief ``z`` (N, W).

        Reads z ALONE (scale-invariant); returns the term to ADD to ``goal_head(z)``.
        Pure JAX (vmap over the N agents)."""
        return jax.vmap(self.head)(z)                              # (N,K)
