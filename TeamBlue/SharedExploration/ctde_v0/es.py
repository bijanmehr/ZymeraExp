"""Evolution-Strategies (ES) coexistence trainer — evolve the SELECTOR, gradient the executor.

This module is the ES half of a **two-level (feudal) cognition** agent: a SMALL *selector*
head (it picks among skills / sets a sub-goal) sits on top of a LARGE gradient-trained
*executor* (the LPAC backbone + heads in ``nets.Actor``, trained by CTDE-MAPPO in ``ppo.py``).
The agreed division of labour is **MERL / feudal-evolutionary**:

  * **gradient / CTDE** trains the *executor* on a dense, per-step signal (PPO on the goal /
    move policy, with a centralized critic);
  * **ES evolves the *selector*** by **TEAM FITNESS** — the whole-mission return (mean
    episode return over a rollout), the sparse team-level credit signal that gradient
    methods cannot easily assign to a tiny discrete-ish selector.

Because ES and the gradient touch **disjoint parameters** (ES → the selector head's leaves,
gradient → everything else), the two optimizers *compose*: this is the "CTDE+ES coexistence"
of the plan. This file is the **ES machinery only**, written against an **injectable**
interface (``eval_fn`` / ``gradient_step_fn`` / ``sync_in`` / ``sync_out``) so it has **no
dependency on the selector's internals** — the integrator wires the real fitness + selector
get/set + ppo step later (see the wiring note at the bottom of the docstring). To keep this
file import-safe while the selector core / nets are being edited in parallel, it imports
**only** ``jax / jax.numpy / equinox / optax / numpy`` at module level; any ppo/nets wiring
is left to the integrator (or done lazily inside a closure the integrator supplies).

WHY this split is the right one
-------------------------------
* The **selector is small** and its objective is **sparse + non-differentiable-friendly**
  (team return; "which skill, when"). ES is **black-box**: it needs only a scalar fitness,
  ignores non-differentiability, and — crucially for a swarm that can collapse into a
  *huddle* — its Gaussian search provides **huddle-escape / plateau-escape** that a local
  gradient cannot (it perturbs the selector globally and selects by team outcome). The
  antithetic + rank-shaped OpenAI-ES estimator is a low-variance gradient *estimate* of that
  sparse fitness, so the selector still moves smoothly.
* The **executor is large** and has a **dense, well-shaped per-step signal** (PPO advantages
  on the goal/move policy). Gradient descent is *far* more sample-efficient there than ES
  would be on millions of parameters. So we keep the cheap dense signal where it works and
  spend the expensive sparse team signal only on the few selector weights that need it.
* They **coexist** rather than fight because they optimize **disjoint leaves of the same
  Actor**: ``merl_coexist`` interleaves rounds — gradient trains the executor, the
  gradient-current selector is **injected** into the ES mean ("inject the learner", MERL),
  ES evolves it by team fitness, and the evolved selector is **written back** into the train
  state for the next gradient round.

References
----------
* Salimans, Ho, Chen, Sidor, Sutskever (2017), *Evolution Strategies as a Scalable
  Alternative to Reinforcement Learning* — the OpenAI-ES estimator used by ``kind='nes'``:
  antithetic Gaussian sampling + **rank-based fitness shaping**, update
  ``θ ← θ + lr/(pop·σ) · Σ_i f̃_i · ε_i``.
* Pourchot & Sigaud (2019), *CEM-RL: Combining Evolutionary and Gradient-Based Methods for
  Policy Search* — the cross-entropy-method side (``kind='cem'``: keep the elite fraction,
  re-fit the mean, shrink the covariance) and the template for combining a CEM population
  with a gradient learner.
* Khadka & Tumer (2019) / Khadka et al. (2019), *Evolutionary Reinforcement Learning* /
  *Collaborative Evolutionary RL (MERL)* — the interleave skeleton in ``merl_coexist``:
  evolve one part by sparse team fitness, train the other by gradient on a dense signal, and
  periodically **inject the gradient learner into the evolutionary population** so the two
  reinforce instead of diverge.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree


# =============================================================================
# Param helpers — flatten / inject the (inexact-array) leaves of a module
# =============================================================================


def flatten(pytree: Any):
    """Ravel any pytree to a 1-D parameter vector and return ``(theta, unflatten)``.

    ``theta`` is a 1-D ``jnp.float`` array concatenating every leaf (in tree order);
    ``unflatten(theta) -> pytree`` rebuilds the original structure. Thin wrapper over
    :func:`jax.flatten_util.ravel_pytree` so the ES core only ever sees flat vectors.
    """
    theta, unflatten = ravel_pytree(pytree)
    return theta, unflatten


def module_theta(module: eqx.Module) -> jax.Array:
    """Extract the **inexact-array** (float/complex) leaves of an Equinox ``module`` as a
    single 1-D vector ``theta``.

    Uses ``eqx.partition(module, eqx.is_inexact_array)`` to split the trainable float leaves
    from everything else (ints, bools, static fields, callables), then ravels *only* the
    float part. This is the exact leaf set the gradient/ES touch (mirrors
    ``ppo._perturb_actor``'s ``eqx.is_inexact_array`` partition), so a ``theta`` produced
    here round-trips cleanly through :func:`set_module_theta`.
    """
    params, _static = eqx.partition(module, eqx.is_inexact_array)
    theta, _unflatten = ravel_pytree(params)
    return theta


def set_module_theta(module: eqx.Module, theta: jax.Array) -> eqx.Module:
    """Return a copy of ``module`` with its **inexact-array** leaves replaced by ``theta``.

    Inverse of :func:`module_theta`: partition off the float leaves, rebuild them from the
    flat ``theta`` with the matching ``unflatten``, and ``eqx.combine`` them back with the
    untouched static part. ``set_module_theta(m, module_theta(m))`` is the identity.
    """
    params, static = eqx.partition(module, eqx.is_inexact_array)
    _flat, unflatten = ravel_pytree(params)
    new_params = unflatten(theta)
    return eqx.combine(new_params, static)


# =============================================================================
# ES configuration
# =============================================================================


@dataclass
class ESConfig:
    """Hyper-parameters for the ES step / coexistence loop.

    * ``pop_size``     — population size (number of fitness evals per ``es_step``). With
      antithetic sampling the actual evaluations are ``2 * (pop_size // 2)``.
    * ``sigma``        — Gaussian perturbation std (search radius around the mean ``theta``).
    * ``lr``           — OpenAI-ES learning rate (``kind='nes'`` only).
    * ``kind``         — ``'nes'`` (OpenAI-ES rank-shaped gradient estimate) or ``'cem'``
      (cross-entropy method: elite mean + covariance shrink).
    * ``elite_frac``   — fraction of the population kept as elites (``kind='cem'`` only).
    * ``weight_decay`` — optional L2 pull of ``theta`` toward 0 each step (0 disables).
    """
    pop_size: int = 16
    sigma: float = 0.05
    lr: float = 0.05
    kind: str = "nes"          # {'nes','cem'}
    elite_frac: float = 0.25
    weight_decay: float = 0.0


# =============================================================================
# Fitness shaping helpers
# =============================================================================


def _rank_normalize(fitness: jax.Array) -> jax.Array:
    """OpenAI-ES rank-based fitness shaping → centered weights in ``[-0.5, 0.5]``.

    Replace each fitness by its **rank** (0 = worst … P-1 = best) mapped linearly onto
    ``[-0.5, 0.5]`` and mean-centered. This makes the update invariant to the scale /
    outliers of the raw return (Salimans 2017 §"fitness shaping") — only the *ordering* of
    the population matters, which is exactly right for a noisy team-return fitness. A
    degenerate population (P ≤ 1) maps to all-zeros (no update).
    """
    fitness = jnp.asarray(fitness, jnp.float32)
    p = fitness.shape[0]
    if p <= 1:
        return jnp.zeros_like(fitness)
    # argsort-of-argsort = rank of each element (ascending: best gets the largest rank).
    order = jnp.argsort(fitness)
    ranks = jnp.argsort(order).astype(jnp.float32)          # 0..P-1
    shaped = ranks / (p - 1) - 0.5                          # -> [-0.5, 0.5], centered
    return shaped


def _antithetic_noise(dim: int, half: int, key) -> jax.Array:
    """``(2*half, dim)`` antithetic Gaussian noise: the first ``half`` rows are i.i.d.
    ``N(0, I)`` (the ``ε`` directions), the next ``half`` are their negations ``−ε``.

    Antithetic (mirrored) sampling pairs each ``+ε`` with ``−ε`` so the population mean is
    exactly 0 and the OpenAI-ES gradient estimate has lower variance (Salimans 2017). These
    are the **unscaled** directions ``ε`` (the perturbation applied to ``theta`` is
    ``σ·ε``); the ``/σ`` in the NES update divides it back out.
    """
    eps = jax.random.normal(key, (half, dim), dtype=jnp.float32)   # (half, dim)
    return jnp.concatenate([eps, -eps], axis=0)                    # (2*half, dim)


def _evaluate_population(theta: jax.Array, noise: jax.Array, sigma: float,
                         eval_fn: Callable[[jax.Array], float]) -> jax.Array:
    """Evaluate ``eval_fn`` on every perturbed candidate ``θ + σ·εᵢ`` — a **Python loop**.

    ``eval_fn`` runs rollouts and is **impure** (host RNG, env stepping), so it is NEVER
    vmapped; we loop over the population on the host and stack the scalar fitnesses. Each
    candidate is converted to a concrete device array before the call so ``eval_fn`` sees a
    materialized ``theta`` (not a traced value).
    """
    perturbed = theta[None, :] + sigma * noise                    # (P, dim)
    fits = [float(eval_fn(perturbed[i])) for i in range(perturbed.shape[0])]
    return jnp.asarray(fits, dtype=jnp.float32)                   # (P,)


# =============================================================================
# One ES step (OpenAI-ES / CEM)
# =============================================================================


def es_step(theta: jax.Array, eval_fn: Callable[[jax.Array], float],
            es_cfg: ESConfig, key) -> tuple[jax.Array, dict]:
    """One Evolution-Strategies update of the mean parameter vector ``theta``.

    Antithetic Gaussian sampling draws ``half = pop_size // 2`` direction pairs
    ``{+ε, −ε}`` (so ``P = 2·half`` candidates), evaluates each at ``θ + σ·ε`` via the
    injected ``eval_fn`` (HIGHER fitness = better), and updates ``theta`` by one of:

    * ``kind == 'nes'`` (OpenAI-ES, Salimans 2017): rank-shape the fitnesses to
      ``[-0.5, 0.5]`` (``_rank_normalize``) and take the estimated natural-gradient step
      ``θ ← θ + (lr / (P·σ)) · Σ_i f̃_i · ε_i`` (with the unscaled directions ``ε_i``).
      Optional ``weight_decay`` adds a ``−lr·wd·θ`` pull toward 0. ``sigma`` is unchanged.
    * ``kind == 'cem'`` (CEM-RL, Pourchot 2019): keep the top ``elite_frac`` candidates by
      raw fitness, set ``θ ← mean(elites)``, and **shrink** ``sigma`` toward the per-dim
      elite std (``new_sigma`` reported in ``info``). ``lr`` / rank-shaping are unused.

    Args:
      theta:   (D,) current mean parameter vector.
      eval_fn: ``eval_fn(theta_i) -> float`` fitness (impure; runs rollouts). Not vmapped.
      es_cfg:  :class:`ESConfig`.
      key:     PRNG key for the population noise.

    Returns:
      ``(theta_next, info)`` where ``info`` carries ``best_fitness``, ``mean_fitness`` (and
      ``sigma`` for ``kind=='cem'`` — the shrunk std).
    """
    theta = jnp.asarray(theta, jnp.float32)
    dim = theta.shape[0]
    half = max(1, int(es_cfg.pop_size) // 2)
    sigma = float(es_cfg.sigma)

    noise = _antithetic_noise(dim, half, key)                     # (P, dim), P = 2*half
    fitness = _evaluate_population(theta, noise, sigma, eval_fn)  # (P,) host loop

    best_fitness = float(jnp.max(fitness))
    mean_fitness = float(jnp.mean(fitness))

    if es_cfg.kind == "cem":
        # ---- Cross-Entropy Method: elite mean + covariance (sigma) shrink ----
        p = fitness.shape[0]
        n_elite = max(1, int(round(float(es_cfg.elite_frac) * p)))
        # top-n_elite by fitness (descending): take the last n_elite of the ascending sort.
        order = jnp.argsort(fitness)                              # ascending
        elite_idx = order[-n_elite:]                             # the n_elite best
        elite_perturbed = theta[None, :] + sigma * noise[elite_idx]   # (n_elite, dim)
        theta_next = jnp.mean(elite_perturbed, axis=0)           # new mean = elite mean
        # CEM covariance shrink: next std = std of the elite candidates (per-dim, then mean
        # to a scalar so ESConfig.sigma stays a scalar). Guard the n_elite==1 degenerate
        # case (std == 0) by keeping the old sigma.
        if n_elite > 1:
            elite_std = float(jnp.mean(jnp.std(elite_perturbed, axis=0)))
        else:
            elite_std = sigma
        info = {"best_fitness": best_fitness, "mean_fitness": mean_fitness,
                "sigma": elite_std}
        return theta_next, info

    # ---- default: OpenAI-ES (NES) rank-shaped antithetic gradient estimate ----
    shaped = _rank_normalize(fitness)                            # (P,) in [-0.5, 0.5]
    # estimated gradient: (1/(P·σ)) Σ_i f̃_i ε_i   (ε_i = unscaled directions = noise rows).
    grad = (shaped[:, None] * noise).sum(axis=0) / (noise.shape[0] * sigma)   # (dim,)
    step = float(es_cfg.lr) * grad
    if es_cfg.weight_decay:
        step = step - float(es_cfg.lr) * float(es_cfg.weight_decay) * theta
    theta_next = theta + step
    info = {"best_fitness": best_fitness, "mean_fitness": mean_fitness}
    return theta_next, info


# =============================================================================
# MERL coexistence — interleave gradient(executor) with ES(selector)
# =============================================================================


def merl_coexist(*, theta0: jax.Array,
                 eval_fn: Callable[[jax.Array], float],
                 gradient_step_fn: Callable[[Any], Any],
                 sync_in: Callable[[Any, jax.Array], jax.Array],
                 sync_out: Callable[[Any, jax.Array], Any],
                 n_outer: int,
                 grad_steps_per_outer: int,
                 es_cfg: ESConfig,
                 key,
                 gstate0: Any = None) -> tuple[jax.Array, Any, list]:
    """The MERL / feudal-evolutionary **interleave skeleton** — coexist gradient + ES.

    Each outer round (Khadka 2019 MERL):

      (a) **train the executor**: call ``gstate = gradient_step_fn(gstate)``
          ``grad_steps_per_outer`` times (e.g. one ``ppo.train_step`` on the executor each);
      (b) **inject the learner**: ``theta = sync_in(gstate, theta)`` — read the
          gradient-current selector OUT of the train state INTO the ES mean, so ES searches
          around the policy gradient has reached (the MERL "inject the gradient learner");
      (c) **evolve the selector**: ``theta, info = es_step(theta, eval_fn, es_cfg, key_t)``
          — one ES update of the selector by team fitness;
      (d) **write back**: ``gstate = sync_out(gstate, theta)`` — push the evolved selector
          BACK into the train state so the next gradient round uses it.

    ``(outer, info)`` is appended to ``history`` each round. Every collaborator is an
    **injected** callable so this file stays decoupled from ppo/nets (the integrator supplies
    the real ones; see the module docstring's wiring note). ``gstate0`` defaults to ``None``
    when the caller threads the train state purely through ``gradient_step_fn`` / ``sync_*``
    closures, but is accepted explicitly so a real ``ppo.TrainState`` can be passed in.

    Args:
      theta0:               (D,) initial selector mean (ES parameters).
      eval_fn:              team-fitness eval ``eval_fn(theta) -> float`` (rollout return).
      gradient_step_fn:     ``gstate -> gstate`` one executor gradient step.
      sync_in:              ``(gstate, theta) -> theta`` inject the learner's selector → ES.
      sync_out:             ``(gstate, theta) -> gstate`` write the evolved selector → state.
      n_outer:              number of outer interleave rounds.
      grad_steps_per_outer: executor gradient steps per outer round.
      es_cfg:               :class:`ESConfig`.
      key:                  PRNG key (split per outer round for the ES population).
      gstate0:              optional initial gradient/train state (default ``None``).

    Returns:
      ``(theta_final, gstate_final, history)`` — ``history`` is a list of
      ``(outer_index, info_dict)`` of length ``n_outer``.
    """
    theta = jnp.asarray(theta0, jnp.float32)
    gstate = gstate0
    history: list = []
    k = key
    for outer in range(int(n_outer)):
        k, kt = jax.random.split(k)
        # (a) train the executor by gradient for a few steps.
        for _ in range(int(grad_steps_per_outer)):
            gstate = gradient_step_fn(gstate)
        # (b) inject the gradient-current selector into the ES mean.
        theta = sync_in(gstate, theta)
        # (c) evolve the selector by team fitness.
        theta, info = es_step(theta, eval_fn, es_cfg, kt)
        # (d) write the evolved selector back into the train state.
        gstate = sync_out(gstate, theta)
        history.append((outer, info))
    return theta, gstate, history
