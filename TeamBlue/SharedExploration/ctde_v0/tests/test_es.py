"""Unit tests for the ES coexistence trainer (``ctde_v0/es.py``) — CPU-only.

Covers the ES machinery in isolation (no ppo/nets dependency): the OpenAI-ES / CEM
``es_step`` on a toy quadratic fitness, the module-param flatten/inject round-trip, and the
``merl_coexist`` interleave skeleton with toy stand-ins.

Run from the SharedExploration working dir:
    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -m pytest \
        ctde_v0/tests/test_es.py -q
"""
import os; os.environ.setdefault("JAX_PLATFORMS", "cpu")
import sys, os as _os; sys.path.insert(0, _os.path.dirname(_os.path.dirname(_os.path.dirname(_os.path.abspath(__file__)))))

import equinox as eqx  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

from ctde_v0.es import (  # noqa: E402
    ESConfig, es_step, flatten, module_theta, set_module_theta, merl_coexist,
)


# A fixed small target the toy fitness pulls toward.
_TARGET = jnp.array([0.3, -0.5, 0.8, -0.2], dtype=jnp.float32)


def _toy_fitness(theta):
    """f(theta) = -sum((theta - target)^2)  — HIGHER (less negative) = closer to target.

    Impure-style scalar (returns a Python float) to match the ``eval_fn`` contract; runs no
    rollout, just the quadratic."""
    theta = jnp.asarray(theta, jnp.float32)
    return float(-jnp.sum((theta - _TARGET) ** 2))


# ---- (a) es_step improves fitness and drives theta -> target (nes AND cem) ----


def _run_es(kind, steps=40):
    """Run ``steps`` of ``es_step`` of the given kind on the toy quadratic from a fixed
    start; return ``(f_start, f_end, mean0, meanN, start_dist, end_dist)``.

    ``f_start`` / ``f_end`` are the **noise-free** fitness AT the mean ``theta`` (before /
    after the run) — the honest convergence signal for ``theta`` itself. The population-mean
    fitness in ``info`` is floored by the population spread ``σ`` (a population centred on the
    optimum still averages ``≈ −dim·σ²``), so we track the noise-free theta fitness for the
    "improved" check and use ``info['mean_fitness']`` only for the contract assertions.
    """
    cfg = ESConfig(pop_size=32, sigma=0.08, lr=0.15, kind=kind, elite_frac=0.25)
    theta = jnp.zeros_like(_TARGET)                       # start away from target
    start_dist = float(jnp.linalg.norm(theta - _TARGET))
    f_start = _toy_fitness(theta)
    key = jax.random.PRNGKey(0)
    mean0 = meanN = None
    for i in range(steps):
        key, sk = jax.random.split(key)
        theta, info = es_step(theta, _toy_fitness, cfg, sk)
        if i == 0:
            mean0 = info["mean_fitness"]
        meanN = info["mean_fitness"]
        # info contract (holds every step, both kinds)
        assert "best_fitness" in info and "mean_fitness" in info
        assert info["best_fitness"] >= info["mean_fitness"] - 1e-5
        if kind == "cem":
            assert "sigma" in info
    end_dist = float(jnp.linalg.norm(theta - _TARGET))
    f_end = _toy_fitness(theta)
    return f_start, f_end, mean0, meanN, start_dist, end_dist


def test_es_step_nes_improves_and_converges():
    """OpenAI-ES (kind='nes') improves the (noise-free) fitness of the mean over ~40 steps
    and drives ``theta`` toward the target (end distance well below the start distance)."""
    f_start, f_end, _m0, meanN, start_dist, end_dist = _run_es("nes")
    assert f_end > f_start                                 # theta's own fitness improved
    assert end_dist < start_dist * 0.6                     # closed most of the gap to target
    assert np.isfinite(meanN)


def test_es_step_cem_improves_and_converges():
    """CEM (kind='cem') improves the (noise-free) fitness of the mean over ~40 steps, drives
    ``theta`` toward the target, and reports a finite non-negative elite sigma."""
    f_start, f_end, _m0, meanN, start_dist, end_dist = _run_es("cem")
    assert f_end > f_start                                 # theta's own fitness improved
    assert end_dist < start_dist * 0.6                     # closed most of the gap to target
    assert np.isfinite(meanN)

    # the reported sigma should be a finite, non-negative scalar (covariance shrink).
    cfg = ESConfig(pop_size=32, sigma=0.08, lr=0.15, kind="cem", elite_frac=0.25)
    _, info = es_step(jnp.zeros_like(_TARGET), _toy_fitness, cfg, jax.random.PRNGKey(1))
    assert info["sigma"] >= 0.0 and np.isfinite(info["sigma"])


# ---- (b) flatten / module_theta / set_module_theta round-trip ---------------


def test_flatten_roundtrip():
    """``flatten`` ravels a pytree and ``unflatten`` rebuilds it exactly."""
    tree = {"a": jnp.arange(6.0).reshape(2, 3), "b": jnp.array([1.0, 2.0])}
    theta, unflatten = flatten(tree)
    assert theta.ndim == 1 and theta.shape[0] == 8
    rebuilt = unflatten(theta)
    assert jnp.allclose(rebuilt["a"], tree["a"])
    assert jnp.allclose(rebuilt["b"], tree["b"])


def test_module_theta_roundtrip_linear():
    """``module_theta`` / ``set_module_theta`` round-trip an ``eqx.nn.Linear``:
    get -> set -> get is the identity, and injecting a fresh theta then reading it back
    returns exactly that theta."""
    lin = eqx.nn.Linear(5, 3, key=jax.random.PRNGKey(0))
    theta = module_theta(lin)
    assert theta.ndim == 1
    # weight (3x5) + bias (3) = 18 inexact-array leaves.
    assert theta.shape[0] == 3 * 5 + 3

    # set(m, get(m)) leaves the params byte-identical (read back == original theta).
    lin_same = set_module_theta(lin, theta)
    assert jnp.allclose(module_theta(lin_same), theta)

    # inject a DIFFERENT theta, then read it back -> exactly the injected vector.
    new_theta = theta + 1.234
    lin_new = set_module_theta(lin, new_theta)
    assert jnp.allclose(module_theta(lin_new), new_theta)
    # and the static structure is preserved (still a usable Linear of the right shape).
    out = lin_new(jnp.ones((5,), jnp.float32))
    assert out.shape == (3,)


# ---- (c) merl_coexist runs the interleave skeleton with toy stand-ins -------


def test_merl_coexist_runs_with_toy_stand_ins():
    """``merl_coexist`` runs 3 outer rounds with toy injected collaborators and returns a
    history of the right length without error.

    Stand-ins:
      * gradient_step_fn = increment a dummy int train-state (stands for ppo.train_step);
      * sync_in / sync_out = identity passthroughs on theta / gstate;
      * eval_fn = the toy quadratic (team-fitness stand-in).
    """
    n_outer = 3
    grad_steps = 2

    def gradient_step_fn(gstate):                          # dummy executor "training"
        return gstate + 1

    def sync_in(gstate, theta):                            # learner -> ES mean (no-op)
        return theta

    def sync_out(gstate, theta):                           # evolved selector -> state (no-op)
        return gstate

    theta_final, gstate_final, history = merl_coexist(
        theta0=jnp.zeros_like(_TARGET),
        eval_fn=_toy_fitness,
        gradient_step_fn=gradient_step_fn,
        sync_in=sync_in,
        sync_out=sync_out,
        n_outer=n_outer,
        grad_steps_per_outer=grad_steps,
        es_cfg=ESConfig(pop_size=12, sigma=0.1, lr=0.2, kind="nes"),
        key=jax.random.PRNGKey(0),
        gstate0=0,
    )

    # history length == n_outer, each entry is (outer_index, info_dict).
    assert len(history) == n_outer
    for i, (outer, info) in enumerate(history):
        assert outer == i
        assert "best_fitness" in info and "mean_fitness" in info
        assert np.isfinite(info["mean_fitness"])

    # the dummy gradient state advanced by grad_steps_per_outer * n_outer.
    assert gstate_final == grad_steps * n_outer
    # the evolved selector is a finite vector of the original dimension.
    assert theta_final.shape == _TARGET.shape
    assert jnp.all(jnp.isfinite(theta_final))


# ---- also exercise merl_coexist with a REAL sync (module get/set) -----------


def test_merl_coexist_with_module_sync():
    """A slightly more realistic ``merl_coexist``: the train state IS an ``eqx.nn.Linear``
    (a selector stand-in); sync_in reads its theta, sync_out writes the evolved theta back.
    Verifies the get/set wiring the integrator will use against ``actor.selector_head``."""
    selector = eqx.nn.Linear(4, 4, key=jax.random.PRNGKey(0))

    def gradient_step_fn(gstate):                          # leave the executor untouched
        return gstate

    def sync_in(gstate, theta):                            # read selector params -> ES mean
        return module_theta(gstate)

    def sync_out(gstate, theta):                           # write evolved theta -> selector
        return set_module_theta(gstate, theta)

    # fitness on the module's flat params: pull the (16,) param vector toward zeros.
    def eval_fn(theta):
        return float(-jnp.sum(jnp.asarray(theta) ** 2))

    theta0 = module_theta(selector)
    theta_final, sel_final, history = merl_coexist(
        theta0=theta0,
        eval_fn=eval_fn,
        gradient_step_fn=gradient_step_fn,
        sync_in=sync_in,
        sync_out=sync_out,
        n_outer=6,
        grad_steps_per_outer=1,
        # CEM here: the point is the module get/set sync wiring (not convergence speed);
        # CEM reliably shrinks the 16-dim param norm in a handful of rounds.
        es_cfg=ESConfig(pop_size=16, sigma=0.1, lr=0.3, kind="cem", elite_frac=0.25),
        key=jax.random.PRNGKey(1),
        gstate0=selector,
    )

    assert len(history) == 6
    # sync_out actually wrote theta back into the module (read it out == theta_final).
    assert jnp.allclose(module_theta(sel_final), theta_final)
    # and the selector got pulled toward zero (final norm < initial norm).
    assert float(jnp.linalg.norm(theta_final)) < float(jnp.linalg.norm(theta0))
