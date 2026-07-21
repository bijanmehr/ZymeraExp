"""Parity test for the --remat optimization (gradient checkpointing).

remat wraps the per-row actor forward inside the differentiated loss in ``jax.checkpoint``:
the backward pass RECOMPUTES the forward activations instead of storing them. This is
mathematically exact — it must produce IDENTICAL gradients, hence identical parameter updates.

The test runs ONE full ``ppo.train_step`` twice from the SAME init and SAME keys — once with
``loss.remat = False`` (v0), once with ``True`` — and compares the UPDATED (actor, critic, dual)
parameters leaf-by-leaf. collect() and init_state() are remat-independent, so any difference in
the post-step params is a difference in the gradient, i.e. a real (disqualifying) change.

    JAX_PLATFORMS=cpu PYTHONPATH=.:../../../FiedlerValueEstimation \
        <python> -m ctde_v0.tests.test_remat_parity
"""
from __future__ import annotations

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ctde_v0 import env_utils, ppo, train_ctde

# small but REAL: the running spec family (frontier_attn · collision-mask · lagrangian ·
# global_lambda2 · role split · SLAM belief · cover_r 0), shrunk (grid 12, horizon 20, 4 rollouts).
_ARGV = ["--grid", "12", "--n-agents", "4", "--comm-r", "5", "--rollouts", "4", "--horizon", "20",
         "--explorer-tool", "frontier_attn", "--collision-mask", "on", "--mechanism", "lagrangian",
         "--conn-signal", "global_lambda2", "--constraint-threshold", "0.5",
         "--role-picker", "expl_relay", "--w-coverage", "3",
         "--sense-walls", "--sense-free", "--boundary", "--cover-r", "0"]


def _one_step_params(cfg):
    """Init from a fixed key, run one train_step with a fixed key, return the updated
    (actor, critic, dual) inexact-array leaves + the loss log."""
    env = env_utils.build_env(cfg)
    key = jax.random.PRNGKey(0)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, key)
    sk = jax.random.split(key, 2)[1]           # same step key for both arms

    @eqx.filter_jit
    def step(state, k):
        return ppo.train_step(env, state, cfg, k, opt, stencil)

    state2, logs = step(state, sk)
    leaves = jax.tree_util.tree_leaves(
        eqx.filter((state2.actor, state2.critic, state2.dual), eqx.is_inexact_array))
    return leaves, {k: float(v) for k, v in logs.items()}


def _max_abs_leaf(leaves_a, leaves_b):
    m = 0.0
    for a, b in zip(leaves_a, leaves_b):
        d = np.abs(np.asarray(a, np.float64) - np.asarray(b, np.float64))
        m = max(m, float(d.max(initial=0.0)))
    return m


def _max_log(la, lb):
    shared = set(la) & set(lb)
    return max((abs(la[k] - lb[k]) for k in shared), default=0.0)


def _compare():
    """Judge remat exactness by CONTROLLING for backend non-determinism.

    GPU conv/reduction ops (cuDNN) are not bitwise-reproducible run-to-run, so even two
    IDENTICAL train_steps differ by a small floor. We measure that floor (OFF vs OFF) and
    the remat effect (OFF vs ON); remat is exact iff its effect is no larger than the
    non-determinism floor. On CPU both are ~0 (deterministic)."""
    cfg_off, *_ = train_ctde._parse_args(_ARGV)
    cfg_on = dataclasses.replace(cfg_off, loss=dataclasses.replace(cfg_off.loss, remat=True))
    assert cfg_off.loss.remat is False and cfg_on.loss.remat is True

    off1_leaves, off1_logs = _one_step_params(cfg_off)
    off2_leaves, off2_logs = _one_step_params(cfg_off)     # non-determinism control
    on_leaves, on_logs = _one_step_params(cfg_on)
    assert len(off1_leaves) == len(on_leaves), "param tree shape changed under remat"

    floor = _max_abs_leaf(off1_leaves, off2_leaves)         # OFF vs OFF  (backend noise)
    effect = _max_abs_leaf(off1_leaves, on_leaves)          # OFF vs ON   (remat)
    floor_log = _max_log(off1_logs, off2_logs)
    effect_log = _max_log(off1_logs, on_logs)
    return floor, effect, floor_log, effect_log


def test_remat_parity():
    floor, effect, floor_log, effect_log = _compare()
    tol = max(1e-6, 3.0 * floor)                            # exact ⟺ within backend noise
    assert effect <= tol, (f"remat param Δ {effect:.2e} exceeds non-determinism floor "
                           f"{floor:.2e} (tol {tol:.2e}) — remat NOT exact")
    tol_log = max(1e-6, 3.0 * floor_log)
    assert effect_log <= tol_log, (f"remat forward Δ {effect_log:.2e} exceeds floor "
                                   f"{floor_log:.2e} — forward not identical")


if __name__ == "__main__":
    floor, effect, floor_log, effect_log = _compare()
    print(f"param Δ:  OFF-vs-OFF (noise floor) = {floor:.3e}   OFF-vs-ON (remat) = {effect:.3e}")
    print(f"log   Δ:  OFF-vs-OFF (noise floor) = {floor_log:.3e}   OFF-vs-ON (remat) = {effect_log:.3e}")
    ok = (effect <= max(1e-6, 3.0 * floor)) and (effect_log <= max(1e-6, 3.0 * floor_log))
    print("PARITY:", "PASS ✅  (remat effect within backend non-determinism → exact)" if ok
          else "FAIL ❌  (remat effect EXCEEDS the noise floor — a real change)")
    raise SystemExit(0 if ok else 1)
