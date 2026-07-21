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


def _compare():
    cfg_off, *_ = train_ctde._parse_args(_ARGV)
    cfg_on = dataclasses.replace(cfg_off, loss=dataclasses.replace(cfg_off.loss, remat=True))
    assert cfg_off.loss.remat is False and cfg_on.loss.remat is True

    off_leaves, off_logs = _one_step_params(cfg_off)
    on_leaves, on_logs = _one_step_params(cfg_on)

    assert len(off_leaves) == len(on_leaves), "param tree shape changed under remat"
    max_abs = 0.0
    max_rel = 0.0
    for a, b in zip(off_leaves, on_leaves):
        a = np.asarray(a, np.float64)
        b = np.asarray(b, np.float64)
        d = np.abs(a - b)
        max_abs = max(max_abs, float(d.max(initial=0.0)))
        denom = np.abs(a) + 1e-8
        max_rel = max(max_rel, float((d / denom).max(initial=0.0)))
    # max abs difference over every shared scalar log (forward values must be identical).
    shared = set(off_logs) & set(on_logs)
    loss_diff = max((abs(off_logs[k] - on_logs[k]) for k in shared), default=0.0)
    return max_abs, max_rel, loss_diff, off_logs, on_logs


def test_remat_parity():
    max_abs, max_rel, loss_diff, *_ = _compare()
    # gradient checkpointing is exact; allow only tiny fp-reassociation slack.
    assert max_abs < 1e-5, f"post-step param max abs diff {max_abs:.2e} too large (remat not exact)"
    assert max_rel < 1e-4, f"post-step param max rel diff {max_rel:.2e} too large (remat not exact)"
    assert loss_diff < 1e-5, f"max log diff {loss_diff:.2e} (forward should be identical)"


if __name__ == "__main__":
    max_abs, max_rel, loss_diff, off_logs, on_logs = _compare()
    print(f"post-step param max |Δ|   = {max_abs:.3e}   (want < 1e-5)")
    print(f"post-step param max relΔ  = {max_rel:.3e}   (want < 1e-4)")
    print(f"max |Δ| over shared scalar logs = {loss_diff:.3e}   (forward parity)")
    ok = (max_abs < 1e-5) and (max_rel < 1e-4) and (loss_diff < 1e-5)
    print("PARITY:", "PASS ✅  (remat is exact — identical gradients)" if ok
          else "FAIL ❌  (remat changed the update — NOT safe)")
    raise SystemExit(0 if ok else 1)
