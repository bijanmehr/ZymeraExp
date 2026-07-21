"""Compile-only GPU memory probe — report the XLA compiler's PLANNED peak memory for one
``train_step`` WITHOUT running/allocating it. Safe to run alongside a busy GPU (compilation
grabs almost nothing). Use it to compare --remat off vs on for the same config.

    PYTHONPATH=.:../../../FiedlerValueEstimation ~/ZymeraLab/.venv/bin/python -m ctde_v0.mem_probe \
        --grid 32 --n-agents 10 --comm-r 5 --rollouts 16 --horizon 100 \
        --explorer-tool frontier_attn --collision-mask on --mechanism lagrangian \
        --conn-signal global_lambda2 --constraint-threshold 0.5 --role-picker expl_relay \
        --w-coverage 3 --sense-walls --sense-free --boundary --cover-r 0 [--remat]

Prints temp_size (the peak scratch — this is what OOMs), plus argument/output sizes. Peak device
memory ≈ temp + argument + output; the number to watch across --remat off/on is temp_size.
"""
from __future__ import annotations

import sys

import equinox as eqx
import jax

from ctde_v0 import env_utils, ppo, train_ctde


def probe(argv):
    cfg, *_ = train_ctde._parse_args(argv)
    env = env_utils.build_env(cfg)
    key = jax.random.PRNGKey(0)
    opt = ppo.make_optimizer(cfg)
    stencil = ppo.make_stencil(cfg)
    state = ppo.init_state(env, cfg, key)
    sk = jax.random.split(key, 2)[1]

    # raw jax.jit (partition the state into arrays / static) so the compiled object exposes
    # .memory_analysis() — equinox's filter_jit wrapper does not forward it.
    params, static = eqx.partition(state, eqx.is_array)

    def f(p, k):
        st = eqx.combine(p, static)
        return ppo.train_step(env, st, cfg, k, opt, stencil)

    compiled = jax.jit(f).lower(params, sk).compile()   # COMPILE ONLY — no execution
    ma = compiled.memory_analysis()
    return cfg, ma


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    remat = "--remat" in argv
    cfg, ma = probe(argv)
    mb = lambda b: (b or 0) / 1e6
    print(f"=== mem_probe  grid={cfg.world.grid}²/{cfg.world.n_agents}  "
          f"mechanism={cfg.mission_safety.mechanism}  remat={remat} ===")
    for a in ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes",
              "generated_code_size_in_bytes", "host_temp_size_in_bytes"):
        if hasattr(ma, a):
            print(f"  {a:32s} {mb(getattr(ma, a)):10.1f} MB")
    peak = mb(getattr(ma, "temp_size_in_bytes", 0)) + mb(getattr(ma, "argument_size_in_bytes", 0)) \
        + mb(getattr(ma, "output_size_in_bytes", 0))
    print(f"  {'≈ device peak (temp+arg+out)':32s} {peak:10.1f} MB")


if __name__ == "__main__":
    main()
