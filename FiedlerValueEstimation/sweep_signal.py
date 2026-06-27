"""The signal-strength sweep — does a continuous per-edge link quality help, on its own
and TOGETHER with the agent-ID feature?

A binary adjacency throws away *how strong* each comm link is: a real radio link does not
flip from perfect to gone at exactly comm_r, it falls off smoothly with distance. The
signal-strength overlay (`signal_mode="on"`) models that path-loss as a continuous weight
`s = exp(-3*(dist/comm_r)^2)` (~1 close, ~0.05 at comm range) and uses it three ways:
(a) it WEIGHTS the message aggregation -- the soft-weighted adjacency that aligns with the
soft Laplacian governing lambda2; (b) it rides in the messages (content forced to "margin",
the per-edge dist/comm_r); (c) it adds a per-agent mean-neighbor-link-quality node feature.

This sweep runs the combined experiment at op="max" (the current best aggregator) on the
SAME hard-connectivity-guardrail base as the message sweep (`sweep_messages.BASE`), as four
arms that cross the signal overlay with the agent-ID symmetry-breaking feature:

    arm        id_mode   signal_mode
    baseline   none      off          (the permutation-equivariant binary baseline)
    id         random    off          (ID feature only)
    signal     none      on           (signal overlay only)
    id+signal  random    on           (BOTH at the same time -- the "together" arm)

Each trains a ConfigurableGCRN on the multi-N guardrail pool and is scored for overall /
connected accuracy, the connected-flag head, 5-fold CV at N=20, and zero-shot
extrapolation to N in {24,30}.

Run (zymera venv):
    zymera_lab/.venv/bin/python -c "import sweep_signal; sweep_signal.main()"
    # quick smoke of the first 2 arms:
    zymera_lab/.venv/bin/python -c "import sweep_signal; sweep_signal.main(subset=2)"

Results are appended one JSON line per config to `results/sweep_signal.jsonl` as they finish.
"""
from fiedler import sweep
from sweep_messages import BASE

# ----------------------------------------------------------------------------------------
# the four arms at op="max": cross the signal overlay with the agent-ID feature.
# (name_suffix, id_mode, signal_mode)
# ----------------------------------------------------------------------------------------
OP = "max"
ID_DIM = 4
ARMS = [
    ("baseline", "none", "off"),     # permutation-equivariant binary baseline
    ("id", "random", "off"),         # agent-ID feature only
    ("signal", "none", "on"),        # signal-strength overlay only
    ("id+signal", "random", "on"),   # BOTH at the same time -- the "together" arm
]


def _cfg(suffix, id_mode, signal_mode):
    c = dict(BASE)                       # same guardrail base as the message sweep
    c.update(op=OP, content="value", id_mode=id_mode, id_dim=ID_DIM,
             signal_mode=signal_mode)
    c["name"] = f"{OP}-{suffix}"
    return c


def _build_configs():
    return [_cfg(suffix, id_mode, signal_mode)
            for suffix, id_mode, signal_mode in ARMS]


CONFIGS = _build_configs()


def main(out="results/sweep_signal.jsonl", subset=None, base_seed=0, overrides=None):
    """Run the signal-strength sweep (or the first `subset` configs) -> appends to `out`.

    `subset` may be an int (first-N configs) or a list of indices.
    `overrides` is an optional dict merged into every selected config — used to shrink the
    sweep for a quick smoke test (e.g. {"steps": 40, "hidden": 16, "train_N": [4, 8]}).
    Returns the results list.
    """
    configs = CONFIGS
    if subset is not None:
        if isinstance(subset, int):
            configs = CONFIGS[:subset]
        else:
            configs = [CONFIGS[i] for i in subset]
    if overrides:
        merged = []
        for c in configs:
            cc = dict(c)
            cc.update(overrides)
            merged.append(cc)
        configs = merged
    return sweep.run_sweep(configs, out, base_seed=base_seed)


if __name__ == "__main__":
    print(f"sweep_signal: {len(CONFIGS)} arms at op={OP} "
          f"(baseline / id / signal / id+signal)")
    main()
