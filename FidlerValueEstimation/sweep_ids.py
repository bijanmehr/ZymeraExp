"""The agent-identity sweep — does a way to self-distinguish help lambda2 estimation?

The connectivity-aware estimators are permutation-equivariant: agents in symmetric
positions are indistinguishable. This sweep tests three identity regimes against that
symmetry, on the SAME hard-connectivity-guardrail base as the message sweep
(`sweep_messages.BASE`), at content="value":

    id_mode:
      * "none"   : no identity   (the permutation-equivariant baseline; in_size = 6)
      * "random" : a per-agent RANDOM tag, constant over the H window -- a symmetry-
                   breaking label that carries no privileged value, only contrast
                   between agents (in_size = 6 + id_dim)
      * "index"  : the agent's raw node-axis INDEX / N_max -- a fixed position label,
                   expected to HURT size-transfer (in_size = 7)

CONFIGS = op in {mean, attention} x id_mode in {none, random, index}  (6 configs).
Each trains a ConfigurableGCRN on the multi-N guardrail pool and is scored for overall /
connected accuracy, the connected-flag head, 5-fold CV at N=20, and zero-shot
extrapolation to N in {24,30}.

Run (zymera venv):
    zymera_lab/.venv/bin/python -c "import sweep_ids; sweep_ids.main()"
    # quick smoke of the first 2 configs:
    zymera_lab/.venv/bin/python -c "import sweep_ids; sweep_ids.main(subset=2)"

Results are appended one JSON line per config to `results/sweep_ids.jsonl` as they finish.
"""
import itertools

from fidler import sweep
from sweep_messages import BASE

# ----------------------------------------------------------------------------------------
# the agent-identity space (op x id_mode), at content="value", id_dim=4
# ----------------------------------------------------------------------------------------
OPS = ["mean", "attention"]
ID_MODES = ["none", "random", "index"]
ID_DIM = 4


def _cfg(op, id_mode):
    c = dict(BASE)                       # same guardrail base as the message sweep
    c.update(op=op, content="value", id_mode=id_mode, id_dim=ID_DIM)
    c["name"] = f"{op}-value-id_{id_mode}"
    return c


def _build_configs():
    return [_cfg(op, id_mode) for op, id_mode in itertools.product(OPS, ID_MODES)]


CONFIGS = _build_configs()


def main(out="results/sweep_ids.jsonl", subset=None, base_seed=0, overrides=None):
    """Run the agent-identity sweep (or the first `subset` configs) -> appends to `out`.

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
    print(f"sweep_ids: {len(CONFIGS)} configs "
          f"({len(OPS)} ops x {len(ID_MODES)} id_modes, content=value)")
    main()
