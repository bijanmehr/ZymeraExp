"""The connectivity-margin sweep — does an edge-fragility "about-to-disconnect" signal help?

A binary adjacency throws away *how close* each comm edge is to breaking: a boundary edge
at ~comm_r barely holds the graph together (it lowers lambda2), but reads as an identical
`1`. This sweep tests recovering that discarded soft-weighted-Laplacian signal -- the
distance-to-each-neighbor relative to comm range -- against two references, on the SAME
hard-connectivity-guardrail base as the message sweep (`sweep_messages.BASE`):

    per op in {mean, attention}, three arms:
      * value  (margin_mode=off)  -- the binary-adjacency BASELINE (content="value")
      * margin (margin_mode=on)   -- THE IDEA: per-edge dist/comm_r INSIDE the messages
                                     (content forced to "margin") AND the per-agent
                                     weakest-link fragility added as a NODE feature
      * geom   (margin_mode=off)  -- the full-geometry REFERENCE (content="geom":
                                     [z_j, dx, dy, dist]/comm_r per edge)

CONFIGS = op in {mean, attention} x arm in {value, margin_on, geom}  (6 configs).
Each trains a ConfigurableGCRN on the multi-N guardrail pool and is scored for overall /
connected accuracy, the connected-flag head, 5-fold CV at N=20, and zero-shot
extrapolation to N in {24,30}.

Run (zymera venv):
    zymera_lab/.venv/bin/python -c "import sweep_margin; sweep_margin.main()"
    # quick smoke of the first 2 configs:
    zymera_lab/.venv/bin/python -c "import sweep_margin; sweep_margin.main(subset=2)"

Results are appended one JSON line per config to `results/sweep_margin.jsonl` as they finish.
"""
import itertools

from fidler import sweep
from sweep_messages import BASE

# ----------------------------------------------------------------------------------------
# the connectivity-margin space (op x arm), on the shared guardrail base
# ----------------------------------------------------------------------------------------
OPS = ["mean", "attention"]
# (name_suffix, content, margin_mode): the three arms compared per op.
ARMS = [
    ("value", "value", "off"),       # baseline: binary adjacency, plain value messages
    ("margin_on", "value", "on"),    # the idea: margin messages + fragility node feature
    ("geom", "geom", "off"),         # full-geometry reference: [z_j, dx, dy, dist]/comm_r
]


def _cfg(op, suffix, content, margin_mode):
    c = dict(BASE)                       # same guardrail base as the message sweep
    c.update(op=op, content=content, margin_mode=margin_mode)
    c["name"] = f"{op}-{suffix}"
    return c


def _build_configs():
    return [_cfg(op, suffix, content, margin_mode)
            for op, (suffix, content, margin_mode) in itertools.product(OPS, ARMS)]


CONFIGS = _build_configs()


def main(out="results/sweep_margin.jsonl", subset=None, base_seed=0, overrides=None):
    """Run the connectivity-margin sweep (or the first `subset` configs) -> appends to `out`.

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
    print(f"sweep_margin: {len(CONFIGS)} configs "
          f"({len(OPS)} ops x {len(ARMS)} arms: value / margin_on / geom)")
    main()
