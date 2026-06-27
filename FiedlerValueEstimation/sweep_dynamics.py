"""Dynamic-features experiment: does the 'neighbors moving away / degree crashing' trend carry
lambda2 signal the static snapshot lacks?

{max, multihead_attention} x {dynamics off, on}, content=value -> 4 configs. The 'off' arms are
the value baselines (max-value 0.659/.663, multihead-value 0.659/.670) re-run with identical
seeds, so 'on' vs 'off' is a clean controlled comparison.
"""
import itertools

import sweep_messages as sm

OVERRIDES = {"steps": 8000, "n_steps": 150, "n_episodes": 5}
AGGREGATORS = ["max", "multihead_attention"]
DYN = ["off", "on"]


def build():
    cfgs = []
    for op, dyn in itertools.product(AGGREGATORS, DYN):
        c = dict(sm.BASE)
        c.update(OVERRIDES)
        c.update(op=op, content="value", dynamics_mode=dyn)
        c["name"] = f"{op}__dyn_{dyn}"
        cfgs.append(c)
    return cfgs


CONFIGS = build()
