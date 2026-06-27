"""The "go crazy" message-design sweep — Slice 2b entry point.

Enumerates the full message-design space (every aggregation `op` x message `content`) at a
fixed sensible base, plus a handful of net-size (`hidden`) and depth (`n_rounds`) variants on
the best-guess message families. Each config trains a `ConfigurableGCRN` on a multi-N pool
(N in {4,8,12,16,20}) and is scored for overall / connected accuracy, the connected-flag head,
5-fold CV at N=20, and zero-shot extrapolation to N in {24,30}.

Training data is the HARD-CONNECTIVITY-GUARDRAIL dispersion regime (`data="guardrail"`):
always-connected, dispersed rollouts on per-N grids (`grid_for_n`, ~fixed density) over long
(`n_steps=300`) episodes -- the realistic regime the deployed swarm actually runs in, replacing
the illegitimate random-policy data.

Run (zymera venv):
    zymera_lab/.venv/bin/python -c "import sweep_messages; sweep_messages.main()"
    # quick smoke of the first 2 configs:
    zymera_lab/.venv/bin/python -c "import sweep_messages; sweep_messages.main(subset=2)"

Results are appended one JSON line per config to `results/sweep_messages.jsonl` as they finish.
"""
import itertools
import math

from fiedler import sweep

# ----------------------------------------------------------------------------------------
# the message-design space
# ----------------------------------------------------------------------------------------
OPS = ["mean", "gcn", "max", "sum", "attention", "multihead_attention", "gated", "laplacian"]
CONTENTS = ["value", "learned", "geom"]

# Per-N grid that holds node DENSITY roughly fixed at ~0.04 agents/cell (clamped to a
# minimum side of 8): grid = round(sqrt(N / 0.04)). This is the realistic regime the
# guardrail-dispersed swarm runs in -- the same crowding at every agent-count.
def grid_for_n(n):
    return max(8, round(math.sqrt(n / 0.04)))


# fixed sensible base shared by the full op x content grid
BASE = dict(
    hidden=128,
    n_rounds=2,
    heads=4,
    H=5,
    train_N=[4, 8, 12, 16, 20],
    eval_N=[6, 10, 20],          # interpolation sizes + the size-of-interest
    cv_N=20,
    cv_folds=5,
    extrap_N=[24, 30],
    steps=12000,
    lr=3e-4,
    weight_decay=1e-4,
    agree_w=0.05,
    dropedge=0.1,
    batch=128,
    val_frac=0.2,
    patience=15,
    # datagen knobs -- HARD-CONNECTIVITY-GUARDRAIL dispersion data (always-connected,
    # realistic), per-N grids via grid_for_n, long rollouts for trajectory coverage.
    data="guardrail",
    grid_for_n=grid_for_n,
    comm_r=5,
    n_obstacles=0,
    spawn_radius=2,
    n_episodes=8,
    n_steps=300,
)


def _cfg(**over):
    c = dict(BASE)
    c.update(over)
    # a readable name for logging / dedup
    c["name"] = over.get("name") or (
        f"{c['op']}-{c['content']}-h{c['hidden']}-r{c['n_rounds']}")
    return c


def _build_configs():
    cfgs = []
    # 1) full op x content grid at the base
    for op, content in itertools.product(OPS, CONTENTS):
        cfgs.append(_cfg(op=op, content=content))

    # 2) net-size + depth variants on the best-guess message families
    best_ops = ["mean", "attention"]
    best_contents = ["value", "learned"]
    seen = {(c["op"], c["content"], c["hidden"], c["n_rounds"]) for c in cfgs}
    for op, content in itertools.product(best_ops, best_contents):
        for n_rounds in (1, 2, 3):
            for hidden in (64, 128, 256):
                key = (op, content, hidden, n_rounds)
                if key in seen:
                    continue
                seen.add(key)
                cfgs.append(_cfg(op=op, content=content, n_rounds=n_rounds, hidden=hidden))
    return cfgs


CONFIGS = _build_configs()


def main(out="results/sweep_messages.jsonl", subset=None, base_seed=0, overrides=None):
    """Run the message sweep (or the first `subset` configs) -> appends to `out`.

    `subset` may be an int (first-N configs) or a list of indices.
    `overrides` is an optional dict merged into every selected config — used to shrink the
    sweep for a quick smoke test (e.g. {"steps": 40, "hidden": 16, "train_N": [4, 8], "H": 2}).
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
    print(f"sweep_messages: {len(CONFIGS)} configs "
          f"({len(OPS)} ops x {len(CONTENTS)} contents + size/depth variants)")
    main()
