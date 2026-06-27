"""Combinations grid — do the OFAT winners *compound*?

Cartesian product over aggregator x distance-content x identity, all on guardrail data, to test
whether stacking the individually-good ingredients beats each alone (or whether they're redundant
and plateau at the ~0.66 ceiling).

    aggregator : max, multihead_attention             (the two co-best aggregators from OFAT)
    content    : value, learned, geom, margin, signal (margin/signal set their mode flags)
    identity   : none, index                          (random was a no-op; dropped)
    => 2 x 5 x 2 = 20 configs.

Sharded for parallel balthar workers: `run_grid.py <shard_i> <num_shards>` runs the configs where
(global_index %% num_shards == shard_i) and appends to its OWN results/grid_<shard_i>.jsonl (combine
the files for reporting). Resumable: a config already present in ANY grid_*.jsonl is skipped, so a
re-launch never redoes work. Per-config checkpointing + model snapshots come from run_config.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u run_grid.py 0 5     # worker 0 of 5
"""
import glob
import itertools
import json
import os
import sys
import time

from fiedler import sweep
import sweep_messages as sm

OVERRIDES = {"steps": 8000, "n_steps": 150, "n_episodes": 5}

AGGREGATORS = ["max", "multihead_attention"]
# content choice -> (content, margin_mode, signal_mode); margin/signal force the margin content
# inside _build_model, signal additionally soft-weights the adjacency + adds a signal node feature.
CONTENTS = {
    "value":   ("value",   "off", "off"),
    "learned": ("learned", "off", "off"),
    "geom":    ("geom",    "off", "off"),
    "margin":  ("value",   "on",  "off"),
    "signal":  ("value",   "off", "on"),
}
IDS = ["none", "index"]


def build_grid():
    cfgs = []
    for op, ckey, idm in itertools.product(AGGREGATORS, CONTENTS, IDS):
        content, marg, sig = CONTENTS[ckey]
        c = dict(sm.BASE)
        c.update(OVERRIDES)
        c.update(op=op, content=content, margin_mode=marg, signal_mode=sig, id_mode=idm)
        c["name"] = f"{op}__{ckey}__id_{idm}"
        cfgs.append(c)
    return cfgs


GRID = build_grid()


def _key(c):
    return (c.get("op"), c.get("content", "value"), c.get("id_mode", "none"),
            c.get("margin_mode", "off"), c.get("signal_mode", "off"),
            int(c.get("hidden", 128)), int(c.get("n_rounds", 2)))


def _done_keys():
    done = set()
    for f in glob.glob("results/grid_*.jsonl"):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_key(json.loads(line).get("config", {})))
            except Exception:
                pass
    return done


def main(shard, nshards):
    out = f"results/grid_{shard}.jsonl"
    os.makedirs("results", exist_ok=True)
    done = _done_keys()
    mine = [c for i, c in enumerate(GRID) if i % nshards == shard]
    print(f"=== combinations grid: shard {shard}/{nshards} -> {len(mine)} of {len(GRID)} configs "
          f"({len(done)} already done globally) ===", flush=True)
    for c in mine:
        if _key(c) in done:
            print(f"[{c['name']}] already done -- skip", flush=True)
            continue
        t0 = time.time()
        print(f"[{c['name']}] start  -- datagen + train + 5-fold + extrap ...", flush=True)
        r = sweep.run_config(c, base_seed=0)
        with open(out, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        m = {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
        status = "ERROR" if "error" in r else "OK"
        print(f"[{c['name']}] -> [{status}] {time.time() - t0:.0f}s  {m}", flush=True)
    print(f"=== shard {shard} done ===", flush=True)


if __name__ == "__main__":
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nshards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(shard, nshards)
