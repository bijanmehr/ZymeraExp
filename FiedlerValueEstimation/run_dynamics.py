"""Dynamic-features experiment launcher: {max, multihead} x {dynamics off, on}, content=value.

Sharded + resumable (writes results/dyn_<shard>.jsonl). Does adding the temporal-trend features
(Delta-degree, neighbor approach-rate, own speed) move the ~0.66 ceiling?

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u run_dynamics.py 0 2     # worker 0 of 2
"""
import glob
import json
import os
import sys
import time

from fiedler import sweep
import sweep_dynamics as sd


def _key(c):
    return (c.get("op"), c.get("content", "value"), c.get("id_mode", "none"),
            c.get("margin_mode", "off"), c.get("signal_mode", "off"),
            c.get("dynamics_mode", "off"), int(c.get("hidden", 128)), int(c.get("n_rounds", 2)))


def _done_keys():
    done = set()
    for f in glob.glob("results/dyn_*.jsonl"):
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
    out = f"results/dyn_{shard}.jsonl"
    os.makedirs("results", exist_ok=True)
    done = _done_keys()
    mine = [c for i, c in enumerate(sd.CONFIGS) if i % nshards == shard]
    print(f"=== dynamics: shard {shard}/{nshards} -> {len(mine)} of {len(sd.CONFIGS)} configs "
          f"({len(done)} done globally) ===", flush=True)
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
        print(f"[{c['name']}] -> [{'ERROR' if 'error' in r else 'OK'}] {time.time() - t0:.0f}s  {m}",
              flush=True)
    print(f"=== shard {shard} done ===", flush=True)


if __name__ == "__main__":
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nshards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(shard, nshards)
