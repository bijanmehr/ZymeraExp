"""Watchable launcher for the Fiedler agent-identity sweep.

Runs the 6 identity configs -- op in {mean, attention} x id_mode in {none, random, index},
content="value" -- on the hard-connectivity-guardrail data, printing per-config progress so
it can be followed live in tmux. Streams one JSON line per config to results/sweep_ids.jsonl
as each finishes.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u run_ids.py
    # follow live:  tmux attach -t zymera   (or)   tail -f results/sweep_ids.jsonl
"""
import json
import time

from fidler import sweep
import sweep_ids as si

OVERRIDES = {"steps": 8000, "n_steps": 150, "n_episodes": 5}
OUT = "results/sweep_ids.jsonl"


def main():
    configs = []
    for c in si.CONFIGS:
        cc = dict(c)
        cc.update(OVERRIDES)
        configs.append(cc)

    open(OUT, "w").close()
    print("=== Fiedler agent-identity sweep ===", flush=True)
    print(f"    {len(configs)} configs on guardrail data | train_N={configs[0]['train_N']} "
          f"| steps={OVERRIDES['steps']} | 5-fold@20 + extrapolate {configs[0]['extrap_N']}", flush=True)

    for k, c in enumerate(configs):
        t0 = time.time()
        print(f"\n[{k + 1}/{len(configs)}] op={c['op']}  id_mode={c['id_mode']}  ({c['name']})  "
              f"-- datagen + train + 5-fold + extrap ...", flush=True)
        r = sweep.run_config(c, base_seed=0)
        el = time.time() - t0
        with open(OUT, "a") as f:
            f.write(json.dumps(r, default=str) + "\n")
        metrics = {k2: (round(v2, 4) if isinstance(v2, float) else v2)
                   for k2, v2 in r.items() if k2 not in c and k2 != "grid_for_n"}
        status = "ERROR" if "error" in r else "OK"
        print(f"    -> [{status}]  {el:.0f}s   {metrics}", flush=True)

    print("\n=== ALL DONE ===", flush=True)


if __name__ == "__main__":
    main()
