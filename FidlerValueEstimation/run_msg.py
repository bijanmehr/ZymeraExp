"""Watchable launcher for the Fiedler message-aggregation sweep (first batch).

Runs the 8 aggregation ops at content="value" on the hard-connectivity-guardrail data,
printing per-config progress so it can be followed live in tmux. Streams one JSON line
per config to results/sweep_ops_value.jsonl as each finishes.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u run_msg.py
    # follow live:  tmux attach -t zymera   (or)   tail -f results/sweep_ops_value.log
"""
import json
import time

from fidler import sweep
import sweep_messages as sm

IDXS = [0, 3, 6, 9, 12, 15, 18, 21]          # the 8 aggregation ops, all at content="value"
OVERRIDES = {"steps": 8000, "n_steps": 150, "n_episodes": 5}
OUT = "results/sweep_ops_value.jsonl"


def main():
    configs = []
    for i in IDXS:
        c = dict(sm.CONFIGS[i])
        c.update(OVERRIDES)
        configs.append(c)

    open(OUT, "w").close()
    print("=== Fiedler message-aggregation sweep ===", flush=True)
    print(f"    {len(configs)} ops on guardrail data | train_N={configs[0]['train_N']} "
          f"| steps={OVERRIDES['steps']} | 5-fold@20 + extrapolate {configs[0]['extrap_N']}", flush=True)

    for k, c in enumerate(configs):
        t0 = time.time()
        print(f"\n[{k + 1}/{len(configs)}] op={c['op']}  ({c['name']})  "
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
