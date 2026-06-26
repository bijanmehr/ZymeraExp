"""Watchable launcher for the Fiedler connectivity-margin sweep.

Runs the 6 margin configs -- op in {mean, attention} x arm in {value, margin_on, geom}
-- on the hard-connectivity-guardrail data, printing per-config progress so it can be
followed live in tmux. Streams one JSON line per config to results/sweep_margin.jsonl
as each finishes.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u run_margin.py
    # follow live:  tmux attach -t zymera   (or)   tail -f results/sweep_margin.jsonl
"""
import json
import time

from fidler import sweep
import sweep_margin as smg

OVERRIDES = {"steps": 8000, "n_steps": 150, "n_episodes": 5}
OUT = "results/sweep_margin.jsonl"


def main():
    configs = []
    for c in smg.CONFIGS:
        cc = dict(c)
        cc.update(OVERRIDES)
        configs.append(cc)

    open(OUT, "w").close()
    print("=== Fiedler connectivity-margin sweep ===", flush=True)
    print(f"    {len(configs)} configs on guardrail data | train_N={configs[0]['train_N']} "
          f"| steps={OVERRIDES['steps']} | 5-fold@20 + extrapolate {configs[0]['extrap_N']}", flush=True)

    for k, c in enumerate(configs):
        t0 = time.time()
        print(f"\n[{k + 1}/{len(configs)}] op={c['op']}  mode={c['margin_mode']}  ({c['name']})  "
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
