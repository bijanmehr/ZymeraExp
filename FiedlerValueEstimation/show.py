"""Pretty-print a sweep results jsonl: per config, op/id/margin/signal + the headline metrics.

    python show.py results/sweep_ops_value.jsonl
"""
import json
import sys

KEYS = ("accuracy", "connected_accuracy", "cv20_mean", "cv20_std", "extrap")


def main(path):
    for line in open(path):
        r = json.loads(line)
        c = r.get("config", r)                       # run_config nests the echoed cfg under "config"
        tag = (f"op={c.get('op')} id={c.get('id_mode', '-')} "
               f"mar={c.get('margin_mode', '-')} sig={c.get('signal_mode', '-')}")
        if "error" in r:
            print(f"{tag:52s} ERROR")
            continue
        m = {k: (round(v, 4) if isinstance(v, float) else v) for k in KEYS
             if (v := r.get(k)) is not None}
        print(f"{tag:52s} {m}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sweep_ops_value.jsonl")
