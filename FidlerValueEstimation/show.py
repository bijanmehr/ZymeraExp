"""Pretty-print a sweep results jsonl: per config, op/content/id_mode + the metric keys.

    python show.py results/sweep_ops_value.jsonl
"""
import json
import sys

CFG = {"hidden", "n_rounds", "heads", "H", "train_N", "eval_N", "cv_N", "cv_folds", "extrap_N",
       "steps", "lr", "weight_decay", "agree_w", "dropedge", "batch", "val_frac", "patience",
       "data", "grid_for_n", "comm_r", "n_obstacles", "spawn_radius", "n_episodes", "n_steps",
       "name", "id_dim", "margin_mode"}


def main(path):
    for line in open(path):
        r = json.loads(line)
        tag = f"op={r.get('op')} id={r.get('id_mode', '-')} mar={r.get('margin_mode', '-')}"
        metrics = {k: (round(v, 4) if isinstance(v, float) else v)
                   for k, v in r.items()
                   if k not in CFG and k not in ("op", "content", "id_mode", "margin_mode")}
        print(f"{tag:44s} {metrics}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/sweep_ops_value.jsonl")
