"""Focused check: does the `learned` message content actually help?

`learned` content = each neighbor's embedding z_j is passed through a small MLP before
aggregation (vs `value` = raw z_j). It adds trainable capacity but NO new input information
(unlike geom/margin/signal, which add inter-agent distance), so the prior is that it's weak --
this run measures it instead of assuming. Was never actually run in the OFAT sweeps.

Runs ONE aggregator (by sweep_messages.CONFIGS index, argv[1]) at content='learned' and writes
results/sweep_learned_<op>.jsonl. The three ops that span our range -- mean (weak baseline),
max (reliable winner), multihead (accuracy winner) -- run as parallel tmux workers, each to its
OWN file so there is no concurrent-write contention. Directly comparable to the recorded
content=value numbers (same base_seed=0, same data pipeline):
    mean-value 0.580 | max-value 0.656 | multihead-value 0.659.

    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u run_learned.py 1     # mean-learned   (idx 1)
    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u run_learned.py 7     # max-learned    (idx 7)
    XLA_PYTHON_CLIENT_PREALLOCATE=false python -u run_learned.py 16    # multihead      (idx 16)
"""
import json
import os
import sys
import time

from fiedler import sweep
import sweep_messages as sm

OVERRIDES = {"steps": 8000, "n_steps": 150, "n_episodes": 5}


def main(idx):
    c = dict(sm.CONFIGS[idx])
    c.update(OVERRIDES)
    assert c["content"] == "learned", f"idx {idx} is {c['op']}-{c['content']}, expected *-learned"
    out = f"results/sweep_learned_{c['op']}.jsonl"
    if os.path.exists(out) and os.path.getsize(out) > 0:
        print(f"[learned] {c['name']} already in {out} -- skip", flush=True)
        return
    print(f"=== learned check: {c['op']} x content=learned -> {out} ===", flush=True)
    t0 = time.time()
    r = sweep.run_config(c, base_seed=0)
    with open(out, "a") as f:
        f.write(json.dumps(r, default=str) + "\n")
    m = {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}
    print(f"-> [{'ERROR' if 'error' in r else 'OK'}]  {time.time() - t0:.0f}s   {m}", flush=True)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
