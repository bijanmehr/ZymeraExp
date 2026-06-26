# zymera_experiments

Experiments live HERE, outside the `zymera_lab/` library. Each file imports
`zymera` (the simulator + `nets` + `train`) and writes its own learning stack —
wire blocks into a policy, pick/compose a trainer, define the mission, run.

**Dependency rule (one-way):** `zymera_experiments → zymera`. Never edit the
library to make an experiment work; if a block or trainer proves out across >1
experiment, graduate it into `zymera/nets.py` / `zymera/train.py` with a test.

Run an experiment against the lab's venv, e.g.:

    ../zymera_lab/.venv/bin/python 00_random_rollout.py
