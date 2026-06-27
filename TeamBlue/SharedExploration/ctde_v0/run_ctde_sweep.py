"""Increment-1 permutation sweep — does labor-division (roles) + anti-overlap
break the v0 *huddle* (coverage collapses to ~7% while agents clump for trivial
100% connectivity)?

The I1 permutation (8 configs), with index / edge-content / compass left at their
v0 defaults for now:

    role_picker  {off, expl_relay}      # homogeneous goal head vs a learned role head
  x mechanism    {action_mask, soft_lambda}  # the mission-safety axis
  x anti_overlap {off, on}              # subtract same_step_overlap from the reward
  = 8 configs.

Sharded for parallel CPU workers (this experiment is CPU-only — do NOT run on the
GPU server): ``run_ctde_sweep.py <shard> <nshards>`` runs the configs where
``global_index % nshards == shard`` and appends each result to its OWN
``results/ctde_<name>.jsonl`` (combine the files for reporting). Resumable: a
config already present in ANY ``results/ctde_*.jsonl`` is skipped, so a re-launch
never redoes work. Every run saves the full §5 config (``config.py``) alongside
its metrics, so any result is reproducible.

Per config we log the FINAL-iteration: coverage%, connectivity%, aux-λ₂ accuracy,
controller-valid%, episode reward, and the role split (explorer / relay fraction).

    JAX_PLATFORMS=cpu \
    /Users/bijanmehr/Project.Zymera/zymera_lab/.venv/bin/python -u \
        ctde_v0/run_ctde_sweep.py 0 1            # single worker, all 8 configs

Tunables via env vars (so the runner stays argv-compatible with run_grid.py):
    CTDE_ITERS (default 50) · CTDE_ROLLOUTS (8) · CTDE_GRID (16) · CTDE_NAGENTS (4)
    · CTDE_HORIZON (100) · CTDE_SEED (0) · CTDE_AO_WEIGHT (1.0)
"""
from __future__ import annotations

import dataclasses
import glob
import itertools
import json
import os
import sys
import time

# Allow `python ctde_v0/run_ctde_sweep.py` (script) and `-m ctde_v0.run_ctde_sweep`.
if __package__ in (None, ""):
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from ctde_v0 import env_utils, ppo  # type: ignore
    from ctde_v0.config import (  # type: ignore
        Backbone, CTDEConfig, Loss, MissionSafety, Regularization, Reward,
        Trainer, World,
    )
else:
    from . import env_utils, ppo
    from .config import (
        Backbone, CTDEConfig, Loss, MissionSafety, Regularization, Reward,
        Trainer, World,
    )

# results live next to this file so shards from any cwd agree on one directory.
_RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

# ---- the I1 permutation axes ------------------------------------------------
ROLE_PICKERS = ["off", "expl_relay"]
MECHANISMS = ["action_mask", "soft_lambda"]
ANTI_OVERLAP = ["off", "on"]


def _env_int(name: str, default: int) -> int:
    return int(os.environ.get(name, default))


def _env_float(name: str, default: float) -> float:
    return float(os.environ.get(name, default))


def build_config(role_picker: str, mechanism: str, anti_overlap: str) -> CTDEConfig:
    """One full §5 config for a permutation cell (defaults = the 16×16/4 v0 slice;
    index / edge-content / compass stay at v0 defaults for now)."""
    grid = _env_int("CTDE_GRID", 16)
    n_agents = _env_int("CTDE_NAGENTS", 4)
    horizon = _env_int("CTDE_HORIZON", 100)
    return CTDEConfig(
        world=World(grid=grid, n_agents=n_agents, comm_r=5, horizon=horizon),
        backbone=Backbone(),                       # lpac / max agg / mp_rounds 2
        action_head=CTDEConfig().action_head,      # goal_pointer K=9 stride 3
        mission_safety=MissionSafety(mechanism=mechanism),
        reward=Reward(),
        loss=Loss(),
        trainer=Trainer(),
        regularization=Regularization(),
        role_picker=role_picker,
        reward_anti_overlap=anti_overlap,
        anti_overlap_weight=_env_float("CTDE_AO_WEIGHT", 1.0),
        scale=f"{grid}x{grid}/{n_agents}",
        iters=_env_int("CTDE_ITERS", 50),
        rollouts_per_iter=_env_int("CTDE_ROLLOUTS", 8),
        seed=_env_int("CTDE_SEED", 0),
    )


def config_name(role_picker: str, mechanism: str, anti_overlap: str) -> str:
    return f"role_{role_picker}__{mechanism}__ao_{anti_overlap}"


def build_grid() -> list[dict]:
    """The 8 permutation cells, each a dict the runner can name + skip by key."""
    cells = []
    for rp, mech, ao in itertools.product(ROLE_PICKERS, MECHANISMS, ANTI_OVERLAP):
        cells.append({"role_picker": rp, "mechanism": mech, "anti_overlap": ao,
                      "name": config_name(rp, mech, ao)})
    return cells


GRID = build_grid()


def _key(cfg_dict: dict) -> tuple:
    """Resume key from a saved config dict: the three swept axes uniquely identify
    a permutation cell (the other knobs are fixed defaults for I1)."""
    ms = cfg_dict.get("mission_safety", {})
    return (cfg_dict.get("role_picker", "off"),
            ms.get("mechanism", "action_mask"),
            cfg_dict.get("reward_anti_overlap", "off"))


def _cell_key(cell: dict) -> tuple:
    return (cell["role_picker"], cell["mechanism"], cell["anti_overlap"])


def _done_keys() -> set:
    """Permutation cells already present in ANY results/ctde_*.jsonl (resume)."""
    done = set()
    for f in glob.glob(os.path.join(_RESULTS_DIR, "ctde_*.jsonl")):
        for line in open(f):
            line = line.strip()
            if not line:
                continue
            try:
                done.add(_key(json.loads(line).get("config", {})))
            except Exception:
                pass
    return done


def run_config(cell: dict) -> dict:
    """Train one permutation cell to ``iters`` and return its result record
    (final-iter metrics + the full saved config). Errors are captured, not raised,
    so one bad cell never kills the shard."""
    import jax  # local import: keep module import cheap / argv-parse fast.

    cfg = build_config(cell["role_picker"], cell["mechanism"], cell["anti_overlap"])
    rec: dict = {"name": cell["name"], "config": cfg.to_dict()}
    try:
        env = env_utils.build_env(cfg)
        _state, history = ppo.train(env, cfg, key=jax.random.PRNGKey(cfg.seed))
        last = history[-1]
        rec.update({
            "iters": len(history),
            "coverage_pct": last["coverage_pct"],
            "connectivity_pct": last["connectivity_pct"],
            "mean_lambda2": last["mean_lambda2"],
            "aux_acc": last["aux_acc"],
            "median_rel_l2": last["median_rel_l2"],
            "ctrl_valid_frac": last["ctrl_valid_frac"],
            "ep_reward": last["ep_reward"],
            "explorer_frac": last.get("explorer_frac", 1.0),
            "relay_frac": last.get("relay_frac", 0.0),
            "role_entropy": last.get("role_entropy", 0.0),
            # a couple of first->last deltas so the jsonl is self-describing.
            "coverage_pct_first": history[0]["coverage_pct"],
            "connectivity_pct_first": history[0]["connectivity_pct"],
        })
    except Exception as e:  # noqa: BLE001 — record-and-continue by design.
        import traceback
        rec["error"] = f"{type(e).__name__}: {e}"
        rec["traceback"] = traceback.format_exc()
    return rec


def main(shard: int, nshards: int) -> None:
    os.makedirs(_RESULTS_DIR, exist_ok=True)
    out = os.path.join(_RESULTS_DIR, f"ctde_shard{shard}.jsonl")
    done = _done_keys()
    mine = [c for i, c in enumerate(GRID) if i % nshards == shard]
    print(f"=== I1 ctde sweep: shard {shard}/{nshards} -> {len(mine)} of "
          f"{len(GRID)} configs ({len(done)} already done globally) ===", flush=True)
    for cell in mine:
        if _cell_key(cell) in done:
            print(f"[{cell['name']}] already done -- skip", flush=True)
            continue
        t0 = time.time()
        print(f"[{cell['name']}] start -- train {os.environ.get('CTDE_ITERS', 50)} "
              f"iters ...", flush=True)
        rec = run_config(cell)
        with open(out, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if "error" in rec:
            print(f"[{cell['name']}] -> [ERROR] {time.time()-t0:.0f}s  {rec['error']}",
                  flush=True)
        else:
            print(f"[{cell['name']}] -> [OK] {time.time()-t0:.0f}s  "
                  f"cov={rec['coverage_pct']*100:5.1f}% "
                  f"conn={rec['connectivity_pct']*100:5.1f}% "
                  f"aux={rec['aux_acc']*100:5.1f}% "
                  f"expl={rec['explorer_frac']*100:5.1f}% "
                  f"valid={rec['ctrl_valid_frac']*100:5.1f}% "
                  f"rew={rec['ep_reward']:.2f}", flush=True)
    print(f"=== shard {shard} done -> {out} ===", flush=True)


if __name__ == "__main__":
    shard = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    nshards = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    main(shard, nshards)
