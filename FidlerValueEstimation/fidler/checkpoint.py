"""Crash-safe checkpoints for the Fiedler trainer.

Complements the config-level resume in the `run_*.py` launchers (which skip whole finished
configs) with two finer layers:

  * **training-state checkpoints** (`save_state`/`load_state`): the full mid-training state --
    current model, optimizer state, best-so-far model, RNG key, and the (step, best_err,
    patience) scalars -- written atomically every `ckpt_every` steps. A training that dies
    partway through resumes from the last checkpoint instead of restarting; the file is
    removed on clean completion.
  * **deployable model snapshots** (`save_model`/`load_model`): the final trained model plus a
    small JSON of the hyperparameters needed to rebuild its skeleton, so the estimator can be
    reloaded later for deployment / fine-tuning without retraining.

`eqx.tree_serialise_leaves` writes only array leaves, so deserialisation needs a same-shaped
*template* (built from the same config) -- the callers supply it. Writes go to a `.tmp` file
then `os.replace(...)` (atomic on POSIX), so a crash mid-write never corrupts a good ckpt.
"""
import json
import os

import equinox as eqx


def _atomic_serialise(path, pytree):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    eqx.tree_serialise_leaves(tmp, pytree)
    os.replace(tmp, path)                         # atomic rename on POSIX


def exists(path):
    """True iff `path` is a non-empty string pointing at an existing file."""
    return bool(path) and os.path.exists(path)


def remove(path):
    """Delete a checkpoint (and any stale `.tmp`); silent if absent."""
    if not path:
        return
    for p in (path, f"{path}.tmp"):
        try:
            os.remove(p)
        except OSError:
            pass


def save_state(path, state):
    """Atomically write a training-state pytree (dict of arrays / eqx modules)."""
    _atomic_serialise(path, state)


def load_state(path, template):
    """Read a training-state pytree, using `template` for structure + static fields."""
    return eqx.tree_deserialise_leaves(path, template)


def save_model(path, model, meta=None):
    """Write a deployable model snapshot (+ optional hyperparameter meta as a JSON sidecar)."""
    _atomic_serialise(path, model)
    if meta is not None:
        with open(f"{path}.meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)


def load_model(path, template):
    """Reload a deployable model into `template` (a freshly-built, same-config model)."""
    return eqx.tree_deserialise_leaves(path, template)
