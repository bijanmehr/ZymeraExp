"""Deployable model snapshots for the CTDE v0 agent (the warm-start / scale-ladder I/O).

A thin, self-contained mirror of the Fiedler experiment's ``fiedler.checkpoint``
deployable-model helpers, so the SAVE path (``train_ctde --ckpt``) and the LOAD
path (``train_ctde --init-from`` / :func:`ppo.init_state_from_checkpoint`) agree on
ONE on-disk format without depending on ``FiedlerValueEstimation`` being importable
on ``PYTHONPATH``.

``eqx.tree_serialise_leaves`` writes only the array leaves, so deserialisation needs
a same-shaped *template* (a freshly-built actor/critic from the CURRENT config) — the
callers supply it. Because the LPAC backbone + heads are scale-invariant (param shapes
depend on channels / width / depth / mp_rounds / K / n_roles, NOT grid size or agent
count), a snapshot saved at one scale rung deserialises into a template built for the
next rung (16²/4 -> 32²/10) unchanged. Writes go to a ``.tmp`` then ``os.replace`` so
a crash mid-write never corrupts a good checkpoint.

The on-disk format is byte-compatible with ``fiedler.checkpoint.save_model`` /
``load_model`` (a bare ``tree_serialise_leaves`` of the ``(actor, critic)`` tuple + a
``<path>.meta.json`` sidecar), so either module reads the other's files.
"""
from __future__ import annotations

import json
import os

import equinox as eqx


def _atomic_serialise(path: str, pytree) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    tmp = f"{path}.tmp"
    eqx.tree_serialise_leaves(tmp, pytree)
    os.replace(tmp, path)                          # atomic rename on POSIX


def exists(path: str) -> bool:
    """True iff ``path`` is a non-empty string pointing at an existing file."""
    return bool(path) and os.path.exists(path)


def save_model(path: str, model, meta=None) -> None:
    """Write a deployable ``(actor, critic)`` snapshot (+ optional hyperparameter
    meta as a ``<path>.meta.json`` sidecar)."""
    _atomic_serialise(path, model)
    if meta is not None:
        with open(f"{path}.meta.json", "w") as f:
            json.dump(meta, f, indent=2, default=str)


def load_model(path: str, template):
    """Reload a deployable ``(actor, critic)`` snapshot into ``template`` (a
    freshly-built, current-config skeleton). Raises ``FileNotFoundError`` with a
    scale-strategy-aware message if the path is missing."""
    if not exists(path):
        raise FileNotFoundError(
            f"--init-from checkpoint not found: {path!r}. Pass the model.eqx written "
            f"by a prior `train_ctde --ckpt --run-dir <dir>` run (the previous scale "
            f"rung).")
    return eqx.tree_deserialise_leaves(path, template)
