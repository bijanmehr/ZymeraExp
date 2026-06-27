"""Checkpointing: deployable-model round-trip + crash-resume of training."""
import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from fiedler import checkpoint as ckpt
from fiedler import train_eqx as T
from fiedler.models_eqx import ConfigurableGCRN


def _model(seed=0):
    return ConfigurableGCRN(in_size=6, hidden=16, n_rounds=1, op="mean", content="value",
                            heads=2, key=jax.random.PRNGKey(seed), dropedge=0.0, comm_r=5.0)


def _tiny_data(S=8, H=3, N=4, seed=0):
    k = jax.random.PRNGKey(seed)
    k1, k2, k3 = jax.random.split(k, 3)
    return {
        "X_node": jax.random.normal(k1, (S, H, N, 6)),
        "X_adj": jnp.ones((S, H, N, N)),                 # fully connected windows
        "X_pos": jax.random.normal(k2, (S, H, N, 2)),
        "y": jnp.abs(jax.random.normal(k3, (S,))) + 0.5,  # positive lambda2 targets
        "node_mask": jnp.ones((S, N), bool),
    }


def _arrays(m):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(eqx.filter(m, eqx.is_array))]


def test_save_load_model_roundtrip(tmp_path):
    m = _model(seed=0)
    p = str(tmp_path / "m.eqx")
    ckpt.save_model(p, m, meta={"op": "mean", "hidden": 16})
    assert ckpt.exists(p) and (tmp_path / "m.eqx.meta.json").exists()

    loaded = ckpt.load_model(p, _model(seed=999))        # template has *different* init
    for a, b in zip(_arrays(m), _arrays(loaded)):
        assert np.allclose(a, b)                          # loaded == saved, not the template


def test_training_resumes_from_checkpoint(tmp_path, monkeypatch):
    data = _tiny_data()
    p = str(tmp_path / "state.eqx")
    m = _model(seed=1)

    # Simulate a crash: make the validation metric raise on its 2nd call, AFTER a checkpoint
    # has been written (eval_every == ckpt_every == 1, so step 0 checkpoints then step 1 dies).
    real = T._val_median_rel_err_connected
    calls = {"n": 0}

    def flaky(*a, **k):
        calls["n"] += 1
        if calls["n"] >= 2:
            raise RuntimeError("simulated crash")
        return real(*a, **k)

    monkeypatch.setattr(T, "_val_median_rel_err_connected", flaky)
    with pytest.raises(RuntimeError):
        T.train_configurable(m, data, steps=20, eval_every=1, patience=99, batch=4,
                             ckpt_path=p, ckpt_every=1)
    assert ckpt.exists(p)                                 # crash left a resumable checkpoint

    monkeypatch.undo()                                    # "restart" with a healthy metric
    _, info = T.train_configurable(m, data, steps=20, eval_every=1, patience=99, batch=4,
                                   ckpt_path=p, ckpt_every=1)
    assert info["steps_run"] == 20                        # resumed and ran through to the end
    assert not ckpt.exists(p)                             # cleaned up on clean completion


def test_no_ckpt_path_is_noop(tmp_path):
    # training without a ckpt_path must behave exactly as before (no files, normal return)
    _, info = T.train_configurable(_model(), _tiny_data(), steps=6, eval_every=2, patience=99,
                                   batch=4)
    assert info["steps_run"] == 6
    assert not list(tmp_path.iterdir())
