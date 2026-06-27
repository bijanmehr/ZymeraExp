"""Tests for the sweep runner (fiedler.sweep)."""
import json

import pytest

from fiedler import sweep

METRIC_KEYS = {
    "accuracy", "connected_accuracy", "connected_flag_accuracy",
    "cv20_mean", "cv20_std", "extrap",
}


def _tiny_cfg(op="mean", content="value", **over):
    cfg = {
        "op": op, "content": content, "n_rounds": 1, "hidden": 16, "H": 2,
        "train_N": [4, 8], "eval_N": [8],
        "steps": 30, "agree_w": 0.0, "dropedge": 0.0, "weight_decay": 1e-4,
        # keep heavy paths tiny so the unit test is fast but still exercised
        "cv_N": 8, "cv_folds": 2, "extrap_N": [6],
        "n_episodes": 2, "n_steps": 8, "grid": 12, "comm_r": 5,
    }
    cfg.update(over)
    return cfg


def test_run_config_returns_metric_keys():
    cfg = _tiny_cfg()
    out = sweep.run_config(cfg, base_seed=0)
    assert METRIC_KEYS <= set(out.keys())
    assert "config" in out
    # accuracies in [0,1]
    for k in ("accuracy", "connected_accuracy", "connected_flag_accuracy"):
        assert 0.0 <= out[k] <= 1.0
    # extrap is a dict keyed by N (as str or int)
    assert isinstance(out["extrap"], dict)


def test_run_sweep_appends_jsonl_per_config(tmp_path):
    out_path = tmp_path / "sweep.jsonl"
    configs = [_tiny_cfg("mean", "value"), _tiny_cfg("gcn", "value")]
    results = sweep.run_sweep(configs, str(out_path), base_seed=0)

    assert out_path.exists()
    lines = out_path.read_text().strip().splitlines()
    assert len(lines) == 2
    for ln in lines:
        rec = json.loads(ln)
        assert METRIC_KEYS <= set(rec.keys())
        assert "config" in rec
    assert len(results) == 2


# --------------------------------------------------------------------------------------
# agent-identity knob
# --------------------------------------------------------------------------------------
def test_id_in_size_helper():
    """_build_model in_size: 6 (none) / 6+id_dim (random) / 7 (index)."""
    assert sweep._id_in_size(_tiny_cfg(id_mode="none")) == 6
    assert sweep._id_in_size(_tiny_cfg()) == 6                       # default is 'none'
    assert sweep._id_in_size(_tiny_cfg(id_mode="random", id_dim=4)) == 10
    assert sweep._id_in_size(_tiny_cfg(id_mode="random", id_dim=3)) == 9
    assert sweep._id_in_size(_tiny_cfg(id_mode="index", id_dim=4)) == 7


@pytest.mark.parametrize("id_mode", ["none", "random", "index"])
def test_run_config_with_each_id_mode(id_mode):
    """run_config returns a full metrics dict for every id_mode (tiny config)."""
    cfg = _tiny_cfg(id_mode=id_mode, id_dim=4)
    out = sweep.run_config(cfg, base_seed=0)
    assert METRIC_KEYS <= set(out.keys())
    for k in ("accuracy", "connected_accuracy", "connected_flag_accuracy"):
        assert 0.0 <= out[k] <= 1.0
    assert isinstance(out["extrap"], dict)


# --------------------------------------------------------------------------------------
# connectivity-margin knob
# --------------------------------------------------------------------------------------
def test_model_in_size_margin_off_default():
    """margin_mode defaults to 'off' -> in_size unchanged from the id-only width."""
    assert sweep._model_in_size(_tiny_cfg()) == 6
    assert sweep._model_in_size(_tiny_cfg(margin_mode="off")) == 6
    assert sweep._model_in_size(_tiny_cfg(id_mode="random", id_dim=4)) == 10


def test_model_in_size_margin_on_adds_one():
    """margin_mode='on' adds exactly one to the model in_size, on top of any id width."""
    assert sweep._model_in_size(_tiny_cfg(margin_mode="on")) == 7
    assert sweep._model_in_size(_tiny_cfg(margin_mode="on", id_mode="index")) == 8
    assert sweep._model_in_size(_tiny_cfg(margin_mode="on", id_mode="random", id_dim=4)) == 11


def test_margin_mode_on_forces_margin_content():
    """When margin_mode='on', the built model's message content is forced to 'margin'."""
    import jax
    m_off = sweep._build_model(_tiny_cfg(content="value", margin_mode="off"),
                               jax.random.PRNGKey(0))
    m_on = sweep._build_model(_tiny_cfg(content="value", margin_mode="on"),
                              jax.random.PRNGKey(0))
    assert m_off.mp.content == "value"
    assert m_on.mp.content == "margin"
    # and the encoder in_size reflects the +1 margin feature
    assert m_on.encoder.in_features == 7
    assert m_off.encoder.in_features == 6


def test_run_config_margin_on_vs_off_both_return_metrics():
    """run_config works with margin_mode on and off (tiny config); both full metrics."""
    out_off = sweep.run_config(_tiny_cfg(margin_mode="off"), base_seed=0)
    out_on = sweep.run_config(_tiny_cfg(margin_mode="on"), base_seed=0)
    for out in (out_off, out_on):
        assert METRIC_KEYS <= set(out.keys())
        for k in ("accuracy", "connected_accuracy", "connected_flag_accuracy"):
            assert 0.0 <= out[k] <= 1.0
        assert isinstance(out["extrap"], dict)


# --------------------------------------------------------------------------------------
# signal-strength knob
# --------------------------------------------------------------------------------------
def test_model_in_size_signal_off_default():
    """signal_mode defaults to 'off' -> in_size unchanged from the id/margin width."""
    assert sweep._model_in_size(_tiny_cfg()) == 6
    assert sweep._model_in_size(_tiny_cfg(signal_mode="off")) == 6


def test_model_in_size_signal_on_adds_one():
    """signal_mode='on' adds exactly one to the model in_size, composing with id & margin."""
    assert sweep._model_in_size(_tiny_cfg(signal_mode="on")) == 7
    # composes independently with id (+id_dim) and margin (+1)
    assert sweep._model_in_size(_tiny_cfg(signal_mode="on", id_mode="random", id_dim=4)) == 11
    assert sweep._model_in_size(_tiny_cfg(signal_mode="on", margin_mode="on")) == 8
    assert sweep._model_in_size(
        _tiny_cfg(signal_mode="on", margin_mode="on", id_mode="index")) == 9


def test_signal_mode_on_forces_margin_content():
    """signal_mode='on' carries the per-edge signal by forcing message content to 'margin'."""
    import jax
    m_off = sweep._build_model(_tiny_cfg(content="value", signal_mode="off"),
                               jax.random.PRNGKey(0))
    m_on = sweep._build_model(_tiny_cfg(content="value", signal_mode="on"),
                              jax.random.PRNGKey(0))
    assert m_off.mp.content == "value"
    assert m_on.mp.content == "margin"
    assert m_on.encoder.in_features == 7                  # +1 signal node feature
    assert m_off.encoder.in_features == 6


def test_signal_mode_does_not_override_explicit_margin_content():
    """When margin_mode already set the content (margin), signal_mode keeps it margin."""
    import jax
    m = sweep._build_model(_tiny_cfg(content="value", margin_mode="on", signal_mode="on"),
                           jax.random.PRNGKey(0))
    assert m.mp.content == "margin"
    # +1 (margin feature) +1 (signal feature) on top of the base 6
    assert m.encoder.in_features == 8


def test_augment_signal_on_makes_adj_float_and_adds_feature():
    """The augmentation pipeline under signal_mode='on' yields a FLOAT (soft) X_adj and +1 feat."""
    import numpy as np
    # a tiny padded batch via the normal pool builder, signal OFF first (bool adj baseline).
    cfg_off = _tiny_cfg(signal_mode="off")
    data_off = sweep._build_pool([4, 8], int(cfg_off["H"]), 1, cfg_off)
    assert np.asarray(data_off["X_adj"]).dtype == bool
    f_off = data_off["X_node"].shape[-1]

    cfg_on = _tiny_cfg(signal_mode="on")
    data_on = sweep._build_pool([4, 8], int(cfg_on["H"]), 1, cfg_on)
    adj_on = np.asarray(data_on["X_adj"])
    # X_adj is now float soft weights in [0,1], same nonzero pattern off-diagonal as the bool adj
    assert np.issubdtype(adj_on.dtype, np.floating)
    assert adj_on.min() >= 0.0 and adj_on.max() <= 1.0
    off = ~np.eye(adj_on.shape[-1], dtype=bool)
    np.testing.assert_array_equal(
        (adj_on[..., :, :] != 0) & off, np.asarray(data_off["X_adj"])[..., :, :].astype(bool) & off)
    # exactly one extra node feature was appended
    assert data_on["X_node"].shape[-1] == f_off + 1


def test_run_config_signal_on_vs_off_both_return_metrics():
    """run_config works with signal_mode on and off (tiny config); both full metrics."""
    out_off = sweep.run_config(_tiny_cfg(signal_mode="off"), base_seed=0)
    out_on = sweep.run_config(_tiny_cfg(signal_mode="on"), base_seed=0)
    for out in (out_off, out_on):
        assert METRIC_KEYS <= set(out.keys())
        for k in ("accuracy", "connected_accuracy", "connected_flag_accuracy"):
            assert 0.0 <= out[k] <= 1.0
        assert isinstance(out["extrap"], dict)


def test_run_config_id_and_signal_together():
    """The combined arm (id_mode=random + signal_mode=on) trains and returns full metrics."""
    out = sweep.run_config(
        _tiny_cfg(id_mode="random", id_dim=4, signal_mode="on"), base_seed=0)
    assert METRIC_KEYS <= set(out.keys())
    for k in ("accuracy", "connected_accuracy", "connected_flag_accuracy"):
        assert 0.0 <= out[k] <= 1.0
