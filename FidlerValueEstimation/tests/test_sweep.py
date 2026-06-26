"""Tests for the sweep runner (fidler.sweep)."""
import json

import pytest

from fidler import sweep

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
