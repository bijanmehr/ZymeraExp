"""Tests for the sweep runner (fidler.sweep)."""
import json

from fidler import sweep

METRIC_KEYS = {
    "accuracy", "connected_accuracy", "connected_flag_accuracy",
    "cv20_mean", "cv20_std", "extrap",
}


def _tiny_cfg(op="mean", content="value"):
    return {
        "op": op, "content": content, "n_rounds": 1, "hidden": 16, "H": 2,
        "train_N": [4, 8], "eval_N": [8],
        "steps": 30, "agree_w": 0.0, "dropedge": 0.0, "weight_decay": 1e-4,
        # keep heavy paths tiny so the unit test is fast but still exercised
        "cv_N": 8, "cv_folds": 2, "extrap_N": [6],
        "n_episodes": 2, "n_steps": 8, "grid": 12, "comm_r": 5,
    }


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
