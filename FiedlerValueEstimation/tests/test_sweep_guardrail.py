"""Tests for the guardrail `data` knob in the sweep runner (fiedler.sweep)."""
import math

from fiedler import sweep

METRIC_KEYS = {
    "accuracy", "connected_accuracy", "connected_flag_accuracy",
    "cv20_mean", "cv20_std", "extrap",
}


def _tiny_guardrail_cfg(**over):
    cfg = {
        "op": "mean", "content": "value", "n_rounds": 1, "hidden": 16, "H": 2,
        "train_N": [4, 6], "eval_N": [6],
        "steps": 30, "agree_w": 0.0, "dropedge": 0.0, "weight_decay": 1e-4,
        "cv_N": 6, "cv_folds": 2, "extrap_N": [8],
        "n_episodes": 1, "n_steps": 8, "grid": 12, "comm_r": 5,
        "data": "guardrail",
    }
    cfg.update(over)
    return cfg


def test_run_config_guardrail_returns_metric_keys():
    out = sweep.run_config(_tiny_guardrail_cfg(), base_seed=0)
    assert METRIC_KEYS <= set(out.keys())
    assert out["config"]["data"] == "guardrail"
    for k in ("accuracy", "connected_accuracy", "connected_flag_accuracy"):
        assert 0.0 <= out[k] <= 1.0
    assert isinstance(out["extrap"], dict)


def test_run_config_default_is_guardrail():
    """Omitting `data` defaults to the guardrail generator (no 'random' key needed)."""
    cfg = _tiny_guardrail_cfg()
    cfg.pop("data")
    out = sweep.run_config(cfg, base_seed=1)
    assert METRIC_KEYS <= set(out.keys())


def test_run_config_random_path_still_works():
    """data='random' falls back to the original generate_dataset path."""
    out = sweep.run_config(_tiny_guardrail_cfg(data="random"), base_seed=2)
    assert METRIC_KEYS <= set(out.keys())


def test_grid_for_n_applied_per_N():
    """A grid_for_n callable overrides the static grid per agent-count.

    We can't see the datasets directly, but with guardrail data + a grid_for_n that
    holds density fixed, run_config must complete and return metrics (the per-N grids
    are large enough to fit the agents). This exercises the per-N grid path.
    """
    grid_for_n = lambda n: max(8, round(math.sqrt(n / 0.04)))
    cfg = _tiny_guardrail_cfg(train_N=[4, 8], eval_N=[8], extrap_N=[10], cv_N=8)
    cfg["grid_for_n"] = grid_for_n
    out = sweep.run_config(cfg, base_seed=3)
    assert METRIC_KEYS <= set(out.keys())
