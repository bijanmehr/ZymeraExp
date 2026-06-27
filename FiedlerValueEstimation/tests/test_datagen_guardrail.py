"""Tests for the hard-connectivity-guardrail dataset generators (fiedler.datagen)."""
import math
import numpy as np

from fiedler.config import DataCfg
from fiedler import datagen


def test_guardrail_dataset_schema_and_shapes():
    cfg = DataCfg(n_agents=6, grid=14, comm_r=5, n_episodes=2, n_steps=10, seed=0)
    ds = datagen.generate_dataset_guardrail(cfg)
    T = cfg.n_steps + 1
    # same schema as generate_dataset, plus connected_frac
    assert ds["features"].shape == (cfg.n_episodes, T, 6, 6)
    assert ds["adjacency"].shape == (cfg.n_episodes, T, 6, 6)
    assert ds["lambda2"].shape == (cfg.n_episodes, T)
    assert ds["positions"].shape == (cfg.n_episodes, T, 6, 2)
    assert ds["features"].dtype == np.float32
    assert ds["adjacency"].dtype == bool
    assert ds["lambda2"].dtype == np.float32
    assert int(ds["n_agents"]) == 6
    assert int(ds["comm_r"]) == 5
    assert "connected_frac" in ds


def test_guardrail_keeps_connected_real_episode():
    """The guardrail should keep essentially every (episode,step) connected."""
    cfg = DataCfg(n_agents=8, grid=16, comm_r=5, n_episodes=2, n_steps=40, seed=1)
    ds = datagen.generate_dataset_guardrail(cfg)
    assert float(ds["connected_frac"]) >= 0.98
    # lambda2 > 0 on essentially all steps (allow a tiny slack for the reset frame).
    frac_pos = float(np.mean(ds["lambda2"] > 1e-3))
    assert frac_pos >= 0.98
    assert np.all(ds["lambda2"] >= 0.0)


def test_guardrail_connected_frac_matches_lambda2():
    """connected_frac is exactly the fraction of (E,T+1) steps with lambda2 > 1e-3."""
    cfg = DataCfg(n_agents=5, grid=14, comm_r=5, n_episodes=2, n_steps=12, seed=2)
    ds = datagen.generate_dataset_guardrail(cfg)
    expected = float(np.mean(ds["lambda2"] > 1e-3))
    # connected_frac is stored as float32, so compare with float32 tolerance.
    assert math.isclose(float(ds["connected_frac"]), expected, rel_tol=0, abs_tol=1e-6)


def test_guardrail_save_roundtrip(tmp_path):
    cfg = DataCfg(n_agents=4, grid=12, comm_r=5, n_episodes=1, n_steps=6, seed=3)
    ds = datagen.generate_dataset_guardrail(cfg)
    p = tmp_path / "g.npz"
    datagen.save_npz(str(p), ds)
    back = np.load(str(p))
    assert np.allclose(back["lambda2"], ds["lambda2"])
    assert np.allclose(back["connected_frac"], ds["connected_frac"])


def test_generate_multi_guardrail_one_dict_per_N():
    n_list = [4, 6, 8]
    grid_for_n = lambda n: max(8, round(math.sqrt(n / 0.04)))
    out = datagen.generate_multi_guardrail(
        n_list, grid_for_n, comm_r=5, n_episodes=1, n_steps=6, seed0=0
    )
    assert isinstance(out, list)
    assert len(out) == len(n_list)
    for n, ds in zip(n_list, out):
        assert int(ds["n_agents"]) == n
        assert ds["positions"].shape[2] == n
        assert "connected_frac" in ds
