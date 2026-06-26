import numpy as np
from fidler.config import DataCfg
from fidler import datagen


def test_generate_dataset_shapes_and_labels():
    cfg = DataCfg(n_agents=4, grid=12, comm_r=5, n_episodes=2, n_steps=8, seed=0)
    ds = datagen.generate_dataset(cfg)
    T = cfg.n_steps + 1
    assert ds["features"].shape == (cfg.n_episodes, T, 4, 6)
    assert ds["adjacency"].shape == (cfg.n_episodes, T, 4, 4)
    assert ds["lambda2"].shape == (cfg.n_episodes, T)
    assert np.all(ds["lambda2"] >= 0.0)
    assert ds["lambda2"].dtype == np.float32


def test_generate_dataset_includes_positions():
    cfg = DataCfg(n_agents=4, grid=12, comm_r=5, n_episodes=2, n_steps=8, seed=0)
    ds = datagen.generate_dataset(cfg)
    T = cfg.n_steps + 1
    assert "positions" in ds
    assert ds["positions"].shape == (cfg.n_episodes, T, 4, 2)
    assert ds["positions"].dtype == np.float32

def test_save_and_load_roundtrip(tmp_path):
    cfg = DataCfg(n_agents=4, grid=12, comm_r=5, n_episodes=1, n_steps=4, seed=1)
    ds = datagen.generate_dataset(cfg)
    p = tmp_path / "ds.npz"
    datagen.save_npz(str(p), ds)
    back = np.load(str(p))
    assert np.allclose(back["lambda2"], ds["lambda2"])
