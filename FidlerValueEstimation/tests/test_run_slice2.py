import json
from run_slice2 import run_slice2


def test_run_slice2_writes_accuracy_json(tmp_path):
    out = tmp_path / "acc2.json"
    res = run_slice2(n_agents=8, H_list=(1, 3), n_episodes=2, n_steps=8, steps=50,
                     out_path=str(out))
    assert out.exists()
    saved = json.loads(out.read_text())
    assert set(saved.keys()) == {"gru", "gcrn", "power_iteration"}
    for k in ("gru", "gcrn", "power_iteration"):
        assert len(saved[k]) == 2
        assert all(0.0 <= a <= 1.0 for a in saved[k])
    # function returns the same dict it wrote
    assert set(res.keys()) == {"gru", "gcrn", "power_iteration"}


def test_run_slice2_writes_record_when_requested(tmp_path):
    out = tmp_path / "acc2.json"
    rec = tmp_path / "rec2.json"
    run_slice2(n_agents=8, H_list=(1,), n_episodes=2, n_steps=8, steps=30,
               out_path=str(out), record_path=str(rec))
    assert rec.exists()
    record = json.loads(rec.read_text())
    for key in ("experiment", "timestamp", "purpose", "config", "results"):
        assert key in record
    assert set(record["results"]) >= {"gru", "gcrn", "power_iteration"}
