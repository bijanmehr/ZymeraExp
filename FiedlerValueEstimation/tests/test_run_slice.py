import json
from run_slice import run_slice


def test_run_slice_produces_accuracy_curve(tmp_path):
    out = tmp_path / "acc.json"
    res = run_slice(n_agents_list=(4, 8), H_list=(1, 3, 5), n_episodes=2, n_steps=8, out_path=str(out))
    assert out.exists()
    saved = json.loads(out.read_text())
    assert set(saved.keys()) == {"power_iteration", "degree_regression"}
    assert len(saved["power_iteration"]) == 3
    assert all(0.0 <= a <= 1.0 for a in saved["power_iteration"])
    assert res["power_iteration"][-1] >= res["power_iteration"][0]
