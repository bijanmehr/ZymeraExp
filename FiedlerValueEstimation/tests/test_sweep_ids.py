"""Tests for the top-level agent-identity sweep config + entry point."""
import sweep_ids

REQUIRED_KEYS = {"op", "content", "id_mode", "n_rounds", "hidden", "H", "train_N", "steps"}
OPS = {"mean", "attention"}
ID_MODES = {"none", "random", "index"}


def test_configs_are_the_op_x_idmode_grid():
    cfgs = sweep_ids.CONFIGS
    assert isinstance(cfgs, list)
    assert len(cfgs) == len(OPS) * len(ID_MODES) == 6
    pairs = {(c["op"], c["id_mode"]) for c in cfgs}
    for op in OPS:
        for id_mode in ID_MODES:
            assert (op, id_mode) in pairs, f"missing ({op},{id_mode})"


def test_configs_well_formed():
    for c in sweep_ids.CONFIGS:
        assert isinstance(c, dict)
        assert REQUIRED_KEYS <= set(c.keys())
        assert c["op"] in OPS
        assert c["content"] == "value"
        assert c["id_mode"] in ID_MODES
        assert c["id_dim"] >= 1
        assert isinstance(c["train_N"], list) and len(c["train_N"]) >= 1


def test_shares_guardrail_base_with_message_sweep():
    """sweep_ids reuses sweep_messages.BASE (same guardrail regime, train_N, etc.)."""
    from sweep_messages import BASE
    c = next(c for c in sweep_ids.CONFIGS if c["op"] == "mean" and c["id_mode"] == "none")
    assert c["data"] == BASE["data"] == "guardrail"
    assert c["train_N"] == BASE["train_N"]
    assert c["H"] == BASE["H"]


def test_main_is_callable():
    assert callable(sweep_ids.main)
