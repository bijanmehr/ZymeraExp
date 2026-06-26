"""Tests for the top-level message sweep config + entry point."""
import sweep_messages

REQUIRED_KEYS = {"op", "content", "n_rounds", "hidden", "H", "train_N", "steps"}
OPS = {"mean", "gcn", "max", "sum", "attention", "multihead_attention", "gated", "laplacian"}
CONTENTS = {"value", "learned", "geom"}


def test_configs_nonempty_and_well_formed():
    cfgs = sweep_messages.CONFIGS
    assert isinstance(cfgs, list)
    assert len(cfgs) > 0
    for c in cfgs:
        assert isinstance(c, dict)
        assert REQUIRED_KEYS <= set(c.keys())
        assert c["op"] in OPS
        assert c["content"] in CONTENTS
        assert isinstance(c["train_N"], list) and len(c["train_N"]) >= 1
        assert c["n_rounds"] >= 1
        assert c["hidden"] >= 1


def test_full_op_x_content_grid_present():
    """Every (op, content) combination appears at least once in the base grid."""
    pairs = {(c["op"], c["content"]) for c in sweep_messages.CONFIGS}
    for op in OPS:
        for content in CONTENTS:
            assert (op, content) in pairs, f"missing ({op},{content})"


def test_main_is_callable_with_subset():
    assert callable(sweep_messages.main)
