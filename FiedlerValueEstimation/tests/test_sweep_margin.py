"""Tests for the top-level connectivity-margin sweep config + entry point."""
import sweep_margin

REQUIRED_KEYS = {"op", "content", "margin_mode", "n_rounds", "hidden", "H", "train_N", "steps"}
OPS = {"mean", "attention"}
# the three arms compared per op: value baseline / margin-on idea / geom full-geometry ref
ARMS = {("value", "off"), ("value", "on"), ("geom", "off")}


def test_configs_are_the_op_x_arm_grid():
    cfgs = sweep_margin.CONFIGS
    assert isinstance(cfgs, list)
    assert len(cfgs) == len(OPS) * len(ARMS) == 6
    triples = {(c["op"], c["content"], c["margin_mode"]) for c in cfgs}
    for op in OPS:
        for content, margin_mode in ARMS:
            assert (op, content, margin_mode) in triples, \
                f"missing ({op},{content},{margin_mode})"


def test_configs_well_formed():
    for c in sweep_margin.CONFIGS:
        assert isinstance(c, dict)
        assert REQUIRED_KEYS <= set(c.keys())
        assert c["op"] in OPS
        assert (c["content"], c["margin_mode"]) in ARMS
        assert c["margin_mode"] in {"on", "off"}
        assert isinstance(c["train_N"], list) and len(c["train_N"]) >= 1


def test_names_are_readable():
    """Each config carries a readable name like mean-margin_on / mean-value / mean-geom."""
    names = {c["name"] for c in sweep_margin.CONFIGS}
    expected = {
        "mean-value", "mean-margin_on", "mean-geom",
        "attention-value", "attention-margin_on", "attention-geom",
    }
    assert names == expected


def test_margin_on_arm_uses_margin_mode_on():
    """The 'idea' arm sets margin_mode='on' (content is irrelevant, forced to margin)."""
    on = [c for c in sweep_margin.CONFIGS if c["margin_mode"] == "on"]
    assert len(on) == 2                                  # one per op
    for c in on:
        assert c["name"].endswith("margin_on")


def test_baseline_and_geom_arms_have_margin_off():
    off = [c for c in sweep_margin.CONFIGS if c["margin_mode"] == "off"]
    assert len(off) == 4                                 # value + geom, per op
    contents = {c["content"] for c in off}
    assert contents == {"value", "geom"}


def test_shares_guardrail_base_with_message_sweep():
    """sweep_margin reuses sweep_messages.BASE (same guardrail regime, train_N, etc.)."""
    from sweep_messages import BASE
    c = next(c for c in sweep_margin.CONFIGS
             if c["op"] == "mean" and c["name"] == "mean-value")
    assert c["data"] == BASE["data"] == "guardrail"
    assert c["train_N"] == BASE["train_N"]
    assert c["H"] == BASE["H"]


def test_main_is_callable():
    assert callable(sweep_margin.main)
