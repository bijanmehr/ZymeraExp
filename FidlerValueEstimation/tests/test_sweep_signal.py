"""Tests for the top-level signal-strength combined sweep config + entry point.

Four arms at op="max" on the shared guardrail base, toggling the signal-strength overlay
and the agent-ID feature, including the combined "at the same time" arm:
    baseline  (id none,   signal off)
    id        (id random, signal off)
    signal    (id none,   signal on)
    id+signal (id random, signal on)
"""
import sweep_signal

REQUIRED_KEYS = {"op", "id_mode", "signal_mode", "n_rounds", "hidden", "H", "train_N", "steps"}
# (name_suffix, id_mode, signal_mode) for the four arms.
ARMS = {
    ("baseline", "none", "off"),
    ("id", "random", "off"),
    ("signal", "none", "on"),
    ("id+signal", "random", "on"),
}


def test_four_arms_all_at_op_max():
    cfgs = sweep_signal.CONFIGS
    assert isinstance(cfgs, list)
    assert len(cfgs) == 4
    assert all(c["op"] == "max" for c in cfgs)
    triples = {(c["id_mode"], c["signal_mode"]) for c in cfgs}
    assert triples == {(idm, sg) for _, idm, sg in ARMS}


def test_names_are_readable():
    names = {c["name"] for c in sweep_signal.CONFIGS}
    assert names == {"max-baseline", "max-id", "max-signal", "max-id+signal"}


def test_arm_name_matches_its_toggles():
    by_name = {c["name"]: c for c in sweep_signal.CONFIGS}
    for suffix, id_mode, signal_mode in ARMS:
        c = by_name[f"max-{suffix}"]
        assert c["id_mode"] == id_mode
        assert c["signal_mode"] == signal_mode


def test_combined_arm_turns_both_on():
    """The id+signal arm runs the ID feature AND the signal overlay at the same time."""
    c = next(c for c in sweep_signal.CONFIGS if c["name"] == "max-id+signal")
    assert c["id_mode"] == "random"
    assert c["signal_mode"] == "on"


def test_configs_well_formed():
    for c in sweep_signal.CONFIGS:
        assert isinstance(c, dict)
        assert REQUIRED_KEYS <= set(c.keys())
        assert c["op"] == "max"
        assert c["id_mode"] in {"none", "random"}
        assert c["signal_mode"] in {"on", "off"}
        assert isinstance(c["train_N"], list) and len(c["train_N"]) >= 1


def test_shares_guardrail_base_with_message_sweep():
    """sweep_signal reuses sweep_messages.BASE (same guardrail regime, train_N, etc.)."""
    from sweep_messages import BASE
    c = next(c for c in sweep_signal.CONFIGS if c["name"] == "max-baseline")
    assert c["data"] == BASE["data"] == "guardrail"
    assert c["train_N"] == BASE["train_N"]
    assert c["H"] == BASE["H"]


def test_main_is_callable():
    assert callable(sweep_signal.main)
