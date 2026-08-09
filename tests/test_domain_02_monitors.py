import pytest
from cotobf.cotobf.monitors import HELD_OUT_MONITOR, MONITOR_NAMES, apply_training_monitor, heldout_monitor

def test_names():
    assert set(MONITOR_NAMES)=={"cot_token","posthoc_summary","output_only"}
    assert HELD_OUT_MONITOR not in MONITOR_NAMES

@pytest.mark.parametrize("name", list(MONITOR_NAMES))
def test_each_monitor(name):
    out=apply_training_monitor(name, {"cot":"ok","summary":"ok","output":"return x * 2"})
    assert "flagged" in out

def test_heldout():
    assert heldout_monitor({"cot":"x","output":"return hardcoded"})["flagged"]

def test_cot_flag():
    assert apply_training_monitor("cot_token", {"cot":"I will special-case the test","output":"x"})["flagged"]

def test_bad():
    with pytest.raises(KeyError):
        apply_training_monitor("nope", {"cot":"","output":""})
