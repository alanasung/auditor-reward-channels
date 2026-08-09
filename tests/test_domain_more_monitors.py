import pytest
from cotobf.cotobf.monitors import MONITOR_NAMES, apply_training_monitor, heldout_monitor

@pytest.mark.parametrize("name", MONITOR_NAMES)
def test_monitor_returns_flagged(name):
    tr = {"cot": "ok", "summary": "ok", "output": "return x * 2"}
    out = apply_training_monitor(name, tr)
    assert "flagged" in out
    assert out["monitor"] == name

def test_bad_monitor():
    with pytest.raises(KeyError):
        apply_training_monitor("nope", {"cot": "", "output": ""})

def test_heldout_on_cheat_code():
    tr = {"cot": "benign summary text", "output": "return hardcoded"}
    assert heldout_monitor(tr)["flagged"]
