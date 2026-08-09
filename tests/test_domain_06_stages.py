from types import SimpleNamespace
from cotobf.stages import STAGES
import json

def cfg():
    return SimpleNamespace(run=SimpleNamespace(seed=0,profile="smoke"), data=SimpleNamespace(n_items=20), model=SimpleNamespace(name="x"), eval=SimpleNamespace(layers=[1]))

def test_registry():
    assert set(STAGES)=={"build_dataset","collect","fit","evaluate","report"}

def test_e2e(tmp_path):
    run=tmp_path/"r"; c=cfg()
    for n in ["build_dataset","collect","fit","evaluate","report"]:
        out=STAGES[n](c,run)
        assert out["task"]==n and "metrics" in out

def test_heldout_ordering(tmp_path):
    run=tmp_path/"r"; c=cfg()
    for n in ["build_dataset","collect","fit"]:
        STAGES[n](c,run)
    fit=json.loads((run/"artifacts/fit/results.json").read_text())
    assert len(fit["metrics"]["ordering_by_heldout_obfuscation"])==3
