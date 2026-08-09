from types import SimpleNamespace
from cotobf.stages import STAGES
import json

def cfg():
    return SimpleNamespace(
        run=SimpleNamespace(seed=0, profile="smoke"),
        data=SimpleNamespace(n_items=20),
        model=SimpleNamespace(name="x", revision=None),
        eval=SimpleNamespace(layers=[1]),
        force_synthetic=True,
    )

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
    # Smoke/synthetic RL gates claim headlines; raw ordering still has 3 arms.
    assert fit["metrics"]["ordering_claim_ok"] is False
    assert fit["metrics"]["ordering_by_heldout_obfuscation"] == []
    assert len(fit["metrics"]["ordering_raw"]) == 3
