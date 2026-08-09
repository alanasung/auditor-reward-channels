from types import SimpleNamespace
from cotobf.stages import STAGES
import json

def cfg():
    return SimpleNamespace(
        run=SimpleNamespace(seed=0, profile="pilot"),
        data=SimpleNamespace(n_items=20),
        model=SimpleNamespace(name="x", revision=None),
        eval=SimpleNamespace(layers=[1]),
        force_synthetic=True,
    )

def test_collect_validates_entailment(tmp_path):
    run = tmp_path / "r"
    c = cfg()
    STAGES["build_dataset"](c, run)
    out = STAGES["collect"](c, run)
    assert "entailment_validated" in out["metrics"]

def test_fit_ordering(tmp_path):
    run = tmp_path / "r"
    c = cfg()
    for s in ("build_dataset", "collect", "fit"):
        STAGES[s](c, run)
    fit = json.loads((run/"artifacts/fit/results.json").read_text())
    # Synthetic smoke gates claim headlines; raw ordering still lists all arms.
    assert fit["metrics"]["ordering_claim_ok"] is False
    assert fit["metrics"]["ordering_by_heldout_obfuscation"] == []
    assert set(fit["metrics"]["ordering_raw"]) == {
        "cot_token",
        "posthoc_summary",
        "output_only",
    }

def test_evaluate_lineages(tmp_path):
    run = tmp_path / "r"
    c = cfg()
    for s in ("build_dataset", "collect", "fit", "evaluate"):
        STAGES[s](c, run)
    assert (run/"artifacts/evaluate/lineages.json").is_file()
