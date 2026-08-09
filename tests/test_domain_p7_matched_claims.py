"""P7: FPR-matched monitors, synthetic GRPO gating, obfuscation curve, pilot honesty."""

from __future__ import annotations

from omegaconf import OmegaConf

from cotobf.cotobf.env import RewardHackEnv
from cotobf.cotobf.lineages import run_lineage_comparison
from cotobf.cotobf.monitors import (
    apply_training_monitor,
    fit_fpr_matched_thresholds,
    power_status,
)
from cotobf.cotobf.obfuscation import obfuscation_over_training_curve
from cotobf.cotobf.pipeline import (
    stage_build_dataset,
    stage_collect,
    stage_evaluate,
    stage_fit,
    stage_report,
)
from cotobf.cotobf.rl import synthetic_grpo
from cotobf.cotobf.rollouts import synthesize_trajectory


def _mixed_rollouts(n: int = 24, seed: int = 0) -> list[dict]:
    env = RewardHackEnv(n_tasks=n, seed=seed)
    rows = []
    for i, task in enumerate(env.to_dict()["tasks"]):
        rows.append(
            synthesize_trajectory(
                task, seed=seed + i, obfuscate=0.3 if i % 2 else 0.0, cheat_rate=0.4
            )
        )
    return rows


def test_fpr_matched_thresholds_stamp():
    rollouts = _mixed_rollouts(32, seed=1)
    fit = fit_fpr_matched_thresholds(rollouts, target_fpr=0.10, seed=1)
    assert "thresholds" in fit
    assert set(fit["thresholds"]) >= {"cot", "summary", "output"}
    assert "fpr_matched" in fit
    assert fit["n_negatives_holdout"] >= 1
    # Applying with thresholds stamps fpr_matched when fit succeeded.
    tr = rollouts[0]
    out = apply_training_monitor(
        "cot_token",
        tr,
        thresholds=fit["thresholds"],
        fpr_matched=bool(fit["fpr_matched"]),
    )
    assert out["monitor_matched_strength"] is True
    assert "threshold" in out
    assert out["fpr_matched"] is bool(fit["fpr_matched"])


def test_synthetic_grpo_is_synthetic_and_gated():
    env = RewardHackEnv(n_tasks=6, seed=0)
    out = synthetic_grpo(env, reward_shape="cot_token", steps=2, seed=0)
    assert out["mode"] == "synthetic_grpo"
    assert out["is_synthetic"] is True
    assert out["claims_gated"] is True


def test_obfuscation_evolves_requires_measured_steps():
    # Rising measured curve with enough steps → may claim.
    hist = [{"obfuscation_index_heldout": 0.1 * i} for i in range(6)]
    measured = obfuscation_over_training_curve(hist, measured=True, min_steps=3, seed=0)
    assert measured["n_steps"] == 6
    assert measured["slope"] > 0
    assert measured["obfuscation_evolves"] is True
    # Same curve but synthetic → no claim.
    synth = obfuscation_over_training_curve(hist, measured=False, min_steps=3, seed=0)
    assert synth["obfuscation_evolves"] is False
    # Too few steps → no claim.
    short = obfuscation_over_training_curve(hist[:2], measured=True, min_steps=3, seed=0)
    assert short["obfuscation_evolves"] is False


def test_lineage_requires_fpr_matched():
    out = run_lineage_comparison(
        lineages=("a", "b"), steps=1, seed=0, n_tasks=4, force_synthetic=True, fpr_matched=False
    )
    assert out["any_lineage_synthetic"] is True
    assert out["ordering_claim_ok"] is False
    assert out["fpr_matched"] is False


def test_power_status_honest():
    assert power_status(32) == "micro"
    assert power_status(128) == "underpowered"
    assert power_status(512) == "powered"


def test_pipeline_gates_synthetic_ordering(tmp_path):
    cfg = OmegaConf.create(
        {
            "force_synthetic": True,
            "run": {"seed": 0},
            "data": {"n_items": 24},
            "model": {"name": "synthetic", "revision": None},
        }
    )
    run_dir = tmp_path / "run"
    stage_build_dataset(cfg, run_dir)
    stage_collect(cfg, run_dir)
    fit = stage_fit(cfg, run_dir)
    assert fit["metrics"]["fpr_matched"] in {True, False}
    assert "monitor_thresholds" in fit["metrics"]
    assert fit["metrics"]["any_synthetic_rl"] is True
    assert fit["metrics"]["ordering_claim_ok"] is False
    assert fit["metrics"]["ordering_by_heldout_obfuscation"] == []
    assert fit["metrics"]["power_status"] == "micro"
    assert fit["metrics"]["obfuscation_evolves"] is False
    ev = stage_evaluate(cfg, run_dir)
    assert ev["metrics"]["ordering_claim_ok"] is False
    assert ev["metrics"]["fpr_matched"] == fit["metrics"]["fpr_matched"]
    rep = stage_report(cfg, run_dir)
    assert rep["metrics"]["ordering"] == []
    assert rep["metrics"]["ordering_claim_ok"] is False
    assert rep["metrics"]["obfuscation_evolves"] is False
    assert "power_status" in rep["metrics"]
