"""Measured-path APIs with Hub downloads monkeypatched away."""

from __future__ import annotations

from cotobf.cotobf.env import RewardHackEnv
from cotobf.cotobf.obfuscation import LABELED_ENTAILMENT_PAIRS, validate_entailment_scorer
from cotobf.cotobf.rl import synthetic_grpo, train_with_grpo
from cotobf.cotobf.rollouts import collect_rollouts, synthesize_trajectory
from cotobf.cotobf.model_runtime import try_load_causal_lm


def test_hand_labeled_entailment_validation():
    assert len(LABELED_ENTAILMENT_PAIRS) >= 4
    val = validate_entailment_scorer([])
    assert val["validated"] is True
    assert val["hand_labeled_accuracy"] >= 0.8


def test_force_synthetic_rollouts_no_hub(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("Hub load must not run under force_synthetic")

    monkeypatch.setattr("cotobf.cotobf.rollouts.try_load_causal_lm", boom)
    env = RewardHackEnv(n_tasks=4, seed=0)
    trs = collect_rollouts(env, n=3, seed=0, force_synthetic=True)
    assert len(trs) == 3
    assert all(t["mode"] == "synthetic" for t in trs)


def test_train_force_synthetic(monkeypatch):
    def boom(*_a, **_k):
        raise AssertionError("local PG must not run under force_synthetic")

    monkeypatch.setattr("cotobf.cotobf.rl.local_policy_gradient", boom)
    out = train_with_grpo(
        RewardHackEnv(n_tasks=8, seed=0),
        reward_shape="cot_token",
        steps=3,
        seed=0,
        force_synthetic=True,
    )
    assert out["mode"] == "synthetic_grpo"
    assert out["fallback_reason"] == "force_synthetic=True"


def test_try_load_respects_force_synthetic():
    assert try_load_causal_lm("gpt2", force_synthetic=True) is None


def test_synth_trajectory_keys():
    env = RewardHackEnv(n_tasks=2, seed=0)
    tr = synthesize_trajectory(env.to_dict()["tasks"][0], seed=0)
    assert {"cot", "output", "summary", "mode"} <= set(tr)


def test_synthetic_grpo_history():
    out = synthetic_grpo(RewardHackEnv(n_tasks=8, seed=0), reward_shape="output_only", steps=2, seed=0)
    assert len(out["history"]) == 2
