from cotobf.cotobf.env import RewardHackEnv
from cotobf.cotobf.rl import PINNED_TRL_NOTE, _try_import_grpo, synthetic_grpo, train_with_grpo
from cotobf.cotobf.expert_iteration import expert_iteration_smoke

def test_synth():
    out=synthetic_grpo(RewardHackEnv(n_tasks=10,seed=0), reward_shape="cot_token", steps=4, seed=0)
    assert len(out["history"])==4

def test_train():
    assert "trl_status" in train_with_grpo(RewardHackEnv(n_tasks=10,seed=0), reward_shape="output_only", steps=3, seed=0)

def test_smoke_note():
    assert "not a substitute" in expert_iteration_smoke(RewardHackEnv(n_tasks=8,seed=0), reward_shape="output_only")["note"]

def test_trl_pin():
    assert "trl" in PINNED_TRL_NOTE
    assert isinstance(_try_import_grpo()[1], str)
