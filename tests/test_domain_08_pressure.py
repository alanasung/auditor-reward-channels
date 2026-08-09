from cotobf.cotobf.env import RewardHackEnv
from cotobf.cotobf.rl import synthetic_grpo

def test_cot_ge_output():
    env=RewardHackEnv(n_tasks=16,seed=0)
    cot=synthetic_grpo(env,reward_shape="cot_token",steps=6,seed=0)["final"]["obfuscation_index_heldout"]
    out=synthetic_grpo(env,reward_shape="output_only",steps=6,seed=0)["final"]["obfuscation_index_heldout"]
    assert cot >= out - 1e-9
