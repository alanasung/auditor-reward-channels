from cotobf.cotobf.env import RewardHackEnv
from cotobf.cotobf.rl import synthetic_grpo
from cotobf.cotobf.monitors import MONITOR_NAMES

def test_pressure_ordering_tendency():
    env = RewardHackEnv(n_tasks=20, seed=0)
    finals = {s: synthetic_grpo(env, reward_shape=s, steps=8, seed=0)["final"]["obfuscation_index_heldout"] for s in MONITOR_NAMES}
    # cot pressure should be >= output_only in this synthetic dynamics
    assert finals["cot_token"] >= finals["output_only"] - 1e-9

def test_history_steps_field():
    env = RewardHackEnv(n_tasks=10, seed=1)
    out = synthetic_grpo(env, reward_shape="output_only", steps=3, seed=1)
    assert [h["step"] for h in out["history"]] == [0, 1, 2]
