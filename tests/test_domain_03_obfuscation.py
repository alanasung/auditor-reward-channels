from cotobf.cotobf.env import RewardHackEnv
from cotobf.cotobf.monitors import apply_training_monitor
from cotobf.cotobf.obfuscation import entailment_score, obfuscation_index, validate_entailment_scorer
from cotobf.cotobf.rollouts import collect_rollouts

def test_entail_order():
    assert entailment_score("general solution for all inputs", "if x == 1: return 2 # cheat") < entailment_score("special-case the test", "if x == 1: return 2 # cheat")

def test_validate():
    env=RewardHackEnv(n_tasks=20,seed=0)
    assert "validated" in validate_entailment_scorer(collect_rollouts(env, obfuscate=0.7, cheat_rate=0.7, seed=0))

def test_index_keys():
    env=RewardHackEnv(n_tasks=12,seed=0)
    trs=collect_rollouts(env,seed=0)
    flags=[apply_training_monitor("cot_token",t)["flagged"] for t in trs]
    idx=obfuscation_index(trs,flags)
    assert "obfuscation_index_heldout" in idx
