from cotobf.cotobf.env import RewardHackEnv
from cotobf.cotobf.lineages import DEFAULT_LINEAGES, run_lineage_comparison
from cotobf.cotobf.rollouts import collect_rollouts, synthesize_trajectory

def test_lineages():
    assert len(DEFAULT_LINEAGES)==3
    out=run_lineage_comparison(lineages=DEFAULT_LINEAGES[:2], steps=2, seed=0, n_tasks=8)
    assert "ordering_stable_across_lineages" in out

def test_rollout_keys():
    env=RewardHackEnv(n_tasks=4,seed=0)
    tr=synthesize_trajectory(env.to_dict()["tasks"][0], seed=0)
    assert {"cot","output","summary"} <= set(tr)
    assert len(collect_rollouts(env,n=3,seed=0))==3
