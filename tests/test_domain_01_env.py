from cotobf.cotobf.env import RewardHackEnv, detect_cheat

def test_cheat_yes():
    assert detect_cheat("if x == 3: return 6 # cheat")

def test_cheat_no():
    assert not detect_cheat("def f(x):\n    return x * 2\n")

def test_env_len():
    assert len(RewardHackEnv(n_tasks=9, seed=0).tasks) == 9

def test_score_pass():
    env=RewardHackEnv(n_tasks=3,seed=0)
    assert env.score_solution(env.tasks[0], "def f(x):\n    return x * 2\n")["passed"]

def test_to_dict():
    assert RewardHackEnv(n_tasks=4,seed=1).to_dict()["n_tasks"]==4
