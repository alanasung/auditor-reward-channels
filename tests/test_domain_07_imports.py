def test_imports():
    from cotobf.cotobf import env, monitors, rollouts, rl, expert_iteration, lineages, obfuscation, pipeline
    assert env.detect_cheat

def test_stages():
    from cotobf import stages
    assert len(stages.STAGES)==5
