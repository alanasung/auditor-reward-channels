def test_imports():
    from cotobf.cotobf import env
    assert env.detect_cheat

def test_stages():
    from cotobf import stages
    assert len(stages.STAGES)==5
