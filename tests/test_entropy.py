from disha.evidence.entropy import predictive_entropy

def test_entropy_low_for_certain_distribution():
    h = predictive_entropy({"sad": 0.99, "happy": 0.01})
    assert 0 <= h <= 1
    assert h < 0.2

def test_entropy_high_for_uniform_distribution():
    h = predictive_entropy({"sad": 0.5, "happy": 0.5})
    assert abs(h - 1.0) < 1e-6
