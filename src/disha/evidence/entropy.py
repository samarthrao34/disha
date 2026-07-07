import math
from typing import Dict

def predictive_entropy(probabilities: Dict[str, float], normalize: bool = True) -> float:
    if not probabilities:
        raise ValueError("probabilities cannot be empty")
    total = sum(probabilities.values())
    if total <= 0:
        raise ValueError("probability sum must be positive")
    probs = [max(float(v) / total, 1e-12) for v in probabilities.values()]
    h = -sum(p * math.log(p) for p in probs)
    return h / math.log(len(probs)) if normalize and len(probs) > 1 else h
