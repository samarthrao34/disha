import numpy as np

def expected_calibration_error(confidences, correctness, n_bins: int = 15) -> float:
    confidences = np.asarray(confidences, dtype=float)
    correctness = np.asarray(correctness, dtype=float)
    if confidences.shape != correctness.shape:
        raise ValueError("confidences and correctness must have same shape")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i+1]
        mask = (confidences >= lo) & (confidences <= hi) if i == 0 else (confidences > lo) & (confidences <= hi)
        if np.any(mask):
            ece += np.mean(mask) * abs(np.mean(confidences[mask]) - np.mean(correctness[mask]))
    return float(ece)
