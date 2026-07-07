import numpy as np

def rms_energy(audio: np.ndarray) -> float:
    if audio is None or len(audio) == 0:
        raise ValueError("audio cannot be empty")
    return float(np.sqrt(np.mean(np.square(audio))))

def clipping_ratio(audio: np.ndarray, threshold: float = 0.99) -> float:
    if audio is None or len(audio) == 0:
        raise ValueError("audio cannot be empty")
    return float(np.mean(np.abs(audio) >= threshold))
