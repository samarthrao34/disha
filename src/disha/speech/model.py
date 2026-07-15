from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib

from disha.speech.features import extract_file_features


DEFAULT_MODEL_PATH = Path("checkpoints/speech_ravdess_mfcc_svm.joblib")

CANONICAL_MAP = {
    "neutral": "neutral",
    "calm": "neutral",
    "happy": "happy",
    "sad": "sad",
    "angry": "angry",
    "fearful": "fear",
    "disgust": "disgust",
    "surprised": "surprise",
}


@lru_cache(maxsize=2)
def load_speech_model(path: str = str(DEFAULT_MODEL_PATH)):
    model_path = Path(path)
    if not model_path.is_file():
        return None
    return joblib.load(model_path)


def predict_speech_file(path: str | Path, model_path: str = str(DEFAULT_MODEL_PATH)) -> Dict[str, float]:
    payload = load_speech_model(model_path)
    if payload is None:
        raise FileNotFoundError(f"trained speech model not found: {model_path}")
    features = extract_file_features(path).reshape(1, -1)
    probabilities = payload["pipeline"].predict_proba(features)[0]
    canonical = {label: 0.0 for label in ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")}
    for label, probability in zip(payload["classes"], probabilities):
        canonical[CANONICAL_MAP[label]] += float(probability)
    return canonical
