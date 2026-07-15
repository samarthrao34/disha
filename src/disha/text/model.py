"""Trained GoEmotions baseline used by the professor-demo runtime."""

from functools import lru_cache
from pathlib import Path
from typing import Dict

import joblib


MELD_MODEL_PATH = Path("checkpoints/text_meld_tfidf_logreg.joblib")
GOEMOTIONS_MODEL_PATH = Path("checkpoints/text_goemotions_tfidf_sgd.joblib")
DEFAULT_MODEL_PATH = MELD_MODEL_PATH if MELD_MODEL_PATH.is_file() else GOEMOTIONS_MODEL_PATH

# Transparent projection from GoEmotions' 28 labels to DISHA's seven-class
# evidence space. The 28-label benchmark is reported separately without this
# projection.
BASIC_EMOTION_MAP = {
    "angry": {"anger", "annoyance", "disapproval"},
    "disgust": {"disgust"},
    "fear": {"fear", "nervousness"},
    "happy": {
        "admiration", "amusement", "approval", "caring", "desire",
        "excitement", "gratitude", "joy", "love", "optimism", "pride", "relief",
    },
    "neutral": {"neutral"},
    "sad": {"disappointment", "embarrassment", "grief", "remorse", "sadness"},
    "surprise": {"confusion", "curiosity", "realization", "surprise"},
}


@lru_cache(maxsize=2)
def load_text_model(path: str = str(DEFAULT_MODEL_PATH)):
    model_path = Path(path)
    if not model_path.is_file():
        return None
    return joblib.load(model_path)


def predict_basic_emotions(text: str, path: str = str(DEFAULT_MODEL_PATH)) -> Dict[str, float] | None:
    payload = load_text_model(path)
    if payload is None:
        return None
    vector = payload["vectorizer"].transform([text])
    fine_probs = payload["classifier"].predict_proba(vector)[0]
    if payload.get("canonical_labels"):
        direct = dict(zip(payload["labels"], (float(value) for value in fine_probs)))
        total = sum(direct.values())
        return {label: direct.get(label, 0.0) / total for label in BASIC_EMOTION_MAP}
    fine = dict(zip(payload["labels"], (float(value) for value in fine_probs)))
    basic = {
        basic_label: sum(fine.get(label, 0.0) for label in fine_labels)
        for basic_label, fine_labels in BASIC_EMOTION_MAP.items()
    }
    total = sum(basic.values())
    if total <= 0:
        return {label: 1.0 / len(basic) for label in basic}
    return {label: value / total for label, value in basic.items()}
