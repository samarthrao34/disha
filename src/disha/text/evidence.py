"""Deterministic text evidence baseline for end-to-end integration testing.

This module is deliberately small and interpretable. It is not a substitute
for the planned trained text-emotion and crisis-risk models.
"""

import re
import time
from typing import Dict, Iterable, List

from disha.evidence.entropy import predictive_entropy
from disha.evidence.schema import (
    AvailabilityStatus,
    EvidenceObject,
    Modality,
    ReliabilityStatus,
)
from disha.text.model import predict_basic_emotions


EMOTIONS = ("angry", "disgust", "fear", "happy", "neutral", "sad", "surprise")

EMOTION_TERMS: Dict[str, Iterable[str]] = {
    "angry": ("angry", "furious", "mad", "annoyed", "frustrated", "hate"),
    "disgust": ("disgusted", "disgusting", "gross", "revolting"),
    "fear": ("afraid", "anxious", "scared", "terrified", "worried", "panic"),
    "happy": ("happy", "glad", "great", "joy", "excited", "proud", "better"),
    "sad": ("sad", "down", "depressed", "lonely", "miserable", "hopeless", "crying"),
    "surprise": ("surprised", "shocked", "unexpected", "wow"),
}

DIRECT_CRISIS_PATTERNS = (
    r"\bkill myself\b",
    r"\bend my life\b",
    r"\btake my life\b",
    r"\bhurt myself\b",
    r"\bsuicide\b",
    r"\bdon't want to live\b",
    r"\bdo not want to live\b",
)

HOPELESSNESS_PATTERNS = (
    r"\bno point\b",
    r"\bnothing will ever\b",
    r"\bcan't go on\b",
    r"\bcannot go on\b",
    r"\bno way out\b",
)


def _matched_patterns(text: str, patterns: Iterable[str]) -> List[str]:
    return [pattern for pattern in patterns if re.search(pattern, text, flags=re.IGNORECASE)]


def assess_crisis_indicators(text: str) -> Dict[str, object]:
    """Return bounded rule signals; this does not diagnose or estimate probability."""
    direct = _matched_patterns(text, DIRECT_CRISIS_PATTERNS)
    hopelessness = _matched_patterns(text, HOPELESSNESS_PATTERNS)
    if direct:
        level = "high"
    elif len(hopelessness) >= 2:
        level = "elevated"
    elif hopelessness:
        level = "watch"
    else:
        level = "low"
    return {
        "level": level,
        "direct_indicator_count": len(direct),
        "hopelessness_indicator_count": len(hopelessness),
        # Store categories/counts, not the user's raw text.
        "indicator_categories": [
            category
            for category, matches in (("direct_self_harm", direct), ("hopelessness", hopelessness))
            if matches
        ],
        "method": "deterministic_integration_baseline",
    }


def _emotion_probabilities(text: str) -> Dict[str, float]:
    lowered = text.lower()
    scores = {emotion: 0.05 for emotion in EMOTIONS}
    scores["neutral"] = 0.20
    matches = 0
    for emotion, terms in EMOTION_TERMS.items():
        count = sum(bool(re.search(rf"\b{re.escape(term)}\b", lowered)) for term in terms)
        if count:
            scores[emotion] += 0.55 * count
            matches += count
    if matches:
        scores["neutral"] = 0.05
    total = sum(scores.values())
    return {emotion: value / total for emotion, value in scores.items()}


def build_text_evidence(text: str, timestamp: float | None = None) -> EvidenceObject:
    """Convert text to sanitized canonical evidence without retaining raw text."""
    if text is None or not text.strip():
        uniform = {emotion: 1.0 / len(EMOTIONS) for emotion in EMOTIONS}
        return EvidenceObject(
            modality=Modality.TEXT,
            emotion_probabilities=uniform,
            calibrated_confidence=None,
            predictive_entropy=1.0,
            reliability_score=0.0,
            reliability_status=ReliabilityStatus.UNUSABLE,
            availability_status=AvailabilityStatus.UNAVAILABLE,
            timestamp=time.time() if timestamp is None else timestamp,
            quality_metadata={"token_count": 0, "crisis": {"level": "low"}},
            model_name="text_rules_integration_baseline",
        )

    trained_probabilities = predict_basic_emotions(text)
    probabilities = trained_probabilities or _emotion_probabilities(text)
    token_count = len(text.split())
    signal_count = sum(
        1
        for terms in EMOTION_TERMS.values()
        for term in terms
        if re.search(rf"\b{re.escape(term)}\b", text, flags=re.IGNORECASE)
    )
    length_score = min(token_count / 8.0, 1.0)
    signal_score = min(signal_count / 2.0, 1.0)
    reliability = 0.35 + 0.35 * length_score + 0.30 * signal_score
    status = (
        ReliabilityStatus.RELIABLE
        if reliability >= 0.60
        else ReliabilityStatus.DEGRADED
    )
    return EvidenceObject(
        modality=Modality.TEXT,
        emotion_probabilities=probabilities,
        # Rule scores are not calibrated probabilities.
        calibrated_confidence=None,
        predictive_entropy=predictive_entropy(probabilities),
        reliability_score=float(reliability),
        reliability_status=status,
        availability_status=(
            AvailabilityStatus.AVAILABLE
            if status == ReliabilityStatus.RELIABLE
            else AvailabilityStatus.DEGRADED
        ),
        timestamp=time.time() if timestamp is None else timestamp,
        quality_metadata={
            "token_count": token_count,
            "emotion_signal_count": signal_count,
            "crisis": assess_crisis_indicators(text),
            "thresholds": "research_defaults",
        },
        model_name=(
            "goemotions_tfidf_sgd"
            if trained_probabilities is not None
            else "text_rules_integration_baseline"
        ),
    )
