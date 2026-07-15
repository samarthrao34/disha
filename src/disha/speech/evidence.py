import time
from pathlib import Path

import numpy as np

from disha.evidence.entropy import predictive_entropy
from disha.evidence.schema import (
    AvailabilityStatus,
    EvidenceObject,
    Modality,
    ReliabilityStatus,
)
from disha.speech.features import SAMPLE_RATE, load_audio
from disha.speech.model import predict_speech_file


def build_speech_evidence(path: str | Path, timestamp: float | None = None) -> EvidenceObject:
    audio = load_audio(path)
    probabilities = predict_speech_file(path)
    rms = float(np.sqrt(np.mean(np.square(audio))))
    clipping = float(np.mean(np.abs(audio) >= 0.99))
    duration = float(len(audio) / SAMPLE_RATE)
    energy_score = float(np.clip(rms / 0.08, 0.0, 1.0))
    duration_score = float(np.clip(duration / 2.0, 0.0, 1.0))
    clipping_score = 1.0 - float(np.clip(clipping / 0.02, 0.0, 1.0))
    reliability = 0.40 * energy_score + 0.35 * duration_score + 0.25 * clipping_score
    status = (
        ReliabilityStatus.RELIABLE
        if reliability >= 0.60
        else ReliabilityStatus.DEGRADED
        if reliability >= 0.30
        else ReliabilityStatus.UNUSABLE
    )
    return EvidenceObject(
        modality=Modality.SPEECH,
        emotion_probabilities=probabilities,
        calibrated_confidence=None,
        predictive_entropy=predictive_entropy(probabilities),
        reliability_score=float(reliability),
        reliability_status=status,
        availability_status=(
            AvailabilityStatus.AVAILABLE
            if status == ReliabilityStatus.RELIABLE
            else AvailabilityStatus.DEGRADED
            if status == ReliabilityStatus.DEGRADED
            else AvailabilityStatus.UNAVAILABLE
        ),
        timestamp=time.time() if timestamp is None else timestamp,
        quality_metadata={
            "duration_seconds": duration,
            "rms_energy": rms,
            "clipping_ratio": clipping,
            "thresholds": "research_defaults",
        },
        model_name="ravdess_mfcc_svm",
    )
