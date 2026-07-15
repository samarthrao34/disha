"""Canonical evidence produced by the face-emotion modality."""

import time
from typing import List, Optional

import numpy as np

from disha.evidence.schema import (
    AvailabilityStatus,
    EvidenceObject,
    Modality,
    ReliabilityStatus,
)
from disha.face.reliability import FaceQualityAssessment

from .calibration import entropy_from_probs


def build_face_evidence(
    probs: np.ndarray,
    class_names: List[str],
    model_name: str = "resnet18_fer2013",
    calibrated: bool = False,
    availability: bool = True,
    reliability: Optional[float] = None,
    reliability_status: Optional[ReliabilityStatus] = None,
    quality_metadata: Optional[dict] = None,
    quality_assessment: Optional[FaceQualityAssessment] = None,
    timestamp: Optional[float] = None,
) -> EvidenceObject:
    """Build canonical face evidence from one calibrated probability vector."""
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if len(probs) != len(class_names):
        raise ValueError(
            f"probs has {len(probs)} entries but class_names has {len(class_names)}"
        )
    if np.any(probs < 0.0) or np.any(probs > 1.0):
        raise ValueError("probs must be in [0, 1]")
    total = probs.sum()
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(f"probs must sum to 1, got {total:.6f}")

    if quality_assessment is not None:
        reliability = quality_assessment.score
        reliability_status = quality_assessment.status
        quality_metadata = quality_assessment.metadata

    if not availability:
        reliability = 0.0
        reliability_status = ReliabilityStatus.UNUSABLE
        availability_status = AvailabilityStatus.UNAVAILABLE
    else:
        reliability = 1.0 if reliability is None else float(reliability)
        if reliability_status is None:
            reliability_status = (
                ReliabilityStatus.RELIABLE
                if reliability >= 0.60
                else ReliabilityStatus.DEGRADED
                if reliability >= 0.30
                else ReliabilityStatus.UNUSABLE
            )
        availability_status = (
            AvailabilityStatus.AVAILABLE
            if reliability_status == ReliabilityStatus.RELIABLE
            else AvailabilityStatus.DEGRADED
        )

    entropy_normalized = float(entropy_from_probs(probs, normalize=True))
    return EvidenceObject(
        modality=Modality.FACE,
        emotion_probabilities={name: float(prob) for name, prob in zip(class_names, probs)},
        calibrated_confidence=float(probs.max()) if calibrated else None,
        predictive_entropy=entropy_normalized,
        reliability_score=reliability,
        reliability_status=reliability_status,
        availability_status=availability_status,
        timestamp=time.time() if timestamp is None else float(timestamp),
        quality_metadata=quality_metadata or {},
        model_name=model_name,
    )
