"""Evidence object produced by the face modality for downstream fusion.

The evidence dictionary intentionally contains no raw image data, only
probabilities and metadata.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional

import numpy as np

from .calibration import entropy_from_probs


@dataclass
class FaceEvidence:
    modality: str
    emotion_probs: Dict[str, float]
    predicted_emotion: str
    confidence: float
    entropy: float
    reliability: float
    availability: bool
    timestamp: str
    model_name: str

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "emotion_probs": self.emotion_probs,
            "predicted_emotion": self.predicted_emotion,
            "confidence": self.confidence,
            "entropy": self.entropy,
            "reliability": self.reliability,
            "availability": self.availability,
            "timestamp": self.timestamp,
            "model_name": self.model_name,
        }


def build_face_evidence(
    probs: np.ndarray,
    class_names: List[str],
    model_name: str = "resnet18_fer2013",
    availability: bool = True,
    reliability: Optional[float] = None,
    timestamp: Optional[str] = None,
) -> dict:
    """Build a face-evidence dictionary from a single probability vector.

    Args:
        probs: 1-D array of shape (n_classes,) summing to 1.
        class_names: Class names aligned with ``probs``.
        model_name: Identifier of the producing model.
        availability: Whether the face modality was available for this sample.
        reliability: Optional externally supplied reliability score in [0, 1].
            If None, a simple heuristic (1 - normalized entropy) is used.
            This heuristic is a placeholder and is NOT a validated
            calibration-based reliability estimate.
        timestamp: ISO-8601 timestamp; defaults to current UTC time.

    Returns:
        Plain dictionary (no raw image data).
    """
    probs = np.asarray(probs, dtype=np.float64).reshape(-1)
    if len(probs) != len(class_names):
        raise ValueError(
            f"probs has {len(probs)} entries but class_names has {len(class_names)}"
        )
    total = probs.sum()
    if not np.isclose(total, 1.0, atol=1e-3):
        raise ValueError(f"probs must sum to 1, got {total:.6f}")

    pred_idx = int(probs.argmax())
    entropy_norm = float(entropy_from_probs(probs, normalize=True))
    entropy_nats = float(entropy_from_probs(probs, normalize=False))

    if reliability is None:
        reliability = 1.0 - entropy_norm

    evidence = FaceEvidence(
        modality="face",
        emotion_probs={name: float(p) for name, p in zip(class_names, probs)},
        predicted_emotion=class_names[pred_idx],
        confidence=float(probs[pred_idx]),
        entropy=entropy_nats,
        reliability=float(reliability),
        availability=bool(availability),
        timestamp=timestamp or datetime.now(timezone.utc).isoformat(),
        model_name=model_name,
    )
    return evidence.to_dict()
