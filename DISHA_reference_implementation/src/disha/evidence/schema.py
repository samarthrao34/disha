from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, Optional, Any

class Modality(str, Enum):
    FACE = "face"
    SPEECH = "speech"
    TEXT = "text"

class AvailabilityStatus(str, Enum):
    AVAILABLE = "available"
    DEGRADED = "degraded"
    UNAVAILABLE = "unavailable"

class ReliabilityStatus(str, Enum):
    RELIABLE = "reliable"
    DEGRADED = "degraded"
    UNUSABLE = "unusable"

@dataclass(frozen=True)
class EvidenceObject:
    """Structured evidence passed from Perception Layer to SUTRA. No raw sensor data."""
    modality: Modality
    emotion_probabilities: Dict[str, float]
    calibrated_confidence: Optional[float]
    predictive_entropy: Optional[float]
    reliability_status: ReliabilityStatus
    availability_status: AvailabilityStatus
    timestamp: float
    quality_metadata: Dict[str, Any] = field(default_factory=dict)

    def is_usable(self) -> bool:
        return self.availability_status != AvailabilityStatus.UNAVAILABLE and self.reliability_status != ReliabilityStatus.UNUSABLE
