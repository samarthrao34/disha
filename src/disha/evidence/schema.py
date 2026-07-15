from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Dict, Optional


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
    """Canonical modality evidence passed to SUTRA.

    Raw images, audio, and text are intentionally excluded. Reliability is an
    input-quality estimate; predictive entropy is model uncertainty. Keeping
    them separate prevents a confident model from making a poor sensor sample
    look reliable (and vice versa).
    """

    modality: Modality
    emotion_probabilities: Dict[str, float]
    calibrated_confidence: Optional[float]
    predictive_entropy: Optional[float]
    reliability_score: float
    reliability_status: ReliabilityStatus
    availability_status: AvailabilityStatus
    timestamp: float
    quality_metadata: Dict[str, Any] = field(default_factory=dict)
    model_name: Optional[str] = None

    def __post_init__(self) -> None:
        if not self.emotion_probabilities:
            raise ValueError("emotion_probabilities cannot be empty")
        probabilities = [float(value) for value in self.emotion_probabilities.values()]
        if any(value < 0.0 or value > 1.0 for value in probabilities):
            raise ValueError("emotion probabilities must be in [0, 1]")
        if abs(sum(probabilities) - 1.0) > 1e-3:
            raise ValueError("emotion probabilities must sum to 1")
        if not 0.0 <= float(self.reliability_score) <= 1.0:
            raise ValueError("reliability_score must be in [0, 1]")
        for name, value in (
            ("calibrated_confidence", self.calibrated_confidence),
            ("predictive_entropy", self.predictive_entropy),
        ):
            if value is not None and not 0.0 <= float(value) <= 1.0:
                raise ValueError(f"{name} must be in [0, 1]")

    @property
    def predicted_emotion(self) -> str:
        return max(self.emotion_probabilities, key=self.emotion_probabilities.get)

    def is_usable(self) -> bool:
        return (
            self.availability_status != AvailabilityStatus.UNAVAILABLE
            and self.reliability_status != ReliabilityStatus.UNUSABLE
        )

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload["modality"] = self.modality.value
        payload["reliability_status"] = self.reliability_status.value
        payload["availability_status"] = self.availability_status.value
        payload["predicted_emotion"] = self.predicted_emotion
        return payload
