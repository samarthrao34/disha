from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict


@dataclass(frozen=True)
class EmotionState:
    turn: int
    dominant_emotion: str
    current_probability: float
    previous_probability: float | None
    trend: str
    smoothed_probabilities: Dict[str, float]


class EmotionTracker:
    """In-memory session tracker; stores probabilities, never raw inputs."""

    def __init__(self, max_turns: int = 50, alpha: float = 0.35) -> None:
        self.history: Deque[Dict[str, float]] = deque(maxlen=max_turns)
        self.alpha = alpha
        self.smoothed: Dict[str, float] = {}

    def update(self, probabilities: Dict[str, float]) -> EmotionState:
        previous = self.history[-1] if self.history else None
        self.history.append(dict(probabilities))
        if not self.smoothed:
            self.smoothed = dict(probabilities)
        else:
            labels = set(self.smoothed) | set(probabilities)
            self.smoothed = {
                label: (1.0 - self.alpha) * self.smoothed.get(label, 0.0)
                + self.alpha * probabilities.get(label, 0.0)
                for label in labels
            }
        dominant = max(probabilities, key=probabilities.get)
        current_value = probabilities[dominant]
        previous_value = None if previous is None else previous.get(dominant, 0.0)
        if previous_value is None or abs(current_value - previous_value) < 0.05:
            trend = "stable"
        elif current_value > previous_value:
            trend = "increasing"
        else:
            trend = "decreasing"
        return EmotionState(
            turn=len(self.history),
            dominant_emotion=dominant,
            current_probability=current_value,
            previous_probability=previous_value,
            trend=trend,
            smoothed_probabilities=dict(self.smoothed),
        )
