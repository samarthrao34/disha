from dataclasses import dataclass
from typing import Any, Dict, List

from disha.evidence.schema import EvidenceObject
from disha.sutra.actions import ResponseAction


@dataclass(frozen=True)
class SutraDecision:
    action: ResponseAction
    reasoning_trace: Dict[str, Any]


class SutraReasoner:
    """Conservative, inspectable SUTRA integration baseline.

    Decision cutoffs are research defaults and must be calibrated before any
    real-user deployment.
    """

    HIGH_UNCERTAINTY = 0.85
    NEGATIVE_EMOTIONS = {"angry", "disgust", "fear", "sad"}

    def _crisis_level(self, evidence: List[EvidenceObject]) -> str:
        rank = {"low": 0, "watch": 1, "elevated": 2, "high": 3, "critical": 4}
        levels = [
            str(item.quality_metadata.get("crisis", {}).get("level", "low"))
            for item in evidence
        ]
        return max(levels, key=lambda level: rank.get(level, 0), default="low")

    def _fuse(self, evidence: List[EvidenceObject]) -> Dict[str, float]:
        labels = sorted({label for item in evidence for label in item.emotion_probabilities})
        weights = []
        for item in evidence:
            confidence_factor = item.calibrated_confidence
            if confidence_factor is None:
                confidence_factor = 1.0
            weights.append(max(item.reliability_score * confidence_factor, 1e-6))
        total_weight = sum(weights)
        return {
            label: sum(
                weight * item.emotion_probabilities.get(label, 0.0)
                for weight, item in zip(weights, evidence)
            )
            / total_weight
            for label in labels
        }

    def decide(self, evidence: List[EvidenceObject]) -> SutraDecision:
        usable = [item for item in evidence if item.is_usable()]
        if not usable:
            return SutraDecision(ResponseAction.SAFE_FALLBACK, {"reason": "no_usable_evidence"})

        crisis_level = self._crisis_level(usable)
        if crisis_level in {"high", "critical"}:
            return SutraDecision(
                ResponseAction.PROVIDE_CRISIS_RESOURCES,
                {
                    "reason": "explicit_crisis_indicator",
                    "crisis_level": crisis_level,
                    "thresholds": "research_defaults",
                },
            )

        fused = self._fuse(usable)
        dominant = max(fused, key=fused.get)
        all_degraded = all(item.reliability_score < 0.60 for item in usable)
        all_uncertain = all(
            item.predictive_entropy is not None
            and item.predictive_entropy >= self.HIGH_UNCERTAINTY
            for item in usable
        )
        if all_degraded:
            action = ResponseAction.CLARIFY
            reason = "all_usable_evidence_degraded"
        elif all_uncertain:
            action = ResponseAction.CLARIFY
            reason = "all_usable_evidence_high_uncertainty"
        elif crisis_level == "elevated":
            action = ResponseAction.RECOMMEND_HUMAN_SUPPORT
            reason = "elevated_hopelessness_indicators"
        elif dominant in self.NEGATIVE_EMOTIONS:
            action = ResponseAction.EXPLORE
            reason = "negative_emotion_context"
        else:
            action = ResponseAction.ACKNOWLEDGE
            reason = "neutral_or_positive_context"

        return SutraDecision(
            action,
            {
                "reason": reason,
                "dominant_emotion": dominant,
                "fused_probabilities": fused,
                "crisis_level": crisis_level,
                "modalities_used": [item.modality.value for item in usable],
                "thresholds": "research_defaults",
            },
        )
