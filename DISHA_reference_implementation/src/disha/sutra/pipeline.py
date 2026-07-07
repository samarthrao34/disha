from dataclasses import dataclass
from typing import List, Dict, Any
from disha.evidence.schema import EvidenceObject
from disha.sutra.actions import ResponseAction

@dataclass
class SutraDecision:
    action: ResponseAction
    reasoning_trace: Dict[str, Any]

class SutraReasoner:
    """Conservative scaffold. Operational thresholds must be calibrated before research use."""
    def decide(self, evidence: List[EvidenceObject]) -> SutraDecision:
        usable = [e for e in evidence if e.is_usable()]
        if not usable:
            return SutraDecision(ResponseAction.SAFE_FALLBACK, {"reason": "no_usable_evidence"})
        high_uncertainty = any(e.predictive_entropy is not None and e.predictive_entropy > 0.80 for e in usable)
        if high_uncertainty:
            return SutraDecision(ResponseAction.CLARIFY, {"reason": "high_uncertainty_placeholder", "warning": "threshold requires calibration"})
        return SutraDecision(ResponseAction.ACKNOWLEDGE, {"reason": "default_conservative_action"})
