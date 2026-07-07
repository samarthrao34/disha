from dataclasses import dataclass
from disha.sutra.actions import ResponseAction

@dataclass
class SafetyDecision:
    approved: bool
    action: ResponseAction
    reason: str

class SafetyPolicy:
    """Safety-governance scaffold, not a clinical safety system."""
    def approve_action(self, action: ResponseAction) -> SafetyDecision:
        return SafetyDecision(True, action, "placeholder_policy_approved")
