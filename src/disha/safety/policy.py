from dataclasses import dataclass

from disha.sutra.actions import ResponseAction


@dataclass(frozen=True)
class SafetyDecision:
    approved: bool
    action: ResponseAction
    reason: str


class SafetyPolicy:
    """Bounded action policy for the integration baseline."""

    def approve_action(
        self,
        action: ResponseAction,
        *,
        crisis_level: str = "low",
    ) -> SafetyDecision:
        if crisis_level in {"high", "critical"}:
            return SafetyDecision(
                True,
                ResponseAction.PROVIDE_CRISIS_RESOURCES,
                "crisis_override",
            )
        if action == ResponseAction.PROVIDE_CRISIS_RESOURCES and crisis_level == "low":
            return SafetyDecision(False, ResponseAction.SAFE_FALLBACK, "unsupported_crisis_action")
        return SafetyDecision(True, action, "bounded_action_approved")
