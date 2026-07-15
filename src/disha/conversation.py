from typing import Dict

from disha.sutra.actions import ResponseAction


RESPONSES: Dict[ResponseAction, str] = {
    ResponseAction.CLARIFY: (
        "I want to understand rather than guess. Could you tell me a little more "
        "about what you are feeling or what happened?"
    ),
    ResponseAction.ACKNOWLEDGE: (
        "Thank you for sharing that with me. What would feel most useful to talk about next?"
    ),
    ResponseAction.EXPLORE: (
        "That sounds difficult. I am here to listen—what part of this is weighing on you most?"
    ),
    ResponseAction.ENCOURAGE_COPING: (
        "Would it help to choose one small, manageable step you can take right now?"
    ),
    ResponseAction.RECOMMEND_HUMAN_SUPPORT: (
        "This sounds like a lot to carry. If you can, consider reaching out to someone you "
        "trust or a qualified mental-health professional for support."
    ),
    ResponseAction.PROVIDE_CRISIS_RESOURCES: (
        "I am concerned by what you shared. If you may act on thoughts of harming yourself, "
        "contact your local emergency service now or go to the nearest emergency department. "
        "If you can, tell a trusted person nearby and stay with them while you seek help."
    ),
    ResponseAction.SAFE_FALLBACK: (
        "I may not have enough reliable information to respond safely. Please tell me more, "
        "or reach out to a trusted person if you need immediate support."
    ),
}


def render_response(action: ResponseAction) -> str:
    return RESPONSES.get(action, RESPONSES[ResponseAction.SAFE_FALLBACK])
