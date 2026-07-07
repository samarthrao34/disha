from enum import Enum

class ResponseAction(str, Enum):
    CLARIFY = "clarify"
    ACKNOWLEDGE = "acknowledge"
    EXPLORE = "explore"
    ENCOURAGE_COPING = "encourage_coping"
    RECOMMEND_HUMAN_SUPPORT = "recommend_human_support"
    PROVIDE_CRISIS_RESOURCES = "provide_crisis_resources"
    SAFE_FALLBACK = "safe_fallback"
