from disha.runtime import DishaEngine
from disha.text.evidence import build_text_evidence


def test_text_evidence_does_not_retain_raw_text():
    raw = "I am worried about work tomorrow"
    evidence = build_text_evidence(raw)
    payload = evidence.to_dict()
    assert raw not in str(payload)
    assert evidence.modality.value == "text"
    assert evidence.quality_metadata["token_count"] == 6


def test_negative_text_selects_explore():
    result = DishaEngine().process_text("I feel sad and worried about work today")
    assert result.action == "explore"
    assert result.safety["approved"] is True


def test_ambiguous_short_text_requests_clarification():
    result = DishaEngine().process_text("hello")
    assert result.action == "clarify"


def test_direct_crisis_indicator_forces_crisis_action():
    result = DishaEngine().process_text("I want to kill myself")
    assert result.action == "provide_crisis_resources"
    assert result.safety["reason"] == "crisis_override"
    assert "emergency" in result.response.lower()
