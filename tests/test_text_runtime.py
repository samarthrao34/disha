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


def test_missing_face_file_degrades_instead_of_blocking_text_result(tmp_path):
    missing = tmp_path / "missing.jpg"
    result = DishaEngine().process(
        text="I feel sad and worried about work today",
        image_path=str(missing),
    )
    assert result.action == "explore"
    face = next(item for item in result.evidence if item["modality"] == "face")
    assert face["availability_status"] == "unavailable"
    assert face["quality_metadata"]["failure_reason"] in {
        "ValueError",
        "FileNotFoundError",
    }
    assert str(missing) in face["quality_metadata"]["failure_detail"]


def test_fail_fast_preserves_modality_exception(tmp_path):
    missing = tmp_path / "missing.jpg"
    engine = DishaEngine(continue_on_modality_error=False)
    try:
        engine.process(text="hello", image_path=str(missing))
    except Exception as exc:
        assert str(missing) in str(exc)
    else:
        raise AssertionError("expected missing modality input to raise in fail-fast mode")
