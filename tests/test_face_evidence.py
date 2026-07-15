"""Tests for canonical face evidence."""

import numpy as np
import pytest

from disha.evidence.schema import EvidenceObject, ReliabilityStatus
from src.face.dataset import FER2013_CLASSES
from src.face.evidence import build_face_evidence


def test_evidence_fields_and_values():
    probs = np.array([0.05, 0.05, 0.05, 0.6, 0.1, 0.1, 0.05])
    evidence = build_face_evidence(
        probs,
        FER2013_CLASSES,
        model_name="test_model",
        calibrated=True,
        reliability=0.8,
    )

    assert isinstance(evidence, EvidenceObject)
    assert evidence.modality.value == "face"
    assert evidence.predicted_emotion == "happy"
    assert evidence.calibrated_confidence == pytest.approx(0.6)
    assert 0.0 < evidence.predictive_entropy < 1.0
    assert evidence.reliability_score == pytest.approx(0.8)
    assert evidence.reliability_status == ReliabilityStatus.RELIABLE
    assert evidence.model_name == "test_model"
    assert sum(evidence.emotion_probabilities.values()) == pytest.approx(1.0, abs=1e-6)


def test_evidence_contains_no_raw_image_data():
    evidence = build_face_evidence(np.full(7, 1.0 / 7.0), FER2013_CLASSES)
    payload = evidence.to_dict()
    for value in payload.values():
        assert not isinstance(value, (bytes, np.ndarray))


def test_uniform_probs_have_high_uncertainty_but_independent_reliability():
    evidence = build_face_evidence(np.full(7, 1.0 / 7.0), FER2013_CLASSES)
    assert evidence.predictive_entropy == pytest.approx(1.0, abs=1e-6)
    assert evidence.reliability_score == pytest.approx(1.0, abs=1e-6)


def test_unavailable_evidence_is_unusable():
    evidence = build_face_evidence(
        np.full(7, 1.0 / 7.0),
        FER2013_CLASSES,
        availability=False,
    )
    assert not evidence.is_usable()


def test_invalid_probs_raise():
    with pytest.raises(ValueError):
        build_face_evidence(np.array([0.5, 0.5]), FER2013_CLASSES)
    with pytest.raises(ValueError):
        build_face_evidence(np.full(7, 0.5), FER2013_CLASSES)
