"""Tests for src.face.evidence."""

import numpy as np
import pytest

from src.face.dataset import FER2013_CLASSES
from src.face.evidence import build_face_evidence


def test_evidence_fields_and_values():
    probs = np.array([0.05, 0.05, 0.05, 0.6, 0.1, 0.1, 0.05])
    evidence = build_face_evidence(probs, FER2013_CLASSES, model_name="test_model")

    expected_keys = {
        "modality", "emotion_probs", "predicted_emotion", "confidence",
        "entropy", "reliability", "availability", "timestamp", "model_name",
    }
    assert set(evidence.keys()) == expected_keys
    assert evidence["modality"] == "face"
    assert evidence["predicted_emotion"] == "happy"
    assert evidence["confidence"] == pytest.approx(0.6)
    assert evidence["entropy"] > 0.0
    assert 0.0 <= evidence["reliability"] <= 1.0
    assert evidence["availability"] is True
    assert evidence["model_name"] == "test_model"
    assert sum(evidence["emotion_probs"].values()) == pytest.approx(1.0, abs=1e-6)


def test_evidence_contains_no_raw_image_data():
    probs = np.full(7, 1.0 / 7.0)
    evidence = build_face_evidence(probs, FER2013_CLASSES)
    for value in evidence.values():
        assert not isinstance(value, (bytes, np.ndarray))


def test_uniform_probs_give_max_entropy_low_reliability():
    probs = np.full(7, 1.0 / 7.0)
    evidence = build_face_evidence(probs, FER2013_CLASSES)
    assert evidence["entropy"] == pytest.approx(np.log(7), abs=1e-6)
    assert evidence["reliability"] == pytest.approx(0.0, abs=1e-6)


def test_invalid_probs_raise():
    with pytest.raises(ValueError):
        build_face_evidence(np.array([0.5, 0.5]), FER2013_CLASSES)
    with pytest.raises(ValueError):
        build_face_evidence(np.full(7, 0.5), FER2013_CLASSES)
