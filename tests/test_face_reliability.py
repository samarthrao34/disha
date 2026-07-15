import numpy as np

from disha.evidence.schema import ReliabilityStatus
from disha.face.reliability import assess_face_quality


def test_cropped_clear_face_like_image_has_quality_metadata():
    image = np.zeros((100, 100), dtype=np.uint8)
    image[:, ::2] = 200
    result = assess_face_quality(image, assume_cropped_face=True)
    assert 0.0 <= result.score <= 1.0
    assert result.metadata["face_detected"] is True
    assert result.metadata["coverage_score"] == 1.0
    assert result.status in {
        ReliabilityStatus.RELIABLE,
        ReliabilityStatus.DEGRADED,
        ReliabilityStatus.UNUSABLE,
    }


def test_no_detected_face_is_unusable():
    blank = np.zeros((100, 100), dtype=np.uint8)
    result = assess_face_quality(blank)
    assert result.score == 0.0
    assert result.status == ReliabilityStatus.UNUSABLE
    assert result.metadata["face_detected"] is False
