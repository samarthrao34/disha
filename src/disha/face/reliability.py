from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from disha.evidence.schema import ReliabilityStatus


# These are conservative research defaults, not calibrated operational cutoffs.
BLUR_REFERENCE_VARIANCE = 100.0
RELIABLE_THRESHOLD = 0.60
DEGRADED_THRESHOLD = 0.30


@dataclass(frozen=True)
class FaceQualityAssessment:
    score: float
    status: ReliabilityStatus
    metadata: Dict[str, Any]


def variance_of_laplacian(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def mean_luminance(image: np.ndarray) -> float:
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.mean(gray))


def detect_largest_face(image: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    """Detect the largest frontal face with OpenCV's bundled Haar cascade."""
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    # Some OpenCV 5 builds omit the legacy cascade API. Treat detection as
    # unavailable instead of silently assigning a reliable score.
    if not hasattr(cv2, "CascadeClassifier"):
        return None
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(cascade_path)
    if detector.empty():
        raise RuntimeError(f"could not load face detector: {cascade_path}")
    faces = detector.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(40, 40),
    )
    if len(faces) == 0:
        return None
    return tuple(int(value) for value in max(faces, key=lambda box: box[2] * box[3]))


def crop_face(image: np.ndarray, face_box: Tuple[int, int, int, int]) -> np.ndarray:
    x, y, width, height = face_box
    if width <= 0 or height <= 0:
        raise ValueError("face box must have positive width and height")
    image_height, image_width = image.shape[:2]
    x0, y0 = max(0, x), max(0, y)
    x1, y1 = min(image_width, x + width), min(image_height, y + height)
    cropped = image[y0:y1, x0:x1]
    if cropped.size == 0:
        raise ValueError("face box does not intersect the image")
    return cropped


def assess_face_quality(
    image: np.ndarray,
    face_box: Optional[Tuple[int, int, int, int]] = None,
    *,
    assume_cropped_face: bool = False,
) -> FaceQualityAssessment:
    """Estimate blur, lighting, and face coverage without retaining pixels."""
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")

    if face_box is None and not assume_cropped_face:
        face_box = detect_largest_face(image)

    if face_box is None and not assume_cropped_face:
        return FaceQualityAssessment(
            score=0.0,
            status=ReliabilityStatus.UNUSABLE,
            metadata={
                "face_detected": False,
                "detector_available": hasattr(cv2, "CascadeClassifier"),
                "thresholds": "research_defaults",
            },
        )

    region = image if assume_cropped_face else crop_face(image, face_box)
    blur_variance = variance_of_laplacian(region)
    luminance = mean_luminance(region)
    blur_score = float(np.clip(blur_variance / BLUR_REFERENCE_VARIANCE, 0.0, 1.0))
    lighting_score = float(np.clip(1.0 - abs(luminance - 127.5) / 127.5, 0.0, 1.0))

    if assume_cropped_face:
        coverage_score = 1.0
    else:
        _, _, width, height = face_box
        frame_area = float(image.shape[0] * image.shape[1])
        coverage_ratio = (width * height) / frame_area
        # A face covering roughly 10% or more of the frame is sufficient here.
        coverage_score = float(np.clip(coverage_ratio / 0.10, 0.0, 1.0))

    score = 0.35 * blur_score + 0.35 * lighting_score + 0.30 * coverage_score
    if score >= RELIABLE_THRESHOLD:
        status = ReliabilityStatus.RELIABLE
    elif score >= DEGRADED_THRESHOLD:
        status = ReliabilityStatus.DEGRADED
    else:
        status = ReliabilityStatus.UNUSABLE

    return FaceQualityAssessment(
        score=float(score),
        status=status,
        metadata={
            "face_detected": True,
            "face_detection_required": not assume_cropped_face,
            "detector_available": hasattr(cv2, "CascadeClassifier"),
            "blur_variance": blur_variance,
            "blur_score": blur_score,
            "mean_luminance": luminance,
            "lighting_score": lighting_score,
            "coverage_score": coverage_score,
            "thresholds": "research_defaults",
        },
    )
