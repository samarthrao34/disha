"""Image-file inference adapter for the integrated DISHA demo."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from disha.face.reliability import assess_face_quality, crop_face, detect_largest_face
from face.dataset import build_eval_transform
from face.evidence import build_face_evidence
from face.model import create_face_model
from face.temperature_scaling import apply_temperature


CHECKPOINT = Path("checkpoints/face_resnet18_fer2013.pt")
TEMPERATURE = Path("checkpoints/face_resnet18_temperature.pt")


@lru_cache(maxsize=1)
def load_face_runtime():
    if not CHECKPOINT.is_file():
        raise FileNotFoundError(f"face checkpoint not found: {CHECKPOINT}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(CHECKPOINT, map_location=device)
    model_name = checkpoint.get("model_name", "resnet18")
    model = create_face_model(model_name, len(checkpoint["class_names"]), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    temperature = None
    if TEMPERATURE.is_file():
        payload = torch.load(TEMPERATURE, map_location="cpu")
        temperature = float(payload["temperature"] if isinstance(payload, dict) else payload)
    return model, checkpoint["class_names"], model_name, temperature, device


def _center_square(image: np.ndarray) -> np.ndarray:
    height, width = image.shape[:2]
    size = min(height, width)
    x = (width - size) // 2
    y = (height - size) // 2
    return image[y : y + size, x : x + size]


def build_face_evidence_from_bgr(image: np.ndarray):
    """Build sanitized face evidence from an in-memory OpenCV BGR frame."""
    if image is None or image.size == 0:
        raise ValueError("image cannot be empty")
    face_box = detect_largest_face(image)
    detector_fallback = face_box is None
    face = _center_square(image) if detector_fallback else crop_face(image, face_box)
    quality = assess_face_quality(
        face if detector_fallback else image,
        face_box=None if detector_fallback else face_box,
        assume_cropped_face=detector_fallback,
    )
    metadata = dict(quality.metadata)
    metadata["center_crop_fallback"] = detector_fallback

    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
    tensor = build_eval_transform()(Image.fromarray(rgb)).unsqueeze(0)
    model, class_names, model_name, temperature, device = load_face_runtime()
    with torch.no_grad():
        logits = model(tensor.to(device))
        if temperature is not None:
            logits = apply_temperature(logits, temperature)
        probabilities = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
    return build_face_evidence(
        probabilities,
        class_names,
        model_name=f"{model_name}_fer2013",
        calibrated=temperature is not None,
        reliability=quality.score,
        reliability_status=quality.status,
        quality_metadata=metadata,
    )


def build_face_evidence_from_file(path: str | Path):
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"could not read image: {path}")
    return build_face_evidence_from_bgr(image)
