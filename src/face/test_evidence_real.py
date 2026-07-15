"""Smoke test: build a Face Evidence Object from real model output.

Usage:
    python -m src.face.test_evidence_real --data-dir data/raw/fer2013

Loads one image from the FER2013 test set, runs the trained checkpoint,
converts logits to probabilities (temperature-scaled when a learned
temperature exists), builds the evidence object, prints it as JSON, and
verifies it contains no raw image data. No results are hardcoded.
"""

import argparse
import json
import os
import sys

import cv2
import torch
import torch.nn.functional as F
from torchvision import datasets

from disha.face.reliability import assess_face_quality

from .dataset import build_eval_transform
from .evidence import build_face_evidence
from .model import create_resnet18_face_model
from .temperature_scaling import apply_temperature

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "face_resnet18_fer2013.pt")
TEMPERATURE_PATH = os.path.join("checkpoints", "face_resnet18_temperature.pt")

EXPECTED_KEYS = {
    "modality",
    "emotion_probabilities",
    "predicted_emotion",
    "calibrated_confidence",
    "predictive_entropy",
    "reliability_score",
    "reliability_status",
    "availability_status",
    "timestamp",
    "quality_metadata",
    "model_name",
}

# A serialized evidence object holding only probs + metadata should be tiny.
# Raw image tensors (3x224x224 floats) would blow far past this.
MAX_SERIALIZED_BYTES = 4096


def verify_no_raw_image_data(evidence: dict) -> None:
    """Raise AssertionError if the evidence looks like it embeds image data."""
    assert set(evidence.keys()) == EXPECTED_KEYS, (
        f"Unexpected keys in evidence: {sorted(set(evidence) - EXPECTED_KEYS)}"
    )

    suspicious = {"image", "pixels", "tensor", "array", "raw", "frame", "bytes"}
    for key in evidence:
        assert not any(word in key.lower() for word in suspicious), (
            f"Suspicious key suggests raw image data: {key}"
        )

    serialized = json.dumps(evidence)
    assert len(serialized) <= MAX_SERIALIZED_BYTES, (
        f"Evidence payload is {len(serialized)} bytes; too large for "
        "probs + metadata, may contain raw data"
    )

    probs = evidence["emotion_probabilities"]
    assert isinstance(probs, dict), "emotion_probs must be a dict"
    assert all(isinstance(v, float) for v in probs.values()), (
        "emotion_probs values must be plain floats"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a Face Evidence Object from one real FER2013 test image"
    )
    parser.add_argument("--data-dir", type=str, default=os.path.join("data", "raw", "fer2013"))
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--index", type=int, default=0, help="Test-set sample index")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evidence-test] Device: {device}")

    if not os.path.isfile(args.checkpoint):
        raise FileNotFoundError(
            f"Checkpoint not found: {args.checkpoint}. Train first with src.face.train"
        )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    class_names = checkpoint["class_names"]

    model = create_resnet18_face_model(num_classes=len(class_names), pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    test_dir = os.path.join(args.data_dir, "test")
    if not os.path.isdir(test_dir):
        raise FileNotFoundError(f"Test folder not found: {test_dir}")
    test_dataset = datasets.ImageFolder(test_dir, transform=build_eval_transform())

    if not 0 <= args.index < len(test_dataset):
        raise IndexError(
            f"--index {args.index} out of range for test set of size {len(test_dataset)}"
        )
    image, label = test_dataset[args.index]
    source_path = test_dataset.samples[args.index][0]
    true_label = test_dataset.classes[label]
    print(f"[evidence-test] Sample index {args.index}, true label: {true_label}")

    with torch.no_grad():
        logits = model(image.unsqueeze(0).to(device))

    # Use calibrated probabilities when a learned temperature exists.
    temperature = None
    if os.path.isfile(TEMPERATURE_PATH):
        payload = torch.load(TEMPERATURE_PATH, map_location="cpu")
        temperature = float(payload["temperature"]) if isinstance(payload, dict) else float(payload)
        logits = apply_temperature(logits, temperature)
        print(f"[evidence-test] Applied temperature {temperature:.6f}")
    else:
        print("[evidence-test] No temperature file found; using uncalibrated probs")

    probs = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()

    source_image = cv2.imread(source_path)
    if source_image is None:
        raise RuntimeError(f"Could not read source image for quality assessment: {source_path}")
    # FER2013 images are already face crops, so detection coverage is implicit.
    quality = assess_face_quality(source_image, assume_cropped_face=True)

    evidence_object = build_face_evidence(
        probs=probs,
        class_names=test_dataset.classes,
        model_name="resnet18_fer2013",
        calibrated=temperature is not None,
        availability=True,
        quality_assessment=quality,
    )
    evidence = evidence_object.to_dict()

    print("\n[evidence-test] Face Evidence Object:")
    print(json.dumps(evidence, indent=2))

    try:
        verify_no_raw_image_data(evidence)
    except AssertionError as exc:
        print(f"\n[evidence-test] FAILED: {exc}")
        sys.exit(1)

    print("\n[evidence-test] OK: evidence contains no raw image data")


if __name__ == "__main__":
    main()
