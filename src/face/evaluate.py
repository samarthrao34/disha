"""Evaluate the trained face model on the FER2013 test set.

Usage:
    python -m src.face.evaluate --data-dir data/raw/fer2013

Reports accuracy, macro-F1, weighted-F1, ECE (uncalibrated and, when a
learned temperature exists, calibrated), and average per-image inference
latency. Writes:
    experiments/face_resnet18_fer2013_metrics.json
    experiments/face_resnet18_fer2013_confusion_matrix.png
    experiments/face_resnet18_fer2013_classification_report.json
All numbers come from the actual evaluation run.
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")  # headless-safe backend; must precede pyplot import
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from tqdm import tqdm

from .calibration import expected_calibration_error
from .dataset import get_face_dataloaders
from .model import create_resnet18_face_model
from .temperature_scaling import apply_temperature

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "face_resnet18_fer2013.pt")
TEMPERATURE_PATH = os.path.join("checkpoints", "face_resnet18_temperature.pt")
METRICS_PATH = os.path.join("experiments", "face_resnet18_fer2013_metrics.json")
CONFUSION_MATRIX_PATH = os.path.join(
    "experiments", "face_resnet18_fer2013_confusion_matrix.png"
)
CLASSIFICATION_REPORT_PATH = os.path.join(
    "experiments", "face_resnet18_fer2013_classification_report.json"
)


def measure_latency(model, loader, device, n_samples: int = 100, n_warmup: int = 10) -> float:
    """Average single-image forward-pass latency in milliseconds."""
    model.eval()
    single_images = []
    for images, _ in loader:
        for i in range(images.size(0)):
            single_images.append(images[i : i + 1])
            if len(single_images) >= n_samples + n_warmup:
                break
        if len(single_images) >= n_samples + n_warmup:
            break

    timings = []
    with torch.no_grad():
        for idx, image in enumerate(single_images):
            image = image.to(device)
            if device.type == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            model(image)
            if device.type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            if idx >= n_warmup:
                timings.append(elapsed)
    return float(np.mean(timings) * 1000.0)


def load_temperature(path: str = TEMPERATURE_PATH):
    """Load the learned temperature if it exists, else return None."""
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return float(payload["temperature"])
    return float(payload)


def save_confusion_matrix_plot(cm: np.ndarray, class_names, path: str) -> None:
    """Render and save a confusion matrix heatmap with matplotlib only."""
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    fig.colorbar(im, ax=ax)

    n = len(class_names)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title("FER2013 test confusion matrix (ResNet-18)")

    # Annotate each cell with its count, switching text color for contrast.
    threshold = cm.max() / 2.0 if cm.max() > 0 else 0.5
    for i in range(n):
        for j in range(n):
            ax.text(
                j,
                i,
                f"{cm[i, j]:d}",
                ha="center",
                va="center",
                color="white" if cm[i, j] > threshold else "black",
                fontsize=8,
            )

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate face model on FER2013 test set")
    parser.add_argument("--data-dir", type=str, default=os.path.join("data", "raw", "fer2013"))
    parser.add_argument("--checkpoint", type=str, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")

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

    _, _, test_loader, _ = get_face_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=0.1,
        num_workers=args.num_workers,
        seed=args.seed,
    )

    all_logits, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            all_logits.append(model(images).cpu())
            all_labels.append(labels)

    logits = torch.cat(all_logits, dim=0)
    labels = torch.cat(all_labels, dim=0).numpy()

    probs = F.softmax(logits, dim=1).numpy()
    preds = probs.argmax(axis=1)

    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    weighted_f1 = float(f1_score(labels, preds, average="weighted"))
    ece_uncalibrated = expected_calibration_error(probs, labels, n_bins=args.ece_bins)

    # Calibrated ECE: only if a learned temperature checkpoint exists.
    temperature = load_temperature()
    ece_calibrated = None
    if temperature is not None:
        calibrated_probs = F.softmax(apply_temperature(logits, temperature), dim=1).numpy()
        ece_calibrated = expected_calibration_error(
            calibrated_probs, labels, n_bins=args.ece_bins
        )
        print(f"[evaluate] Loaded temperature {temperature:.6f} from {TEMPERATURE_PATH}")
    else:
        print(
            f"[evaluate] No temperature file at {TEMPERATURE_PATH}; "
            "skipping calibrated ECE. Run src.face.temperature_scaling first."
        )

    print("[evaluate] Measuring per-image latency...")
    latency_ms = measure_latency(model, test_loader, device)

    # Confusion matrix and per-class classification report.
    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels, preds, target_names=class_names, output_dict=True, zero_division=0
    )

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)

    save_confusion_matrix_plot(cm, class_names, CONFUSION_MATRIX_PATH)
    with open(CLASSIFICATION_REPORT_PATH, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    metrics = {
        "dataset": "fer2013",
        "model": "resnet18",
        "checkpoint": args.checkpoint,
        "num_test_samples": int(len(labels)),
        "num_params": int(checkpoint.get("num_params", sum(p.numel() for p in model.parameters()))),
        "class_names": class_names,
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "ece_uncalibrated": ece_uncalibrated,
        "ece_calibrated": ece_calibrated,
        "ece_bins": args.ece_bins,
        "latency_ms_per_image": latency_ms,
        "temperature": temperature,
        "device": str(device),
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n[evaluate] Results (FER2013 test set):")
    print(f"  Accuracy            : {accuracy:.4f}")
    print(f"  Macro-F1            : {macro_f1:.4f}")
    print(f"  Weighted-F1         : {weighted_f1:.4f}")
    print(f"  ECE uncalibrated    : {ece_uncalibrated:.4f} ({args.ece_bins} bins)")
    if ece_calibrated is not None:
        print(f"  ECE calibrated      : {ece_calibrated:.4f} (T={temperature:.4f})")
    print(f"  Latency per image   : {latency_ms:.2f} ms ({device})")
    print(f"\n[evaluate] Metrics saved to {METRICS_PATH}")
    print(f"[evaluate] Confusion matrix saved to {CONFUSION_MATRIX_PATH}")
    print(f"[evaluate] Classification report saved to {CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
