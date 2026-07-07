"""Evaluate the trained face model on the FER2013 test set.

Usage:
    python -m src.face.evaluate --data-dir data/raw/fer2013

Reports accuracy, macro-F1, weighted-F1, ECE, and average per-image
inference latency, then writes them to
``experiments/face_resnet18_fer2013_metrics.json``.
All numbers come from the actual evaluation run.
"""

import argparse
import json
import os
import time

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

from .calibration import expected_calibration_error
from .dataset import get_face_dataloaders
from .model import create_resnet18_face_model

DEFAULT_CHECKPOINT = os.path.join("checkpoints", "face_resnet18_fer2013.pt")
METRICS_PATH = os.path.join("experiments", "face_resnet18_fer2013_metrics.json")


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

    all_probs, all_labels = [], []
    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            probs = F.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())

    probs = np.concatenate(all_probs)
    labels = np.concatenate(all_labels)
    preds = probs.argmax(axis=1)

    accuracy = float(accuracy_score(labels, preds))
    macro_f1 = float(f1_score(labels, preds, average="macro"))
    weighted_f1 = float(f1_score(labels, preds, average="weighted"))
    ece = expected_calibration_error(probs, labels, n_bins=args.ece_bins)

    print("[evaluate] Measuring per-image latency...")
    latency_ms = measure_latency(model, test_loader, device)

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
        "ece": ece,
        "ece_bins": args.ece_bins,
        "avg_latency_ms_per_image": latency_ms,
        "device": str(device),
    }

    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n[evaluate] Results (FER2013 test set):")
    print(f"  Accuracy          : {accuracy:.4f}")
    print(f"  Macro-F1          : {macro_f1:.4f}")
    print(f"  Weighted-F1       : {weighted_f1:.4f}")
    print(f"  ECE ({args.ece_bins} bins)     : {ece:.4f}")
    print(f"  Latency per image : {latency_ms:.2f} ms ({device})")
    print(f"\n[evaluate] Metrics saved to {METRICS_PATH}")


if __name__ == "__main__":
    main()
