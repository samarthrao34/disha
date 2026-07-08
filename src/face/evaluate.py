"""Evaluate trained face models on the FER2013 test set.

Usage:
    python -m src.face.evaluate --data-dir data/raw/fer2013 --model-name resnet18
    python -m src.face.evaluate --data-dir data/raw/fer2013 --model-name mobilenet_v3_small

Reports accuracy, macro-F1, weighted-F1, ECE, calibrated ECE when available,
and average per-image inference latency. All numbers come from the actual run.
"""

import argparse
import json
import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from tqdm import tqdm

from .calibration import expected_calibration_error
from .dataset import get_face_dataloaders
from .model import create_face_model
from .temperature_scaling import apply_temperature


def checkpoint_path(model_name: str) -> str:
    return os.path.join("checkpoints", f"face_{model_name}_fer2013.pt")


def temperature_path(model_name: str) -> str:
    return os.path.join("checkpoints", f"face_{model_name}_temperature.pt")


def metrics_path(model_name: str) -> str:
    return os.path.join("experiments", f"face_{model_name}_fer2013_metrics.json")


def confusion_matrix_path(model_name: str) -> str:
    return os.path.join("experiments", f"face_{model_name}_fer2013_confusion_matrix.png")


def classification_report_path(model_name: str) -> str:
    return os.path.join("experiments", f"face_{model_name}_fer2013_classification_report.json")


def measure_latency(model, loader, device, n_samples: int = 100, n_warmup: int = 10) -> float:
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


def load_temperature(path: str):
    if not os.path.isfile(path):
        return None

    payload = torch.load(path, map_location="cpu")
    if isinstance(payload, dict):
        return float(payload["temperature"])
    return float(payload)


def save_confusion_matrix_plot(cm: np.ndarray, class_names, path: str, model_name: str) -> None:
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
    ax.set_title(f"FER2013 test confusion matrix ({model_name})")

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
    parser.add_argument(
        "--model-name",
        type=str,
        default="resnet18",
        choices=["resnet18", "mobilenet_v3_small", "mobilenet_v3_large"],
    )
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ece-bins", type=int, default=15)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[evaluate] Device: {device}")
    print(f"[evaluate] Model: {args.model_name}")

    ckpt_path = args.checkpoint or checkpoint_path(args.model_name)

    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}. Train first.")

    checkpoint = torch.load(ckpt_path, map_location=device)
    class_names = checkpoint["class_names"]
    checkpoint_model_name = checkpoint.get("model_name", args.model_name)

    model = create_face_model(
        model_name=checkpoint_model_name,
        num_classes=len(class_names),
        pretrained=False,
    )
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

    temp_path = temperature_path(checkpoint_model_name)
    temperature = load_temperature(temp_path)
    ece_calibrated = None

    if temperature is not None:
        calibrated_probs = F.softmax(apply_temperature(logits, temperature), dim=1).numpy()
        ece_calibrated = expected_calibration_error(
            calibrated_probs,
            labels,
            n_bins=args.ece_bins,
        )
        print(f"[evaluate] Loaded temperature {temperature:.6f} from {temp_path}")
    else:
        print(f"[evaluate] No temperature file at {temp_path}; skipping calibrated ECE.")

    print("[evaluate] Measuring per-image latency...")
    latency_ms = measure_latency(model, test_loader, device)

    cm = confusion_matrix(labels, preds)
    report = classification_report(
        labels,
        preds,
        target_names=class_names,
        output_dict=True,
        zero_division=0,
    )

    os.makedirs("experiments", exist_ok=True)

    cm_path = confusion_matrix_path(checkpoint_model_name)
    report_path = classification_report_path(checkpoint_model_name)
    m_path = metrics_path(checkpoint_model_name)

    save_confusion_matrix_plot(cm, class_names, cm_path, checkpoint_model_name)

    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2)

    metrics = {
        "dataset": "fer2013",
        "model": checkpoint_model_name,
        "checkpoint": ckpt_path,
        "num_test_samples": int(len(labels)),
        "num_params": int(checkpoint.get("num_params", sum(p.numel() for p in model.parameters()))),
        "trainable_params": int(
            checkpoint.get(
                "trainable_params",
                sum(p.numel() for p in model.parameters() if p.requires_grad),
            )
        ),
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

    with open(m_path, "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2)

    print("\n[evaluate] Results (FER2013 test set):")
    print(f"  Accuracy            : {accuracy:.4f}")
    print(f"  Macro-F1            : {macro_f1:.4f}")
    print(f"  Weighted-F1         : {weighted_f1:.4f}")
    print(f"  ECE uncalibrated    : {ece_uncalibrated:.4f} ({args.ece_bins} bins)")
    if ece_calibrated is not None:
        print(f"  ECE calibrated      : {ece_calibrated:.4f} (T={temperature:.4f})")
    print(f"  Latency per image   : {latency_ms:.2f} ms ({device})")

    print(f"\n[evaluate] Metrics saved to {m_path}")
    print(f"[evaluate] Confusion matrix saved to {cm_path}")
    print(f"[evaluate] Classification report saved to {report_path}")


if __name__ == "__main__":
    main()