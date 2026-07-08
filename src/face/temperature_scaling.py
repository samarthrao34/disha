"""Temperature scaling for calibrating DISHA face emotion logits.

Usage:
    python -m src.face.temperature_scaling --data-dir data/raw/fer2013

Learns a single scalar temperature ``T`` on validation logits by minimizing
NLL, then saves it to ``checkpoints/face_resnet18_temperature.pt``.
No results are hardcoded; all printed numbers come from the actual run.
"""

import argparse
import os

import torch
import torch.nn.functional as F

from .dataset import get_face_dataloaders
from .model import create_resnet18_face_model

CHECKPOINT_PATH = os.path.join("checkpoints", "face_resnet18_fer2013.pt")
TEMPERATURE_PATH = os.path.join("checkpoints", "face_resnet18_temperature.pt")


def collect_logits_and_labels(model, dataloader, device):
    """Run the model over ``dataloader`` and gather logits and labels.

    Returns:
        (logits, labels): tensors of shape (N, num_classes) and (N,).
    """
    model.eval()
    logits_list, labels_list = [], []
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            logits_list.append(model(images))
            labels_list.append(labels.to(device))
    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def apply_temperature(logits, temperature):
    """Scale logits by a scalar temperature (logits / T)."""
    if not torch.is_tensor(temperature):
        temperature = torch.tensor(
            float(temperature), dtype=logits.dtype, device=logits.device
        )
    return logits / temperature.clamp_min(1e-6)


def tune_temperature(logits, labels, device):
    """Learn a single scalar temperature by minimizing NLL on ``logits``.

    Optimizes ``log T`` with L-BFGS so the temperature stays positive.

    Returns:
        A detached 1-element tensor holding the learned temperature.
    """
    logits = logits.detach().to(device)
    labels = labels.to(device)

    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = torch.optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(apply_temperature(logits, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.exp(log_temperature).detach()


def _load_checkpoint(device):
    if not os.path.isfile(CHECKPOINT_PATH):
        raise FileNotFoundError(
            f"Checkpoint not found: {CHECKPOINT_PATH}. "
            "Run src.face.train first."
        )
    return torch.load(CHECKPOINT_PATH, map_location=device)


def _build_model(checkpoint, device):
    """Rebuild the trained ResNet-18 from a src.face.train checkpoint."""
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
        num_classes = len(checkpoint.get("class_names", [])) or 7
    else:
        # Fall back to a raw state_dict checkpoint.
        state_dict = checkpoint
        num_classes = 7

    model = create_resnet18_face_model(num_classes=num_classes, pretrained=False)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate face emotion logits with temperature scaling"
    )
    parser.add_argument(
        "--data-dir", type=str, default=os.path.join("data", "raw", "fer2013")
    )
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[calibrate] Device: {device}")

    checkpoint = _load_checkpoint(device)

    # Recreate the exact validation split used during training so the
    # temperature is tuned on genuinely held-out data.
    train_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    seed = train_args.get("seed", 42)
    val_split = train_args.get("val_split", 0.1)

    _, val_loader, _, class_names = get_face_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=val_split,
        num_workers=args.num_workers,
        seed=seed,
    )
    print(f"[calibrate] Validation batches: {len(val_loader)}")

    model = _build_model(checkpoint, device)

    logits, labels = collect_logits_and_labels(model, val_loader, device)

    initial_nll = F.cross_entropy(logits, labels).item()
    temperature = tune_temperature(logits, labels, device)
    calibrated_logits = apply_temperature(logits, temperature)
    calibrated_nll = F.cross_entropy(calibrated_logits, labels).item()

    os.makedirs(os.path.dirname(TEMPERATURE_PATH), exist_ok=True)
    torch.save(
        {
            "temperature": temperature.item(),
            "initial_nll": initial_nll,
            "calibrated_nll": calibrated_nll,
            "class_names": class_names,
            "seed": seed,
            "val_split": val_split,
        },
        TEMPERATURE_PATH,
    )
    print(f"[calibrate] Saved temperature to {TEMPERATURE_PATH}")

    print(f"initial NLL: {initial_nll:.6f}")
    print(f"calibrated NLL: {calibrated_nll:.6f}")
    print(f"learned temperature: {temperature.item():.6f}")


if __name__ == "__main__":
    main()
