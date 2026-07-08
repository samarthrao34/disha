"""Temperature scaling for calibrating DISHA face emotion logits."""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim

from src.face.dataset import get_face_dataloaders
from src.face.model import create_resnet18_face_model

CHECKPOINT_PATH = Path("checkpoints/face_resnet18_fer2013.pt")
TEMPERATURE_PATH = Path("checkpoints/face_resnet18_temperature.pt")
DATA_DIR = Path("data/fer2013")


def collect_logits_and_labels(model, dataloader, device):
    """Collect model logits and labels from a dataloader."""
    model.eval()
    logits_list = []
    labels_list = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            logits = model(images)
            logits_list.append(logits)
            labels_list.append(labels)

    return torch.cat(logits_list, dim=0), torch.cat(labels_list, dim=0)


def apply_temperature(logits, temperature):
    """Apply temperature scaling to logits."""
    if torch.is_tensor(temperature):
        t = temperature.clamp_min(1e-6)
    else:
        t = torch.tensor(float(temperature), dtype=logits.dtype, device=logits.device).clamp_min(1e-6)
    return logits / t


def tune_temperature(logits, labels, device):
    """Learn a single scalar temperature with NLL on validation logits."""
    logits = logits.to(device)
    labels = labels.to(device)

    log_temperature = torch.nn.Parameter(torch.zeros(1, device=device))
    optimizer = optim.LBFGS([log_temperature], lr=0.1, max_iter=100)

    def closure():
        optimizer.zero_grad()
        temperature = torch.exp(log_temperature)
        loss = F.cross_entropy(apply_temperature(logits, temperature), labels)
        loss.backward()
        return loss

    optimizer.step(closure)
    return torch.exp(log_temperature).detach()


def _load_model(device):
    model = create_resnet18_face_model(num_classes=7, pretrained=False).to(device)
    state = torch.load(CHECKPOINT_PATH, map_location=device)

    if isinstance(state, dict) and "state_dict" in state:
        model.load_state_dict(state["state_dict"])
    else:
        model.load_state_dict(state)

    model.eval()
    return model


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    _, val_loader, _, _ = get_face_dataloaders(data_dir=str(DATA_DIR))
    model = _load_model(device)

    logits, labels = collect_logits_and_labels(model, val_loader, device)

    initial_nll = F.cross_entropy(logits, labels).item()
    temperature = tune_temperature(logits, labels, device)
    calibrated_logits = apply_temperature(logits, temperature)
    calibrated_nll = F.cross_entropy(calibrated_logits, labels).item()

    TEMPERATURE_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"temperature": temperature.item()}, TEMPERATURE_PATH)

    print(f"initial NLL: {initial_nll:.6f}")
    print(f"calibrated NLL: {calibrated_nll:.6f}")
    print(f"learned temperature: {temperature.item():.6f}")


if __name__ == "__main__":
    main()
