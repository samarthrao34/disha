"""Train ResNet-18 on FER2013 for the DISHA face pipeline.

Usage:
    python -m src.face.train --data-dir data/raw/fer2013 --epochs 15

Saves the best (by validation macro-F1) checkpoint to
``checkpoints/face_resnet18_fer2013.pt``. No results are hardcoded;
all printed numbers come from the actual run.
"""

import argparse
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from tqdm import tqdm

from .dataset import get_face_dataloaders
from .model import create_resnet18_face_model

CHECKPOINT_PATH = os.path.join("checkpoints", "face_resnet18_fer2013.pt")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_validation(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            all_preds.append(logits.argmax(dim=1).cpu())
            all_labels.append(labels.cpu())
    preds = torch.cat(all_preds).numpy()
    labels = torch.cat(all_labels).numpy()
    avg_loss = total_loss / len(labels)
    macro_f1 = f1_score(labels, preds, average="macro")
    return avg_loss, macro_f1


def main() -> None:
    parser = argparse.ArgumentParser(description="Train ResNet-18 on FER2013")
    parser.add_argument("--data-dir", type=str, default=os.path.join("data", "raw", "fer2013"))
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--val-split", type=float, default=0.1)
    args = parser.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[train] Device: {device}")

    train_loader, val_loader, _, class_names = get_face_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_split=args.val_split,
        num_workers=args.num_workers,
        seed=args.seed,
    )
    print(f"[train] Classes: {class_names}")
    print(
        f"[train] Batches -> train: {len(train_loader)}, val: {len(val_loader)}"
    )

    model = create_resnet18_face_model(num_classes=len(class_names), pretrained=True)
    model.to(device)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"[train] Model parameters: {num_params:,}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val_f1 = -1.0
    os.makedirs(os.path.dirname(CHECKPOINT_PATH), exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_seen = 0
        start = time.time()
        progress = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}", leave=False)
        for images, labels in progress:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * images.size(0)
            n_seen += images.size(0)
            progress.set_postfix(loss=f"{epoch_loss / n_seen:.4f}")

        train_loss = epoch_loss / n_seen
        val_loss, val_f1 = run_validation(model, val_loader, criterion, device)
        elapsed = time.time() - start
        print(
            f"[train] Epoch {epoch:03d} | train_loss={train_loss:.4f} | "
            f"val_loss={val_loss:.4f} | val_macro_f1={val_f1:.4f} | {elapsed:.1f}s"
        )

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": class_names,
                    "epoch": epoch,
                    "val_macro_f1": val_f1,
                    "val_loss": val_loss,
                    "args": vars(args),
                    "num_params": num_params,
                },
                CHECKPOINT_PATH,
            )
            print(
                f"[train] New best val macro-F1 {val_f1:.4f}; "
                f"checkpoint saved to {CHECKPOINT_PATH}"
            )

    print(f"[train] Done. Best val macro-F1: {best_val_f1:.4f}")
    print("[train] Run src.face.evaluate to obtain test metrics.")


if __name__ == "__main__":
    main()
