"""FER2013 dataset utilities for the DISHA face emotion pipeline.

Expected layout:
    <data_dir>/train/<class_name>/*.jpg
    <data_dir>/test/<class_name>/*.jpg

Classes: angry, disgust, fear, happy, neutral, sad, surprise
"""

import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

FER2013_CLASSES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = 224


def build_train_transform() -> transforms.Compose:
    """Augmented transform for training images."""
    return transforms.Compose(
        [
            # FER2013 images are grayscale; force 3 identical channels so
            # ImageNet-pretrained ResNet-18 accepts them.
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_eval_transform() -> transforms.Compose:
    """Deterministic transform for validation/test images."""
    return transforms.Compose(
        [
            transforms.Grayscale(num_output_channels=3),
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def get_face_dataloaders(
    data_dir: str,
    batch_size: int = 64,
    val_split: float = 0.1,
    num_workers: int = 2,
    seed: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, List[str]]:
    """Create train/val/test dataloaders for FER2013.

    The validation set is carved out of the ``train`` folder with a
    reproducible split controlled by ``seed``. The ``test`` folder is used
    exclusively for the test loader.

    Returns:
        (train_loader, val_loader, test_loader, class_names)
    """
    train_dir = os.path.join(data_dir, "train")
    test_dir = os.path.join(data_dir, "test")

    for path in (train_dir, test_dir):
        if not os.path.isdir(path):
            raise FileNotFoundError(
                f"Expected dataset folder not found: {path}. "
                "Expected layout: <data_dir>/train/<class>/*.jpg and "
                "<data_dir>/test/<class>/*.jpg"
            )

    # Two views of the same train folder: augmented for training,
    # deterministic for validation.
    train_view = datasets.ImageFolder(train_dir, transform=build_train_transform())
    val_view = datasets.ImageFolder(train_dir, transform=build_eval_transform())
    test_dataset = datasets.ImageFolder(test_dir, transform=build_eval_transform())

    class_names = train_view.classes
    if sorted(class_names) != sorted(FER2013_CLASSES):
        print(
            f"[dataset] Warning: found classes {class_names}, "
            f"expected {FER2013_CLASSES}"
        )

    if not 0.0 < val_split < 1.0:
        raise ValueError(f"val_split must be in (0, 1), got {val_split}")

    n_total = len(train_view)
    n_val = max(1, int(round(n_total * val_split)))
    generator = torch.Generator().manual_seed(seed)
    permutation = torch.randperm(n_total, generator=generator).tolist()
    val_indices = permutation[:n_val]
    train_indices = permutation[n_val:]

    train_subset = Subset(train_view, train_indices)
    val_subset = Subset(val_view, val_indices)

    loader_kwargs = dict(num_workers=num_workers, pin_memory=torch.cuda.is_available())
    train_loader = DataLoader(
        train_subset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), **loader_kwargs
    )
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, **loader_kwargs)

    return train_loader, val_loader, test_loader, class_names
