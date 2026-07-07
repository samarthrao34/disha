"""Tests for src.face.dataset using a tiny synthetic FER2013-style dataset."""

import os

import numpy as np
import pytest
import torch
from PIL import Image

from src.face.dataset import FER2013_CLASSES, get_face_dataloaders


@pytest.fixture
def fake_fer2013(tmp_path):
    """Create a minimal grayscale dataset mimicking the FER2013 layout."""
    rng = np.random.default_rng(0)
    for split, n_images in (("train", 4), ("test", 2)):
        for class_name in FER2013_CLASSES:
            class_dir = tmp_path / split / class_name
            class_dir.mkdir(parents=True)
            for i in range(n_images):
                array = rng.integers(0, 256, size=(48, 48), dtype=np.uint8)
                Image.fromarray(array, mode="L").save(class_dir / f"img_{i}.jpg")
    return str(tmp_path)


def test_dataloaders_shapes_and_classes(fake_fer2013):
    train_loader, val_loader, test_loader, class_names = get_face_dataloaders(
        data_dir=fake_fer2013, batch_size=4, val_split=0.25, num_workers=0, seed=42
    )
    assert sorted(class_names) == sorted(FER2013_CLASSES)

    images, labels = next(iter(train_loader))
    # Grayscale inputs must be expanded to 3 channels and resized to 224x224.
    assert images.shape[1:] == (3, 224, 224)
    assert labels.dtype == torch.int64

    n_train = len(train_loader.dataset)
    n_val = len(val_loader.dataset)
    assert n_train + n_val == 4 * len(FER2013_CLASSES)
    assert n_val == round(0.25 * 4 * len(FER2013_CLASSES))
    assert len(test_loader.dataset) == 2 * len(FER2013_CLASSES)


def test_split_is_reproducible(fake_fer2013):
    _, val_a, _, _ = get_face_dataloaders(
        data_dir=fake_fer2013, batch_size=4, val_split=0.25, num_workers=0, seed=123
    )
    _, val_b, _, _ = get_face_dataloaders(
        data_dir=fake_fer2013, batch_size=4, val_split=0.25, num_workers=0, seed=123
    )
    assert list(val_a.dataset.indices) == list(val_b.dataset.indices)


def test_missing_folder_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_face_dataloaders(data_dir=str(tmp_path / "nope"), num_workers=0)
